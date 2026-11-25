"""모델: CLIP ViT-B/16 (OpenAI clip 패키지), DINOv2 ViT-B/14, head 모드 관리."""
import os, types
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as tcp

from config import CFG, CLASSES
from core.runtime import R, autocast


def model_dirs():
    clip_dir = f"{CFG['DATA_DIR']}/clip"
    hub_dir = f"{CFG['DATA_DIR']}/torch_hub"
    os.environ['TORCH_HOME'] = hub_dir           # DINOv2 코드·가중치를 공유 홈에 캐시
    torch.hub.set_dir(f'{hub_dir}/hub')
    return clip_dir, hub_dir


def layernorm_params(module):
    ps = []
    for m in module.modules():
        if isinstance(m, nn.LayerNorm):
            ps += [m.weight, m.bias]
    return ps


def freeze_all(module):
    for p in module.parameters():
        p.requires_grad_(False)


def install_ckpt_switch(clip_model, enabled):
    """이미지 인코더 transformer에 gradient checkpointing 스위치 설치 (수치 결과는 동일)."""
    tr = clip_model.visual.transformer
    tr._use_ckpt = enabled

    def fwd(self, x, *args, **kwargs):
        if self._use_ckpt and torch.is_grad_enabled():
            for blk in self.resblocks:
                x = tcp.checkpoint(blk, x, use_reentrant=False)
            return x
        return self.resblocks(x)
    tr.forward = types.MethodType(fwd, tr)


def set_grad_ckpt(clip_model, enabled):
    clip_model.visual.transformer._use_ckpt = enabled


def grad_ckpt_on(clip_model):
    return getattr(clip_model.visual.transformer, '_use_ckpt', False)


class CLIPClassifier(nn.Module):
    """head_mode
    - text_init : 학습 가능한 head(K×512), 텍스트 프로토타입으로 초기화. TTA 중에는 고정
    - norm_only / zeroshot : 텍스트 인코더 출력이 분류기. BATCLIP·CDIM TTA에서는 텍스트 LN도 업데이트"""

    def __init__(self, clip_model, tokens, head_mode, logit_scale):
        super().__init__()
        assert head_mode in ('text_init', 'norm_only', 'zeroshot')
        self.clip, self.head_mode, self.logit_scale = clip_model, head_mode, logit_scale
        self.register_buffer('tokens', tokens, persistent=False)
        with torch.no_grad():
            t = F.normalize(self._encode_text().float(), dim=-1)
        self.register_buffer('text_init', t.clone())
        self.head = nn.Parameter(t.clone()) if head_mode == 'text_init' else None

    def _encode_text(self):
        return self.clip.encode_text(self.tokens)

    def image_features(self, x):
        return F.normalize(self.clip.encode_image(x).float(), dim=-1)

    def class_weights(self, text_grad=False):
        if self.head is not None:
            return F.normalize(self.head, dim=-1)
        with torch.set_grad_enabled(text_grad and torch.is_grad_enabled()):
            with autocast():
                t = self._encode_text()
            return F.normalize(t.float(), dim=-1)

    def visual_ln(self):
        return layernorm_params(self.clip.visual)

    def text_ln(self):
        return layernorm_params(self.clip.transformer) + layernorm_params(self.clip.ln_final)

    def state_snapshot(self):
        snap = {'visual_ln': [p.detach().clone() for p in self.visual_ln()],
                'text_ln': [p.detach().clone() for p in self.text_ln()]}
        if self.head is not None:
            snap['head'] = self.head.detach().clone()
        return snap

    def load_snapshot(self, snap):
        with torch.no_grad():
            for p, v in zip(self.visual_ln(), snap['visual_ln']): p.copy_(v)
            for p, v in zip(self.text_ln(), snap['text_ln']): p.copy_(v)
            if self.head is not None: self.head.copy_(snap['head'])


class Models:
    clip = None
    tokens = None
    dino = None
    zs_snapshot = None


M = Models()


def load_models():
    import clip
    clip_dir, _ = model_dirs()
    clip_model, _ = clip.load('ViT-B/16', device=R.device, jit=False, download_root=clip_dir)
    clip_model = clip_model.float().eval()
    freeze_all(clip_model)
    install_ckpt_switch(clip_model, CFG['GRAD_CKPT'])
    tokens = clip.tokenize([f'a photo of a {c}.' for c in CLASSES]).to(R.device)
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', trust_repo=True,
                          skip_validation=True).to(R.device).eval()   # 캐시 사용 (GPU 노드에서 인터넷 불필요)
    freeze_all(dino)
    M.clip, M.tokens, M.dino = clip_model, tokens, dino
    M.zs_snapshot = CLIPClassifier(clip_model, tokens, 'zeroshot', CFG['LOGIT_SCALE']).state_snapshot()
    return M


def new_classifier(head_mode):
    """zero-shot 상태의 분류기 (head는 텍스트 프로토타입)."""
    clf = CLIPClassifier(M.clip, M.tokens, head_mode, CFG['LOGIT_SCALE']).to(R.device)
    zs = zs_with_head(clf)
    clf.load_snapshot(zs)
    return clf, zs


def zs_with_head(clf):
    return {**M.zs_snapshot, **({'head': clf.text_init.clone()} if clf.head is not None else {})}
