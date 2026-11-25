"""Source 학습 (clean CIFAR-10 subset): 이미지 LayerNorm (+ head)."""
import os, math, time
import numpy as np
import torch
import torch.nn.functional as F

from config import CFG
from core.runtime import R, autocast, set_seed, make_scaler, optimizer_step, clip_input, dino_input
from core.models import M, new_classifier, freeze_all
from core.data import stratified_stride_subset, kshot_subset, ratio_str
from core.methods import cdim_ensemble_logits


class Data:
    Xtr = Ytr = Xte = Yte = None


D = Data()


def load_clean():
    from core.data import load_clean_data          # config.DATASET에 따라 CIFAR-10 또는 CIFAR-100
    D.Xtr, D.Ytr, D.Xte, D.Yte = load_clean_data()
    return D


# ---------------- 경로 ----------------
def exp_dir():
    return f"{CFG['EXP_ROOT']}/{CFG['EXP_NAME']}"


def ckpt_dir(kind):
    """CLIP 체크포인트는 테스트 손상·프로토콜과 무관 → SEVERITY·PROTOCOL이 달라도 기본 실험 폴더(CKPT_EXP_NAME)의 것을 쓴다."""
    d = f"{CFG['EXP_ROOT']}/{CFG['CKPT_EXP_NAME']}/source_ckpt" + ('' if kind == 'A' else '_kshot')
    os.makedirs(d, exist_ok=True)
    return d


def _load(path):
    return torch.load(path, map_location=R.device, weights_only=False)


# ---------------- 학습 ----------------
_AUGMIX = None


def augmix_batch(x_u8_nhwc):
    """이미지마다 독립적인 AugMix(all_ops=False): brightness·contrast·color·sharpness 제외 (CIFAR-10-C와 비중첩)."""
    global _AUGMIX
    if _AUGMIX is None:
        from torchvision.transforms import AugMix
        _AUGMIX = AugMix(severity=3, mixture_width=3, chain_depth=-1, alpha=1.0, all_ops=False)
    t = torch.from_numpy(np.ascontiguousarray(x_u8_nhwc)).permute(0, 3, 1, 2)
    return torch.stack([_AUGMIX(img) for img in t]).permute(0, 2, 3, 1).numpy()


def crop_flip_u8(x, rng, pad=4):
    """CIFAR 표준 증강(zero-pad 4 + random crop + hflip)을 32×32 uint8에서 이미지마다 독립으로. x: uint8 NHWC"""
    B, H, W, _ = x.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    i = rng.randint(0, 2 * pad + 1, B)
    j = rng.randint(0, 2 * pad + 1, B)
    out = np.stack([xp[b, i[b]:i[b] + H, j[b]:j[b] + W] for b in range(B)])
    flip = rng.rand(B) < 0.5
    out[flip] = out[flip][:, :, ::-1]
    return out


_TA = None


def trivial_batch(x_u8_nhwc):
    """A2: 이미지마다 독립적인 TrivialAugmentWide에서 brightness·color·contrast·sharpness를 뺀 것
    (남는 연산: identity, autocontrast, equalize, posterize, solarize, rotate, shear, translate)."""
    global _TA
    if _TA is None:
        from torchvision.transforms import TrivialAugmentWide

        class SafeTrivialAugment(TrivialAugmentWide):
            def _augmentation_space(self, num_bins):
                space = super()._augmentation_space(num_bins)
                return {k: v for k, v in space.items() if k not in ('Brightness', 'Color', 'Contrast', 'Sharpness')}
        _TA = SafeTrivialAugment()
    t = torch.from_numpy(np.ascontiguousarray(x_u8_nhwc)).permute(0, 3, 1, 2)
    return torch.stack([_TA(img) for img in t]).permute(0, 2, 3, 1).numpy()


def cutmix_apply(x, pm, lam, rng):
    """A4 CutMix: x[pm]의 사각형(면적 비율 약 1−λ)을 붙이고, 실제 면적으로 λ를 다시 계산."""
    H, W = x.shape[-2:]
    r = math.sqrt(1 - lam)
    h, w = int(H * r), int(W * r)
    cy, cx = rng.randint(H), rng.randint(W)
    y1, y2 = max(cy - h // 2, 0), min(cy + h // 2, H)
    x1, x2 = max(cx - w // 2, 0), min(cx + w // 2, W)
    x = x.clone()
    x[:, :, y1:y2, x1:x2] = x[pm][:, :, y1:y2, x1:x2]
    return x, 1 - (y2 - y1) * (x2 - x1) / (H * W)


def jsd_loss(l0, l1, l2):
    """A3 AugMix의 Jensen-Shannon consistency (원본·AugMix 2개의 예측 분포)."""
    ps = [l.softmax(1) for l in (l0, l1, l2)]
    log_m = (sum(ps) / 3).clamp(1e-7, 1).log()
    return sum(F.kl_div(log_m, p, reduction='batchmean') for p in ps) / 3


def relation_align_loss(C, Dz):
    """D3: 배치 유사도 행렬 CCᵀ와 DDᵀ의 비대각 원소를 각각 표준화한 뒤 MSE (D 쪽은 gradient 없음)."""
    off = ~torch.eye(C.shape[0], dtype=torch.bool, device=C.device)
    z = lambda v: (v - v.mean()) / (v.std() + 1e-6)
    return F.mse_loss(z((C @ C.t())[off]), z((Dz @ Dz.t())[off]).detach())


@torch.no_grad()
def dino_train_features(x_u8):
    """source 학습 배치의 DINOv2 feature (frozen, 정규화). D1·D3용"""
    with autocast():
        f = M.dino(dino_input(x_u8))
    return F.normalize(f.float(), dim=-1)


def train_source(clf, x_u8, y, n_iters, seed, aug='none', mixup=0.0, drop_last=False,
                 cutmix=0.0, jsd=0.0, src_loss='ce', align=0.0):
    """AdamW + warmup/cosine. 이미지 LayerNorm (+ head)만 학습.
    aug     : none | basic(A1 crop+flip) | ta(A2 = A1 + safe TrivialAugment) | augmix(A3 = A1 + AugMix)
    mixup   : A4 Mixup Beta(α,α),  cutmix: A4 CutMix Beta(α,α)  (0이면 끔)
    jsd     : A3 AugMix JSD consistency 가중치 (0이면 끔, AugMix 논문 값 12)
    src_loss: 'ce' = CLIP logit에 CE,  'ens' = D1 CDIM 앙상블 logit Z에 CE
    align   : D3 CLIP·DINO 배치 유사도 정렬 loss 가중치 λ (0이면 끔)
    drop_last=False는 Phase A(Colab 노트북과 동일), k-shot은 True(배치 크기 일정)."""
    need_dino = src_loss == 'ens' or align > 0
    assert aug in ('none', 'basic', 'ta', 'augmix') and src_loss in ('ce', 'ens')
    assert not (mixup > 0 and cutmix > 0), 'mixup과 cutmix는 함께 쓰지 않음'
    assert not (jsd > 0 and (aug != 'augmix' or mixup > 0 or cutmix > 0)), 'JSD는 aug=augmix 단독으로'
    assert not (need_dino and (mixup > 0 or cutmix > 0 or jsd > 0)), 'D1·D3는 섞기 증강·JSD 없이'
    set_seed(seed)
    freeze_all(clf.clip)
    groups = [{'params': clf.visual_ln(), 'weight_decay': 0.0}]
    if clf.head is not None:
        groups.append({'params': [clf.head], 'weight_decay': CFG['SRC_WD_HEAD']})
    for g in groups:
        for p in g['params']: p.requires_grad_(True)
    opt = torch.optim.AdamW(groups, lr=CFG['SRC_LR'])
    warm = max(1, int(CFG['SRC_WARMUP_FRAC'] * n_iters))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: (t + 1) / warm if t < warm else 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, n_iters - warm))))
    scaler = make_scaler()
    gen_np = np.random.RandomState(seed)
    gen_aug = np.random.RandomState(seed + 10007)   # crop·flip 전용 (perm·mixup 난수열과 분리)
    n = len(y); bs = min(CFG['SRC_BATCH'], n)
    y_t = torch.from_numpy(y)
    it, skipped, log = 0, 0, []
    if R.device == 'cuda': torch.cuda.synchronize()
    t0 = time.time()
    try:
        while it < n_iters:
            perm = gen_np.permutation(n)
            last = n - bs + 1 if drop_last else n
            for s in range(0, last, bs):
                if it >= n_iters: break
                idx = perm[s:s + bs]
                xb = x_u8[idx]                              # uint8 32×32
                if jsd > 0:     # AugMix 논문 방식: crop·flip 한 번 → [원본, AugMix, AugMix]
                    xb = crop_flip_u8(xb, gen_aug)
                    xb = np.concatenate([xb, augmix_batch(xb), augmix_batch(xb)])
                else:
                    if aug == 'augmix':
                        xb = augmix_batch(xb)
                    elif aug == 'ta':
                        xb = trivial_batch(xb)
                    if aug != 'none':
                        xb = crop_flip_u8(xb, gen_aug)
                x = clip_input(xb)                          # 224 리사이즈 + 정규화
                Dz = dino_train_features(xb) if need_dino else None
                yb = y_t[idx].to(R.device)
                lam, pm = 1.0, None
                if mixup > 0:   # 정규화는 affine이라 정규화 뒤에 섞어도 섞은 뒤 정규화한 것과 같음
                    lam = float(gen_np.beta(mixup, mixup))
                    pm = torch.from_numpy(gen_np.permutation(len(idx))).to(R.device)
                    x = lam * x + (1 - lam) * x[pm]
                elif cutmix > 0:
                    lam = float(gen_np.beta(cutmix, cutmix))
                    pm = torch.from_numpy(gen_np.permutation(len(idx))).to(R.device)
                    x, lam = cutmix_apply(x, pm, lam, gen_np)
                with autocast():
                    f = clf.image_features(x)
                W = clf.class_weights()
                if src_loss == 'ens':
                    logits = cdim_ensemble_logits(f, Dz, W, CFG['CDIM_CLIP_TAU_INV'], CFG['CDIM_DINO_TAU_INV'])
                else:
                    logits = clf.logit_scale * f @ W.t()
                if jsd > 0:
                    l0, l1, l2 = logits.chunk(3)
                    loss = F.cross_entropy(l0, yb) + jsd * jsd_loss(l0, l1, l2)
                elif pm is not None:
                    loss = lam * F.cross_entropy(logits, yb) + (1 - lam) * F.cross_entropy(logits, yb[pm])
                else:
                    loss = F.cross_entropy(logits, yb)
                if align > 0:
                    loss = loss + align * relation_align_loss(f, Dz)
                skipped += optimizer_step(loss, opt, scaler)
                sched.step(); it += 1
                if it % 50 == 0 or it == n_iters:
                    log.append((it, loss.item()))
                    print(f'    iter {it}/{n_iters} loss {loss.item():.4f}', flush=True)
    finally:
        for g in groups:
            for p in g['params']: p.requires_grad_(False)
    if R.device == 'cuda': torch.cuda.synchronize()
    return dict(train_time_s=time.time() - t0, src_skipped=skipped, loss_log=log, batch=bs)


@torch.no_grad()
def eval_clean(clf, bs=500):
    W = clf.class_weights()
    correct = 0
    for s in range(0, len(D.Yte), bs):
        x = clip_input(D.Xte[s:s + bs])
        with autocast():
            f = clf.image_features(x)
        correct += ((f @ W.t()).argmax(1).cpu() == torch.from_numpy(D.Yte[s:s + bs])).sum().item()
    return 100.0 * correct / len(D.Yte)


# ---------------- Phase A: 비율 ----------------
def n_source_iters(n_train, budget):
    per_epoch_full = math.ceil(50000 / CFG['SRC_BATCH'])
    per_epoch = math.ceil(n_train / CFG['SRC_BATCH'])
    if budget == 'same_epochs':
        return CFG['SRC_EPOCHS_FULL'] * per_epoch
    if budget == 'same_iters':
        return CFG['SRC_EPOCHS_FULL'] * per_epoch_full
    raise ValueError(budget)


def ckpt_path_A(head_mode, budget, ratio, seed):
    return f"{ckpt_dir('A')}/{head_mode}_{CFG['SRC_AUG']}_{budget}_r{int(round(1/ratio))}_s{seed}.pt"


def get_source_model(head_mode, budget, ratio, seed):
    clf, _ = new_classifier(head_mode)
    if head_mode == 'zeroshot':
        return clf, dict(n_train=0, n_iters=0, train_time_s=0.0, src_clean_acc=eval_clean(clf))
    path = ckpt_path_A(head_mode, budget, ratio, seed)
    if os.path.exists(path):
        ck = _load(path); clf.load_snapshot(ck['snap']); info = ck['info']
    else:
        idx = stratified_stride_subset(D.Ytr, ratio, seed)
        n_iters = n_source_iters(len(idx), budget)
        print(f'  [source train] {head_mode} {budget} ratio={ratio_str(ratio)} seed={seed} n={len(idx)} iters={n_iters}', flush=True)
        info = train_source(clf, D.Xtr[idx], D.Ytr[idx], n_iters, seed, CFG['SRC_AUG'])
        info.update(n_train=len(idx), n_iters=n_iters, src_clean_acc=eval_clean(clf))
        torch.save({'snap': clf.state_snapshot(), 'info': info}, path)
        print(f"  학습 시간 {info['train_time_s']/60:.1f}분", flush=True)
    print(f"  source clean acc = {info['src_clean_acc']:.2f}%", flush=True)
    return clf, info


# ---------------- k-shot ----------------
def wise_snapshot(trained, zeroshot, alpha):
    """WiSE-FT: θ = (1-α)·θ_zs + α·θ_ft. head의 zero-shot 값은 텍스트 프로토타입."""
    mix = lambda a, b: [(1 - alpha) * z + alpha * t for z, t in zip(a, b)]
    out = {'visual_ln': mix(zeroshot['visual_ln'], trained['visual_ln']),
           'text_ln': [t.clone() for t in trained['text_ln']]}
    if 'head' in trained:
        out['head'] = (1 - alpha) * zeroshot['head'] + alpha * trained['head']
    return out


def ckpt_path_k(j):
    """source 학습에 영향을 주는 설정만 이름에 넣음 (wise·cdim_m은 학습 후 단계라 같은 체크포인트를 재사용)."""
    extra = (f"_cm{j['cutmix']:g}" if j['cutmix'] else '') + (f"_jsd{j['jsd']:g}" if j['jsd'] else '') \
        + ('_ens' if j['src_loss'] == 'ens' else '') + (f"_al{j['align']:g}" if j['align'] else '')
    return f"{ckpt_dir('k')}/{j['head_mode']}_k{j['shots']}_it{j['iters']}_{j['aug']}_mx{j['mixup']:g}{extra}_s{j['seed']}.pt"


def get_source_model_k(j):
    clf, zs = new_classifier(j['head_mode'])
    path = ckpt_path_k(j)
    if j['iters'] == 0:      # CLIP 학습 없음: zero-shot 그대로 (head_mode='zeroshot'이면 sanity와 같은 모델). D4 등 학습 후 단계만 적용
        trained, info = zs, dict(n_train=0, n_iters=0, train_time_s=0.0)
    elif os.path.exists(path):
        ck = _load(path); trained, info = ck['snap'], ck['info']
    else:
        idx = kshot_subset(D.Ytr, j['shots'], j['seed'])
        print(f"  [source train] {j['head_mode']} {j['shots']}-shot n={len(idx)} iters={j['iters']} aug={j['aug']} "
              f"mixup={j['mixup']:g} cutmix={j['cutmix']:g} jsd={j['jsd']:g} loss={j['src_loss']} "
              f"align={j['align']:g} seed={j['seed']}", flush=True)
        info = train_source(clf, D.Xtr[idx], D.Ytr[idx], j['iters'], j['seed'], j['aug'], j['mixup'], drop_last=True,
                            cutmix=j['cutmix'], jsd=j['jsd'], src_loss=j['src_loss'], align=j['align'])
        info.update(n_train=len(idx), n_iters=j['iters'])
        trained = clf.state_snapshot()
        torch.save({'snap': trained, 'info': info}, path)
        print(f"  학습 시간 {info['train_time_s']/60:.1f}분", flush=True)
    clf.load_snapshot(wise_snapshot(trained, zs, j['wise']) if j['wise'] != 1.0 else trained)
    info = dict(info, src_clean_acc=eval_clean(clf))
    print(f"  source clean acc (wise={j['wise']:g}) = {info['src_clean_acc']:.2f}%", flush=True)
    return clf, info
