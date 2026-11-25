"""실행 환경: GPU 확인, AMP 설정, seed, 전처리."""
import os, sys, random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import CFG


class Runtime:
    device = 'cpu'
    amp_dtype = None
    use_amp = False
    use_scaler = False


R = Runtime()


def init_runtime(require_gpu=True):
    """서버 규칙(매뉴얼 7장)에 맞춰 GPU 1장만 보이는지 확인하고 AMP를 설정."""
    torch.set_num_threads(CFG['CPU_THREADS'])
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
        if cvd is None or n != 1:
            sys.exit(f'❌ GPU가 {n}장 보입니다 (CUDA_VISIBLE_DEVICES={cvd}). '
                     f'할당받은 번호만 보이게 `export CUDA_VISIBLE_DEVICES=<GPU번호>` 후 다시 실행하세요.')
        R.device = 'cuda'
        cap = torch.cuda.get_device_capability()
        R.amp_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16   # V100(7.0) → fp16
        R.use_amp = True
        R.use_scaler = R.amp_dtype == torch.float16
        print(f'GPU: {torch.cuda.get_device_name()} (CUDA_VISIBLE_DEVICES={cvd}) | capability {cap} '
              f'| AMP {R.amp_dtype} | torch {torch.__version__} (CUDA {torch.version.cuda})', flush=True)
    elif require_gpu:
        sys.exit('❌ GPU를 찾지 못했습니다. master가 아닌 GPU 노드(ssh gpu01/gpu02)에서 실행하세요. '
                 '그래도 안 되면 torch가 cu118 빌드인지 확인하세요 (README의 설치 단계).')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return R


def autocast():
    if R.use_amp:
        return torch.autocast(device_type='cuda', dtype=R.amp_dtype)
    return torch.autocast(device_type='cpu', enabled=False)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def make_scaler():
    if R.use_scaler:
        return torch.amp.GradScaler('cuda', init_scale=CFG['FP16_INIT_SCALE'])
    return None


def optimizer_step(loss, opt, scaler):
    """한 step 수행. fp16 overflow로 step이 건너뛰어졌으면 True."""
    opt.zero_grad(set_to_none=True)
    if scaler is None:
        loss.backward(); opt.step(); return False
    scaler.scale(loss).backward()
    before = scaler.get_scale()
    scaler.step(opt); scaler.update()
    return scaler.get_scale() < before


# ---------------- 전처리 (uint8 32×32 → 224 정규화 텐서) ----------------
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
IN_MEAN   = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IN_STD    = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def u8_to_float(x_u8):
    """uint8 NHWC (numpy/tensor) → float NCHW [0,1] on device"""
    if isinstance(x_u8, np.ndarray):
        x_u8 = torch.from_numpy(np.ascontiguousarray(x_u8))
    return x_u8.to(R.device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.)


_BICUBIC = getattr(Image, 'Resampling', Image).BICUBIC


def resize_224_u8(x_u8):
    """CLIP 공식 preprocess와 같은 PIL bicubic 224 리사이즈 (결과는 uint8로 반올림). x_u8: uint8 NHWC"""
    if torch.is_tensor(x_u8):
        x_u8 = x_u8.numpy()
    return np.stack([np.asarray(Image.fromarray(im).resize((224, 224), _BICUBIC)) for im in x_u8])


def _to_input(x_u8, mean, std):
    if CFG['RESIZE'] == 'pil_bicubic':
        x = u8_to_float(resize_224_u8(x_u8))
    elif CFG['RESIZE'] == 'torch_bicubic':   # server_v100_v1까지의 방식. Source가 논문보다 4종 평균 4%p 낮음
        x = F.interpolate(u8_to_float(x_u8), size=224, mode='bicubic', align_corners=False).clamp_(0, 1)
    else:
        raise ValueError(f"RESIZE={CFG['RESIZE']}")
    return (x - mean.to(x.device)) / std.to(x.device)


def clip_input(x_u8):
    """uint8 NHWC 32×32 → CLIP 입력 (224, CLIP 정규화) on device"""
    return _to_input(x_u8, CLIP_MEAN, CLIP_STD)


def dino_input(x_u8):
    """uint8 NHWC 32×32 → DINOv2 입력 (224, ImageNet 정규화) on device"""
    return _to_input(x_u8, IN_MEAN, IN_STD)
