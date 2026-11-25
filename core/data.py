"""데이터: clean CIFAR-10/100, CIFAR-10-C/100-C, subset 선택, 테스트 스트림 순서.
데이터셋은 config.DATASET(환경 변수 DATASET, 기본 cifar10)으로 고른다 (2026-11-24 확장 실험)."""
import os, glob, sys
import numpy as np

from config import CFG, CLASSES

_C_NAME = {'cifar10': 'CIFAR-10-C', 'cifar100': 'CIFAR-100-C'}


def clean_root():
    return f"{CFG['DATA_DIR']}/{CFG['DATASET']}"          # cifar10 → ~/cdim_data/cifar10 (예전과 같음)


def corruption_dir():
    name = _C_NAME[CFG['DATASET']]
    hits = glob.glob(f"{CFG['DATA_DIR']}/**/{name}/labels.npy", recursive=True)
    if not hits and CFG['DATASET'] == 'cifar10':          # 예전 동작 유지 (폴더 이름이 다르게 풀린 경우)
        hits = glob.glob(f"{CFG['DATA_DIR']}/**/labels.npy", recursive=True)
    if not hits:
        sys.exit(f"❌ {name}를 찾지 못했습니다 ({CFG['DATA_DIR']}). "
                 f"먼저 `python prepare_data.py --dataset {CFG['DATASET']}`를 실행하세요.")
    return os.path.dirname(hits[0])


def load_clean_data():
    import torchvision
    ds = torchvision.datasets.CIFAR10 if CFG['DATASET'] == 'cifar10' else torchvision.datasets.CIFAR100
    try:
        tr = ds(clean_root(), train=True, download=False)
        te = ds(clean_root(), train=False, download=False)
    except RuntimeError:
        sys.exit(f"❌ {CFG['DATASET']}가 없습니다. 먼저 `python prepare_data.py --dataset {CFG['DATASET']}`를 실행하세요.")
    if CFG['DATASET'] == 'cifar100':                       # 프롬프트 클래스 이름이 라벨 순서와 맞는지
        names = [c.replace('_', ' ') for c in tr.classes]
        assert names == CLASSES, f'CIFAR-100 클래스 순서 불일치: {names[:5]}... vs {CLASSES[:5]}...'
    return (np.asarray(tr.data), np.asarray(tr.targets, dtype=np.int64),
            np.asarray(te.data), np.asarray(te.targets, dtype=np.int64))


def load_corruption(name, severity):
    """CIFAR-10-C·100-C .npy는 severity 1~5가 10,000장씩 순서대로 쌓여 있음."""
    d = corruption_dir()
    lo, hi = (severity - 1) * 10000, severity * 10000
    x = np.load(f'{d}/{name}.npy', mmap_mode='r')[lo:hi]
    y = np.load(f'{d}/labels.npy')[lo:hi].astype(np.int64)
    assert len(y) == 10000 and y.max() < len(CLASSES), (name, len(y), y.max())
    return np.array(x, copy=True), y


# 예전 이름 (다른 스크립트 호환). 이름과 달리 지금 DATASET을 따른다.
cifar10_root, c10c_dir, load_cifar10_clean = clean_root, corruption_dir, load_clean_data


def stratified_stride_subset(labels, ratio, offset):
    """[Phase A] 클래스별 인덱스를 stride 간격으로 선택. 같은 offset이면 nested, 클래스 균형 보장."""
    stride = int(round(1 / ratio))
    assert abs(stride * ratio - 1) < 1e-9, f'ratio {ratio}는 1/정수여야 합니다'
    off = offset % stride
    idx = [np.flatnonzero(labels == c)[off::stride] for c in np.unique(labels)]
    return np.sort(np.concatenate(idx))


def kshot_subset(labels, k, seed):
    """[k-shot] 클래스별 seed 고정 셔플 후 앞에서 k장 → 같은 seed면 1⊂2⊂4⊂8⊂16 (CoOp 방식)."""
    out = []
    for c in np.unique(labels):
        rng = np.random.RandomState(seed * 1000 + int(c))
        out.append(rng.permutation(np.flatnonzero(labels == c))[:k])
    return np.sort(np.concatenate(out))


def seeds_for(ratio):
    return CFG['SEEDS_MANY'] if ratio <= CFG['MANY_SEEDS_BELOW'] + 1e-12 else [0]


def stream_order(n):
    if CFG['STREAM_SHUFFLE']:
        return np.random.RandomState(CFG['STREAM_SEED']).permutation(n)
    return np.arange(n)


def ratio_str(r):
    r = float(r)
    return '1' if r == 1 else f'1/{int(round(1 / r))}'
