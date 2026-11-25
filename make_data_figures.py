"""연구 페이지용 데이터 대표 이미지 (GPU 불필요, master에서 실행 가능).
  python make_data_figures.py   → experiments/<EXP_NAME>/analysis/figures/*.png

fig1_cifar10_clean.png      CIFAR-10 test (clean), 클래스별 예시. CIFAR-10-C의 원본이고 clean 정확도 측정에만 쓴다
                            (source = 라벨 있는 train 160장은 fig4)
fig2_cifar10c_corruptions.png  같은 test 이미지의 CIFAR-10-C 15종 (severity 5, 실험에 쓴 강도), 유형별 묶음
fig3_severity.png           탐색용 4종의 severity 1~5 (실험은 5만 사용)
fig4_16shot_subset.png      라벨 데이터: 클래스당 16장 subset (seed 0, 총 160장)
fig5_model_input.png        모델 입력: 32×32 → 224×224 PIL bicubic (CLIP 공식 전처리와 같음)
CIFAR-10-C의 i번째 이미지는 CIFAR-10 test의 i번째 이미지에 손상을 입힌 것이라, 같은 index로 clean과 짝지을 수 있다.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

from config import CFG, CLASSES
from core.data import load_cifar10_clean, c10c_dir, kshot_subset

OUT = f"{CFG['EXP_ROOT']}/{CFG['EXP_NAME']}/analysis/figures"
DPI = 200
GROUPS = [('Noise', ['gaussian_noise', 'shot_noise', 'impulse_noise']),
          ('Blur', ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur']),
          ('Weather', ['snow', 'frost', 'fog', 'brightness']),
          ('Digital', ['contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'])]
NAME = {'gaussian_noise': 'Gaussian', 'shot_noise': 'Shot', 'impulse_noise': 'Impulse', 'defocus_blur': 'Defocus',
        'glass_blur': 'Glass', 'motion_blur': 'Motion', 'zoom_blur': 'Zoom', 'snow': 'Snow', 'frost': 'Frost',
        'fog': 'Fog', 'brightness': 'Brightness', 'contrast': 'Contrast', 'elastic_transform': 'Elastic',
        'pixelate': 'Pixelate', 'jpeg_compression': 'JPEG'}
EXAMPLE_CLASSES = ['horse', 'ship', 'automobile']     # fig2 행으로 쓸 클래스 (각 클래스의 첫 test 이미지)
HIGHLIGHT = '#d1495b'


def corr(name, sev, idx):
    x = np.load(f'{c10c_dir()}/{name}.npy', mmap_mode='r')
    return np.array(x[(sev - 1) * 10000 + np.asarray(idx)])


def show(ax, im, title=None, fs=8):
    ax.imshow(im, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=fs, pad=3)


def frame(ax, color, lw=2.5):
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(color); s.set_linewidth(lw)


def title_above(fig, ax, text, pad_in=0.22, fs=12):
    """제목을 맨 윗줄 axes 바로 위(pad_in 인치)에 둔다 (suptitle 기본 위치는 여백이 크게 남음)."""
    y = ax.get_position().y1 + pad_in / fig.get_figheight()
    fig.text(0.5, y, text, ha='center', va='bottom', fontsize=fs)


def save(fig, name):
    p = f'{OUT}/{name}'
    fig.savefig(p, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('저장:', p)


def fig1(Xte, Yte, n=8):
    fig, axes = plt.subplots(10, n, figsize=(n * 0.8 + 1.2, 10 * 0.8 + 0.6))
    for c, cls in enumerate(CLASSES):
        idx = np.flatnonzero(Yte == c)[:n]
        for j, i in enumerate(idx):
            show(axes[c, j], Xte[i])
        axes[c, 0].set_ylabel(cls, rotation=0, ha='right', va='center', fontsize=10)
    plt.subplots_adjust(wspace=0.06, hspace=0.06)
    title_above(fig, axes[0, 0], 'CIFAR-10 test set (clean, 32×32) — originals of CIFAR-10-C')
    save(fig, 'fig1_cifar10_clean.png')


def fig2(Xte, Yte):
    idx = [int(np.flatnonzero(Yte == CLASSES.index(c))[0]) for c in EXAMPLE_CLASSES]
    cols = [('Clean', None)] + [(NAME[c], c) for _, cs in GROUPS for c in cs]
    imgs = {c: corr(c, CFG['SEVERITY'], idx) for _, c in cols if c}
    fig, axes = plt.subplots(len(idx), len(cols), figsize=(len(cols) * 0.95, len(idx) * 1.0 + 0.1))   # 행 간격이 열 간격과 비슷하게
    for r, i in enumerate(idx):
        for k, (title, c) in enumerate(cols):
            show(axes[r, k], Xte[i] if c is None else imgs[c][r], title if r == 0 else None)
        axes[r, 0].set_ylabel(CLASSES[Yte[i]], rotation=0, ha='right', va='center', fontsize=9)
    for r in range(len(idx)):
        frame(axes[r, 0], '0.35', 1.5)
    plt.subplots_adjust(wspace=0.05, hspace=0.06)
    H = fig.get_figheight()
    k = 1
    for g, cs in GROUPS:                                       # 유형 이름 + 밑줄 (열 제목 바로 위)
        p0, p1 = axes[0, k].get_position(), axes[0, k + len(cs) - 1].get_position()
        y = p0.y1 + 0.24 / H
        fig.text((p0.x0 + p1.x1) / 2, y + 0.04 / H, g, ha='center', va='bottom', fontsize=10, weight='bold')
        fig.add_artist(Line2D([p0.x0 + 0.003, p1.x1 - 0.003], [y, y], transform=fig.transFigure, color='0.3', lw=1))
        k += len(cs)
    title_above(fig, axes[0, 0], f"CIFAR-10-C — 15 corruptions at severity {CFG['SEVERITY']} (same test images)",
                pad_in=0.62)
    save(fig, 'fig2_cifar10c_corruptions.png')


def fig3(Xte, Yte):
    i = int(np.flatnonzero(Yte == CLASSES.index(EXAMPLE_CLASSES[0]))[0])
    corrs = CFG['CORR_SWEEP']
    fig, axes = plt.subplots(len(corrs), 6, figsize=(6 * 1.05 + 1.0, len(corrs) * 1.05 + 0.6))
    for r, c in enumerate(corrs):
        show(axes[r, 0], Xte[i], 'Clean' if r == 0 else None, fs=9)
        for s in range(1, 6):
            show(axes[r, s], corr(c, s, [i])[0], f'Severity {s}' if r == 0 else None, fs=9)
        frame(axes[r, 5], HIGHLIGHT)
        axes[r, 0].set_ylabel(NAME[c], rotation=0, ha='right', va='center', fontsize=10)
    plt.subplots_adjust(wspace=0.06, hspace=0.08)
    title_above(fig, axes[0, 0], 'Corruption severity (experiments use severity 5, red)', pad_in=0.3)
    save(fig, 'fig3_severity.png')


def fig4(Xtr, Ytr, k=16, seed=0):
    idx = kshot_subset(Ytr, k, seed)
    fig, axes = plt.subplots(10, k, figsize=(k * 0.62 + 1.2, 10 * 0.62 + 0.6))
    for c, cls in enumerate(CLASSES):
        ic = idx[Ytr[idx] == c]
        for j, i in enumerate(ic):
            show(axes[c, j], Xtr[i])
        axes[c, 0].set_ylabel(cls, rotation=0, ha='right', va='center', fontsize=10)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    title_above(fig, axes[0, 0], f'Labeled source data: {k} images per class '
                                 f'(subset seed {seed}, {10 * k} images in total)')
    save(fig, 'fig4_16shot_subset.png')


def fig5(Xte, Yte):
    i = int(np.flatnonzero(Yte == CLASSES.index(EXAMPLE_CLASSES[0]))[0])
    big = np.asarray(Image.fromarray(Xte[i]).resize((224, 224), getattr(Image, 'Resampling', Image).BICUBIC))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.3))
    show(axes[0], Xte[i], 'CIFAR-10 image (32×32)', fs=10)
    show(axes[1], big, 'Model input (224×224, PIL bicubic)', fs=10)
    plt.subplots_adjust(wspace=0.25)
    save(fig, 'fig5_model_input.png')


def main():
    os.makedirs(OUT, exist_ok=True)
    Xtr, Ytr, Xte, Yte = load_cifar10_clean()
    fig1(Xte, Yte)
    fig2(Xte, Yte)
    fig3(Xte, Yte)
    fig4(Xtr, Ytr)
    fig5(Xte, Yte)


if __name__ == '__main__':
    main()
