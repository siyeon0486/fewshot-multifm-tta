"""연구 페이지용 모델 그림: ours = 16장 CLIP 미세조정 + Few-shot DINO head + DINO LayerNorm 적응 + CDIM (2026-11-24 갱신).
논문에서 흔히 쓰는 모양(사다리꼴 인코더, 🔥/❄ 아이콘, ×·⊕, 수평·수직으로 꺾은 연결선)으로 그리고 글자는 모듈 이름만 둔다.
입력·출력 칸에는 실제 이미지와 실제 예측을 넣는다. head 구성은 config.CHOSEN_HEAD(pick_ours.py가 고름)를 따른다.
기존 모델(16장 미세조정 CLIP + CDIM) 대비 새로 들어간 부분(Class Mean·Prototypes·Few-shot DINO Head·adapted DINOv2)은
강조색(2026-11-25부터 초록, draw.io 팔레트)으로 칠하고, 모듈 안에는 "(Ours)"를 쓰지 않는다 (사용자 2026-11-24). 범례의 "Ours"가 강조색을 설명한다.
  python fig_model.py [--recompute] [--out 경로]   → experiments/<EXP_NAME>/analysis/figures/fig6_model_ours.png

그림의 구조 (코드와 같음, core.methods.run_online):
  - CLIP: 라벨 160장으로 미리 미세조정(이미지 LayerNorm + 텍스트로 초기화한 분류기). TTA 중에는 이미지 LayerNorm만 업데이트.
  - DINOv2 두 갈래: CDIM의 cross-covariance 분기는 원래(source) DINO 특징, head는 TTA 중 LayerNorm을 적응시키는 DINO 특징
    (dlnh). 적응한 DINO는 같은 CDIM 손실의 gradient를 head를 거쳐 받는다.
  - head: 라벨 160장의 DINO 특징 클래스 평균(프로토타입). 구성(patch 평균·중심화·좌우반전)은 CHOSEN_HEAD에 따라 작게 표시.
예측(--recompute, GPU): 손상 4종(유형별 1개)마다 subset 0의 ours로 online TTA를 끝까지 돌린 실제 최종 예측(10,000장)과
  zero-shot CLIP(TTA 없음) 예측. 예시 이미지는 ours가 맞히고 zero-shot CLIP이 틀린 이미지 중 클래스가 겹치지 않는 것을
  스트림 뒤쪽 절반(적응이 진행된 뒤)에서 고른다. 스트림 전체 정확도도 함께 저장한다(RESULTS.md 기록).
결과는 model_fig_preds.json에 저장하고, 다시 그릴 때는 재사용한다(--recompute로 다시 계산).
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle, Rectangle, PathPatch
from matplotlib.path import Path

from config import CFG, CLASSES, CHOSEN_HEAD, J, FT_ITERS
from core.data import load_corruption, kshot_subset, stream_order

OUT = f"{CFG['EXP_ROOT']}/{CFG['EXP_NAME']}/analysis/figures"
PRED_JSON = f'{OUT}/model_fig_preds.json'
CORRS = [('gaussian_noise', 'Gaussian noise'), ('defocus_blur', 'Defocus blur'), ('snow', 'Snow'),
         ('contrast', 'Contrast')]                         # Noise · Blur · Weather · Digital 각 1개 (탐색용 4종)
SEED = 0                                                   # 라벨 subset


def compute_preds():
    """ours(CHOSEN_HEAD, subset 0)로 online TTA를 끝까지 돌린 최종 예측. head 인자는 core.runner와 같은 함수로 만든다."""
    import torch
    from core.runtime import init_runtime, R
    from core.models import load_models, new_classifier
    from core.methods import run_online, get_corruption, get_head_query, clip_image_feats
    from core.runner import _extra_kw_fn
    from core import source as S
    init_runtime(require_gpu=True)
    S.load_clean()
    load_models()
    j = J(16, SEED, iters=FT_ITERS, proto=1.0, proto_src=CHOSEN_HEAD)
    j.pop('methods', None)
    clf, info = S.get_source_model_k(j)
    base = clf.state_snapshot()
    extra = _extra_kw_fn(j)(clf)
    hflags = extra.pop('head_flags', None)
    zs_clf, zs_snap = new_classifier('zeroshot')
    order = stream_order(10000)
    late = set(order[len(order) // 2:].tolist())            # 스트림 뒤쪽 절반
    out = dict(device=R.device, seed=SEED, head=CHOSEN_HEAD, clean_acc_ours_clip=info['src_clean_acc'], corrs={})
    used = set()                                           # 예시 클래스가 겹치지 않게
    for c, _ in CORRS:
        x, y, d = get_corruption(c)
        zs_clf.load_snapshot(zs_snap)                      # CLIP 모델을 공유하므로 zero-shot 상태로 되돌린 뒤 계산
        with torch.no_grad():
            C = clip_image_feats(zs_clf, x).float()
            yc = (C @ zs_clf.class_weights().float().t()).argmax(1).cpu().numpy()
        hkw = {}
        if hflags:
            dln = 'all' if 'dln' in hflags else 'head' if 'dlnh' in hflags else None
            hkw = dict(head_q=None if dln else get_head_query(c, hflags), head_ctr='ctr' in hflags, dino_ln=dln,
                       head_feat='cp' if 'cp' in hflags else 'cls', head_qf='qf' in hflags)
        r = run_online(clf, 'cdim', base, x, y, order, d, **extra, **hkw, keep_pred=True)
        yo = r['pred']
        good = [i for i in np.flatnonzero((yo == y) & (yc != y)) if i in late]
        cand = [i for i in good if y[i] not in used] or good or list(np.flatnonzero(yo == y))
        pick = int(cand[0]); used.add(int(y[pick]))
        out['corrs'][c] = dict(index=pick, label=int(y[pick]), ours=int(yo[pick]), clip=int(yc[pick]),
                               acc_ours=float(r['acc']), acc_clip_noTTA=float((yc == y).mean() * 100))
        print(c, out['corrs'][c], flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(PRED_JSON, 'w') as f:
        json.dump(out, f, indent=1)
    return out


# ---------------- 그리기 (1 unit = 0.1 inch) ----------------
# 색 (2026-11-25 사용자 "다른 학회 논문들이 많이 사용하는 색으로" → "너무 연한데, 톤을 바꿀 수 없나?"):
#   학회 논문 모델 그림에서 흔한 역할별 색(파랑 = 고정, 주황 = 학습, 초록 = 새로 추가, 노랑 = 손실, 회색 = 그 밖)을
#   draw.io 기본 파스텔보다 한 단계 진하게. 글자는 검정 그대로 읽히는 중간 톤. (채움, 테두리)
#   → "색 조합이 이상한데 다른 거 없나": 색을 2~3개로 줄인 후보를 PALETTES에 두고 --palette로 고른다.
#   각 항목: (채움, 테두리). group = CLIP 이미지 인코더 점선, grad = gradient 점선, snow = 눈송이 아이콘
PALETTES = {
    # 처음 판(11-24): 파랑 고정, 주황 학습, 분홍 새로 추가, 연노랑 손실, 회색
    'pink': dict(TRAIN=('#fde3cf', '#e07b39'), FROZEN=('#dde8f6', '#4a6fa5'), OURS=('#fbdde3', '#c8374f'),
                 PLAIN=('#efefef', '#6b6b6b'), LOSS=('#fff1cc', '#c79a2b'), GRAD='#e07b39', CLIP_GROUP='#e3a77a',
                 SNOW='#1e88e5'),
    # 사용자 11-25 "ours_pink가 좋긴한데, 색 톤을 ours랑 비슷하게": 'pink'의 색 조합 그대로, 톤만 'mid'만큼 진하게
    'pink_mid': dict(TRAIN=('#fcc9a3', '#d0661f'), FROZEN=('#b9cff0', '#3a5f99'), OURS=('#f5b3c0', '#b8283f'),
                     PLAIN=('#e3e3e3', '#5c5c5c'), LOSS=('#ffe08f', '#b8891a'), GRAD='#d0661f', CLIP_GROUP='#de8f55',
                     SNOW='#1565c0'),
    'mid': dict(TRAIN=('#ffd0a1', '#cc7a00'), FROZEN=('#b3cdf0', '#3d6db5'), OURS=('#b5dfa8', '#4c9a3c'),
                PLAIN=('#e6e6e6', '#555555'), LOSS=('#ffe599', '#c9a227'), GRAD='#cc7a00', CLIP_GROUP='#e09a3e',
                SNOW='#1565c0'),                                       # 역할별 5색 (11-25 두 번째 판)
    # 기존 모듈은 파랑 계열 하나(고정 = 옅은 파랑, 학습 = 진한 파랑), 새로 추가한 부분만 코랄, 손실·그 밖은 회색
    'blue_coral': dict(TRAIN=('#bcd4f0', '#2f5f9e'), FROZEN=('#e6eef8', '#7f9cc4'), OURS=('#f9cfc4', '#d0573c'),
                       PLAIN=('#f0f0f0', '#7a7a7a'), LOSS=('#f0f0f0', '#7a7a7a'), GRAD='#2f5f9e',
                       CLIP_GROUP='#7da2d6', SNOW='#4a7fc1'),
    # 고정 = 회색, 학습 = 청록, 새로 추가 = 보라, 손실·그 밖은 회색
    'teal_purple': dict(TRAIN=('#c8e6e3', '#2a8c84'), FROZEN=('#ececec', '#8c8c8c'), OURS=('#e0d1f0', '#7b4fa8'),
                        PLAIN=('#f5f5f5', '#8c8c8c'), LOSS=('#f5f5f5', '#8c8c8c'), GRAD='#2a8c84',
                        CLIP_GROUP='#6fb5ae', SNOW='#4a90d9'),
    # 고정 = 회색, 학습 = 주황, 새로 추가 = 파랑, 손실·그 밖은 회색
    'gray_orange_blue': dict(TRAIN=('#fde0c2', '#d9822b'), FROZEN=('#eceff3', '#8492a6'), OURS=('#cde0f7', '#2c6bb3'),
                             PLAIN=('#f4f4f4', '#888888'), LOSS=('#f4f4f4', '#888888'), GRAD='#d9822b',
                             CLIP_GROUP='#e8a868', SNOW='#5b8fd0'),
}
PALETTE = 'pink_mid'                # 11-25 사용자 선택
TRAIN, FROZEN, OURS, PLAIN, LOSS, GRAD, CLIP_GROUP, SNOW = (PALETTES[PALETTE][k] for k in
                                                           ('TRAIN', 'FROZEN', 'OURS', 'PLAIN', 'LOSS', 'GRAD',
                                                            'CLIP_GROUP', 'SNOW'))
NOTE = '#4a4a46'


def use_palette(name):
    """그리기 전에 전역 색을 바꾼다."""
    global TRAIN, FROZEN, OURS, PLAIN, LOSS, GRAD, CLIP_GROUP, SNOW
    p = PALETTES[name]
    TRAIN, FROZEN, OURS, PLAIN, LOSS = p['TRAIN'], p['FROZEN'], p['OURS'], p['PLAIN'], p['LOSS']
    GRAD, CLIP_GROUP, SNOW = p['GRAD'], p['CLIP_GROUP'], p['SNOW']


def trap(ax, x0, x1, yc, h0, h1, style, text, fs=9.5):
    fc, ec = style
    ax.add_patch(Polygon([(x0, yc - h0 / 2), (x1, yc - h1 / 2), (x1, yc + h1 / 2), (x0, yc + h0 / 2)], closed=True,
                         fc=fc, ec=ec, lw=1.6, zorder=2))
    ax.text((x0 + x1) / 2, yc, text, ha='center', va='center', fontsize=fs, linespacing=1.3, zorder=3)


def rbox(ax, x, y, w, h, style, text, fs=9.5, weight='normal', lw=1.6):
    fc, ec = style
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.0', fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, weight=weight, linespacing=1.3, zorder=3)


def group(ax, x, y, w, h, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.2', fc='none', ec=color,
                                lw=1.2, ls=(0, (4, 2)), zorder=1))


def flame(ax, cx, by, h):
    """불꽃 아이콘(학습). 바깥 주황 + 안쪽 노랑 물방울."""
    def drop(s, col, dy=0.0):
        w = 0.42 * h * s
        v = [(cx, by + dy), (cx + w, by + dy), (cx + w * 1.1, by + dy + 0.55 * h * s), (cx, by + dy + h * s),
             (cx - w * 1.1, by + dy + 0.55 * h * s), (cx - w, by + dy), (cx, by + dy)]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        ax.add_patch(PathPatch(Path(v, codes), fc=col, ec='none', zorder=5))
    drop(1.0, '#f4511e'); drop(0.55, '#ffca28', dy=0.05 * h)


def snow(ax, cx, cy, r, col=None):
    col = col or SNOW
    """눈송이 아이콘(동결)."""
    for a in (0, 60, 120):
        t = np.deg2rad(a)
        dx, dy = r * np.cos(t), r * np.sin(t)
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color=col, lw=1.6, zorder=5, solid_capstyle='round')
        for sgn in (1, -1):                                   # 끝의 작은 가지
            ex, ey = cx + sgn * dx * 0.62, cy + sgn * dy * 0.62
            for b in (35, -35):
                tb = t + np.deg2rad(b) + (np.pi if sgn < 0 else 0)
                ax.plot([ex, ex + 0.38 * r * np.cos(tb)], [ey, ey + 0.38 * r * np.sin(tb)], color=col, lw=1.2, zorder=5)


def circ(ax, cx, cy, r, sym):
    ax.add_patch(Circle((cx, cy), r, fc='white', ec='0.25', lw=1.4, zorder=4))
    ax.text(cx, cy - 0.1, sym, ha='center', va='center', fontsize=12, zorder=5)


def arrow(ax, p0, p1, color='0.25', ls='-', lw=1.3, rad=0.0, ms=11):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=ms, color=color, lw=lw, ls=ls,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=0.5, shrinkB=0.5, zorder=4))


def line(ax, pts, color='0.25', ls='-', lw=1.3, z=3):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, ls=ls, lw=lw, zorder=z, solid_joinstyle='miter', solid_capstyle='butt')


def link(ax, pts, color='0.25', ls='-', lw=1.3, ms=11):
    """수평·수직 선분을 꺾어 이은 화살표 (마지막 선분에 화살촉)."""
    if len(pts) > 2:
        line(ax, pts[:-1], color, ls, lw)
    arrow(ax, pts[-2], pts[-1], color, ls, lw, ms=ms)


def img(fig, ax, im, x, y, s, edge='0.45'):
    tf = fig.transFigure.inverted()
    (x0, y0), (x1, y1) = tf.transform(ax.transData.transform((x, y))), tf.transform(ax.transData.transform((x + s, y + s)))
    a = fig.add_axes([x0, y0, x1 - x0, y1 - y0])
    a.imshow(im, interpolation='nearest')
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_color(edge); sp.set_linewidth(0.8)


def head_note(src):
    """head 구성 요소를 그림 안 작은 글자로 (기존 head면 빈 문자열)."""
    fl = set(src.split('+')[1:])
    parts = (['CLS ⊕ patch mean'] if 'cp' in fl else []) + (['centered'] if 'ctr' in fl else []) \
        + (['flip-averaged'] if 'qf' in fl else [])
    return ' · '.join(parts)


def draw(preds, path):
    head = preds.get('head', CHOSEN_HEAD)
    note = head_note(head)
    # 11-25 세로 압축 (90 → 76): 행 간격·빈 공간을 줄이고, 오프라인 칸의 "Labeled Images" 글자는 제목과 겹쳐 뺐다.
    # 행 높이: 적응 DINO A, 원래 DINO S, CLIP C (TTA 칸), 오프라인 행 Q
    Wd, Hd = 119, 76
    A, S, C, Q = 29.6, 41.0, 53.0, 7.8
    fig = plt.figure(figsize=(Wd / 10, Hd / 10))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, Wd); ax.set_ylim(0, Hd); ax.axis('off')

    ax.add_patch(FancyBboxPatch((1, 20), Wd - 2, 54.5, boxstyle='round,pad=0,rounding_size=1.5', fc='#fcfcfc', ec='0.8', lw=1))
    ax.add_patch(FancyBboxPatch((1, 1.5), Wd - 2, 17, boxstyle='round,pad=0,rounding_size=1.5', fc='#f5f5f5', ec='0.8', lw=1))
    ax.text(3, 72.1, 'Test-Time Adaptation', fontsize=11.5, weight='bold', va='center')
    ax.text(3, 16.4, 'Offline: 16 Labeled Images per Class', fontsize=11.5, weight='bold', va='center')

    # 입력 (실제 CIFAR-10-C, severity 5) 2×2, 가운데를 원래 DINO 행에 맞춤
    ims = {c: load_corruption(c, CFG['SEVERITY'])[0][preds['corrs'][c]['index']] for c, _ in CORRS}
    pos = [(3.0, S + 0.4), (11.2, S + 0.4), (3.0, S - 7.8), (11.2, S - 7.8)]
    for (c, _), (x, y) in zip(CORRS, pos):
        img(fig, ax, ims[c], x, y, 7.6)
    ax.text(10.9, S + 9.5, 'Input', ha='center', fontsize=9.5, weight='bold')

    # 텍스트 프롬프트 → (초기화) 분류기. 분류기는 오프라인에서 미세조정, TTA 중에는 고정
    T = 66.2                                   # 분류기 행
    rbox(ax, 3, T - 3.5, 15.8, 7, PLAIN, 'a photo of a\n[CLASS].', fs=8.5)
    ax.texts[-1].set_style('italic')
    rbox(ax, 23, T - 3.5, 16, 7, FROZEN, 'Text-initialized\nClassifier', fs=8.6)
    snow(ax, 38.3, T + 3.3, 1.05)
    link(ax, [(18.8, T), (23, T)], color='0.45', ls=(0, (3, 2)))
    ax.text(20.9, T + 1.0, 'init', ha='center', va='bottom', fontsize=7, color=NOTE, style='italic')

    # CLIP 이미지 인코더 (TTA 중 LayerNorm 업데이트)
    group(ax, 21.8, C - 5.9, 19, 11.8, CLIP_GROUP)
    trap(ax, 23, 39, C, 9, 6.4, TRAIN, 'CLIP\nImage Encoder', fs=9)
    flame(ax, 38.3, C + 3.2, 2.0)

    # DINOv2 두 갈래: source(고정, CDIM 분기) / adapted(LayerNorm 적응, head)
    trap(ax, 23, 39, S, 8.6, 6, FROZEN, 'DINOv2\n(source)', fs=8.6)
    snow(ax, 38.3, S + 3.9, 1.05)
    group(ax, 21.8, A - 5.4, 19, 10.8, OURS[1])
    trap(ax, 23, 39, A, 8.6, 6, TRAIN, 'DINOv2\n(adapted)', fs=8.6)
    flame(ax, 38.3, A + 2.4, 2.0)

    # 입력 → 세 인코더
    line(ax, [(18.8, S), (20.3, S)]); line(ax, [(20.3, A), (20.3, C)])
    link(ax, [(20.3, C), (23, C)]); link(ax, [(20.3, S), (23, S)]); link(ax, [(20.3, A), (23, A)])

    # CLIP 분기: 이미지 × 분류기
    circ(ax, 48, C, 1.7, '×')
    link(ax, [(39, T), (48, T), (48, C + 1.8)])
    link(ax, [(39, C), (46.2, C)])

    # CDIM의 라벨 없는 DINO 분기 / ours head
    rbox(ax, 45, S - 3.5, 18, 7, PLAIN, 'Cross-Covariance\nProjection', fs=8.6)
    link(ax, [(48, C - 1.8), (48, S + 3.6)], color='0.45', ls=(0, (3, 2)))
    link(ax, [(39, S), (45, S)])
    rbox(ax, 45, A - 3.5, 18, 7, OURS, 'Few-shot DINO Head', fs=8.8, weight='bold', lw=2.2)
    link(ax, [(39, A), (45, A)])
    if note:                                   # head 위 빈 곳 (아래는 프로토타입 화살표가 들어옴)
        ax.text(54, A + 4.1, note, ha='center', va='bottom', fontsize=7, color=NOTE, style='italic')

    # 앙상블 ⊕ → loss / OLA
    L, O = S + 8.5, S - 7.5                    # loss / OLA 행
    circ(ax, 70, S, 2.0, '+')
    link(ax, [(49.7, C), (70, C), (70, S + 2.0)])
    link(ax, [(63, S), (68.0, S)])
    link(ax, [(63, A), (70, A), (70, S - 2.0)])
    rbox(ax, 78, L - 4.5, 18, 9, LOSS, 'Gated Info-Max\nLoss', fs=8.8)
    rbox(ax, 78, O - 4.5, 18, 9, PLAIN, 'OLA\nPrediction', fs=8.8)
    line(ax, [(72, S), (75, S)]); line(ax, [(75, O), (75, L)])
    link(ax, [(75, L), (78, L)]); link(ax, [(75, O), (78, O)])

    # gradient: loss → CLIP 이미지 인코더(LayerNorm), loss → (head를 거쳐) 적응하는 DINOv2(LayerNorm)
    g_top, g_bot = C + 7.8, A - 7.6
    link(ax, [(87, L + 4.5), (87, g_top), (31, g_top), (31, C + 6.0)], color=GRAD, ls=(0, (5, 3)), lw=1.5, ms=12)
    link(ax, [(96, L), (98.6, L), (98.6, g_bot), (31, g_bot), (31, A - 5.5)], color=GRAD, ls=(0, (5, 3)), lw=1.5, ms=12)

    # 출력 (같은 이미지 + 실제 예측) 2×2, OLA 화살표가 두 줄 사이를 가리키게
    yt = O + 3.0
    ax.text(108.4, yt + 9.8, 'Prediction', ha='center', fontsize=9.5, weight='bold')
    opos = [(101, yt), (108.6, yt), (101, yt - 11.5), (108.6, yt - 11.5)]
    for (c, _), (x, y) in zip(CORRS, opos):
        p = preds['corrs'][c]
        img(fig, ax, ims[c], x, y, 7.2)
        ok = p['ours'] == p['label']
        ax.text(x + 3.6, y - 0.4, f"{CLASSES[p['ours']]} {'✓' if ok else '✗'}", ha='center', va='top', fontsize=8.2,
                weight='bold', color='#2e7d32' if ok else '#c62828')
    link(ax, [(96, O), (100.6, O)])

    # 오프라인: 라벨 이미지 → DINOv2 → 클래스 평균 → 프로토타입, 그리고 같은 이미지로 CLIP 미세조정
    from core.data import load_clean_data
    Xtr, Ytr, _, _ = load_clean_data()
    idx = kshot_subset(Ytr, 16, SEED)
    for k in range(10):
        i = idx[Ytr[idx] == k][0]
        img(fig, ax, Xtr[i], 3 + (k % 5) * 4.4, (Q + 0.2 if k < 5 else Q - 4.2), 4.0, edge='0.6')
    trap(ax, 27, 39, Q, 8, 5.6, FROZEN, 'DINOv2\nImage Encoder', fs=8)
    snow(ax, 38.2, Q + 4.1, 1.0)
    link(ax, [(24.8, Q), (27, Q)])
    rbox(ax, 42, Q - 3, 8, 6, OURS, 'Class\nMean', fs=8.2)
    link(ax, [(39, Q), (42, Q)])
    rbox(ax, 53, Q - 3.5, 11, 7, OURS, 'Prototypes', fs=8.8, weight='bold', lw=2.2)
    link(ax, [(50, Q), (53, Q)])
    link(ax, [(58, Q + 3.5), (58, A - 3.7)], color=OURS[1], lw=1.8, ms=12)
    F = Q + 6.4                                # CLIP 미세조정 갈래 (DINO 눈송이와 제목 사이)
    line(ax, [(25.9, Q), (25.9, F)]); link(ax, [(25.9, F), (68, F)])
    rbox(ax, 68, F - 3.2, 19, 6.4, TRAIN, 'CLIP Fine-tuning\n(LayerNorm + classifier)', fs=8)
    ax.text(77.5, F - 4.0, '→ initial CLIP for TTA', ha='center', va='top', fontsize=7, color=NOTE, style='italic')

    # 범례
    lx, ly = 68, 4.6
    flame(ax, lx, ly - 1.1, 2.2); ax.text(lx + 2.0, ly, 'Trainable (LayerNorm)', va='center', fontsize=8.5)
    snow(ax, lx + 19.5, ly, 1.1); ax.text(lx + 21.3, ly, 'Frozen', va='center', fontsize=8.5)
    ax.add_patch(Rectangle((lx + 28.5, ly - 1.0), 2.8, 2.0, fc=OURS[0], ec=OURS[1], lw=1.6))
    ax.text(lx + 32.0, ly, 'Ours', va='center', fontsize=8.5)
    ax.plot([lx + 37.5, lx + 41.5], [ly, ly], color=GRAD, lw=1.5, ls=(0, (5, 3)))
    ax.text(lx + 42.3, ly, 'Gradient', va='center', fontsize=8.5)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=220, facecolor='white')
    plt.close(fig)
    print('저장:', path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recompute', action='store_true')
    ap.add_argument('--out', default=f'{OUT}/fig6_model_ours.png', help='그림 경로 (배치 점검용으로 다른 곳에 그릴 때)')
    ap.add_argument('--palette', default=PALETTE, choices=list(PALETTES))
    a = ap.parse_args()
    use_palette(a.palette)
    if a.recompute or not os.path.exists(PRED_JSON):
        preds = compute_preds()
    else:
        with open(PRED_JSON) as f:
            preds = json.load(f)
    draw(preds, a.out)


if __name__ == '__main__':
    main()
