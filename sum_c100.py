"""확장 실험 CIFAR-100-C 표 (2026-11-24, config X_* stage). GPU 불필요.  실행: python sum_c100.py
CIFAR-10과 같은 구성: Main(라벨 없음 / 16장 / 16장 + TTA / 5만 장 참고), Table A(① → ② → ③), 2×2, 부록(TTA 방법 × 라벨 사용).
ours의 head는 ours_choice.txt(pick_ours.py)를 따른다(config.CHOSEN_HEAD). 결과·표는 experiments/server_v100_pil_c100."""
import os, sys
os.environ['DATASET'] = 'cifar100'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import analyze as A
from config import CFG, FT_ITERS, FULL_ITERS, NEW_HEAD, CHOSEN_HEAD, TIP_TUNED

dk = A.load('results_kshot.csv')
if dk is None or dk.empty:
    sys.exit('CIFAR-100 결과 없음')
dk = A._kfill(dk)
G = list(A.CORR_GROUPS)
base = dict(A.KDEFAULT, aug='none', shots=16)
zs = dict(base, iters=0, head_mode='zeroshot')
ft = dict(base, iters=FT_ITERS, head_mode=CFG['HEAD_MODE_MAIN'])
full = dict(base, shots=5000, iters=FULL_ITERS, head_mode=CFG['HEAD_MODE_MAIN'])
h = lambda b, src=CHOSEN_HEAD, lam=1.0: dict(b, proto=lam, proto_src=src)
nohead = lambda b: h(b, NEW_HEAD + '+dln', 1e-6)
OURS = f'**ours** (`{CHOSEN_HEAD}`)'


def table(title, specs, path):
    t = A._collect(dk, specs)
    out = ['| 그룹 | 행 | ' + ' | '.join(G) + ' | Mean | 회 |', '|' + '---|' * (len(G) + 4)]
    for _, r in t.iterrows():
        if not r['ok']:
            out.append(f"| {r['그룹']} | {r['Method']} | " + ' | '.join(['⏳'] * (len(G) + 1)) + f" | {r['seeds']} |"); continue
        m = f"{r['Mean']:.2f}" + ('' if np.isnan(r['std']) else f"±{r['std']:.2f}")
        out.append(f"| {r['그룹']} | {r['Method']} | " + ' | '.join(f'{r[g]:.2f}' for g in G) + f" | {m} | {r['seeds']} |")
    md = '\n'.join(out)
    os.makedirs(A.OUT, exist_ok=True)
    with open(f'{A.OUT}/{path}', 'w') as f:
        f.write(md + '\n')
    print(f'\n### {title}\n' + md)
    return t.set_index('Method')


L0, L16, L16T, LF = '라벨 없음 (zero-shot CLIP)', '클래스당 16장, TTA 없음', '클래스당 16장 + TTA', '참고: 5만 장 (10 epoch)'
main = [(L0, n, zs, m, 1) for n, m in (('CLIP zero-shot (TTA 없음)', 'source'), ('TENT', 'tent'), ('ETA', 'eta'),
                                        ('SAR', 'sar'), ('TDA', 'tda'), ('BATCLIP', 'batclip'), ('CDIM', 'cdim'))]



def tip(src):
    """같은 1,600장의 Tip 계열 head (2026-11-25 추가 실험 1, config Y_tip_a/b). α·β는 diag_tip.py가 CIFAR-100에서 고른 값."""
    a, b = TIP_TUNED[src]
    return dict(zs, proto=a / CFG['TIP_ALPHA'], proto_src=src if b == CFG['TIP_BETA'] else f'{src}@{b:g}')


TIPS = ([('Tip-Adapter', tip('tip_clip')), ('DINO 캐시 (Tip 방식)', tip('tip_dino'))] if TIP_TUNED else [])
main += [(L0, 'CDIM + DINO LN 적응', nohead(zs), 'cdim', 1),
         (L16, 'CLIP 미세조정', ft, 'source', 3)]
main += [(L16, n, s, 'source', 3) for n, s in TIPS]
main += [(L16T, 'CDIM + CLIP 미세조정', ft, 'cdim', 3),
         (L16T, 'CDIM + CLIP 미세조정 + DINO LN 적응', nohead(ft), 'cdim', 3)]
main += [(L16T, f'CDIM + {n}', s, 'cdim', 3) for n, s in TIPS]
main += [
         (L16T, OURS, h(ft), 'cdim', 3),
         (LF, 'CDIM + CLIP 미세조정 (5만 장)', full, 'cdim', 3)]
print(f"CIFAR-100-C severity 5, 15종 평균, 결과: {A.EXP}")
table('Main — CIFAR-100-C', main, 'c100_main.md')

ta = table('Table A — CIFAR-100-C, CDIM', [
    ('①', 'Full 5만 장', full, 'cdim', 3),
    ('②', '16장 CLIP 미세조정', ft, 'cdim', 3),
    ('③', OURS, h(ft), 'cdim', 3),
    ('참고', '② + DINO 적응만 (head 없음)', nohead(ft), 'cdim', 3),
    ('참고', '② + head만 (DINO 고정)', h(ft, CHOSEN_HEAD.replace('+dlnh', '')), 'cdim', 3),
    ('참고', 'CLIP 학습 없이 ours', h(zs), 'cdim', 3),
    ('참고', '라벨 없음 (zero-shot CLIP)', zs, 'cdim', 1),
    ('참고', '5만 장 상한: ① + ours 방식', h(full), 'cdim', 3)], 'c100_table_a.md')
g = lambda n: ta.loc[n, 'Mean'] if n in ta.index and ta.loc[n, 'ok'] else np.nan
f1, f2, f3 = g('Full 5만 장'), g('16장 CLIP 미세조정'), g(OURS)
print(f'\n감소 ②−① {f2 - f1:+.2f} | 회복 ③−② {f3 - f2:+.2f} | ③−① {f3 - f1:+.2f}')
dh, dd = g('② + head만 (DINO 고정)') - f2, g('② + DINO 적응만 (head 없음)') - f2
print(f'2×2 (② 기준): head만 {dh:+.2f}, DINO 적응만 {dd:+.2f}, 둘 다 {f3 - f2:+.2f}, 상호작용 {f3 - f2 - dh - dd:+.2f}')

app = []
for name, b, n in (('사용 안 함 (zero-shot CLIP)', zs, 1), ('CLIP 미세조정', ft, 3), ('CLIP 미세조정 + ours head', h(ft), 3)):
    app += [(name, m, b, m, n) for m in A.METHOD_ORDER]
t = A._collect(dk, app)
out = ['| 라벨 사용 | ' + ' | '.join(A.METHOD_ORDER) + ' |', '|---|' + '---|' * len(A.METHOD_ORDER)]
for name in dict.fromkeys(t['그룹']):
    r = t[t['그룹'] == name].set_index('Method')
    out.append(f'| {name} | ' + ' | '.join(f"{r.loc[m, 'Mean']:.2f}" if r.loc[m, 'ok'] else '⏳' for m in A.METHOD_ORDER) + ' |')
print('\n### 부록 — TTA 방법 × 라벨 사용 (15종 평균)\n' + '\n'.join(out))
with open(f'{A.OUT}/c100_appendix.md', 'w') as fh:
    fh.write('\n'.join(out) + '\n')


def fig8():
    """두 데이터셋의 Table A(① 5만 장 → ② 16장 → ③ ② + ours, CDIM, 15종 평균). 점 + 표준편차, 데이터셋마다 한 칸.
    CIFAR-10 값은 make_figs.py가 다시 만든 analysis/table_a_long.csv(같은 CHOSEN_HEAD). fig7과 같은 색(baseline 파랑, ours 주황)."""
    import shutil
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d10 = f"{CFG['EXP_ROOT']}/server_v100_pil/analysis"
    c10 = pd.read_csv(f'{d10}/table_a_long.csv')
    c10 = c10[(c10.method == 'cdim') & c10.ok.astype(bool)].set_index('cond')
    pts = {'CIFAR-10-C': [(c10.loc[c, 'Mean'], c10.loc[c, 'std']) if c in c10.index else (np.nan, np.nan)
                          for c in ('full', '16shot', 'ours')],
           'CIFAR-100-C': [(g(n), ta.loc[n, 'std'] if n in ta.index else np.nan)
                           for n in ('Full 5만 장', '16장 CLIP 미세조정', OURS)]}
    # 변화량은 반올림 전 값으로 (표와 같은 숫자). CIFAR-10은 table_a()가 반올림 전에 계산해 둔 drop·recover
    am = pd.read_csv(f'{d10}/table_a_all_methods.csv').set_index('method')
    deltas = {'CIFAR-10-C': [am.loc['cdim', 'drop'], am.loc['cdim', 'recover']] if 'cdim' in am.index else [np.nan] * 2,
              'CIFAR-100-C': [f2 - f1, f3 - f2]}
    BLUE, ORANGE, INK, INK2, MUTED, GRID, AXIS = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
    style = [(MUTED, 'o'), (BLUE, 'o'), (ORANGE, 's')]
    ticks = ['① All 50k labels', '② 16 per class', '③ ② + ours']
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    for ax, (name, v) in zip(axes, pts.items()):
        m = np.array([a for a, _ in v], float); sd = np.nan_to_num(np.array([b for _, b in v], float))
        ok = ~np.isnan(m)
        ax.plot(np.arange(3)[ok], m[ok], color=AXIS, lw=1.2, zorder=1)
        for i, ((c, mk), y, e) in enumerate(zip(style, m, sd)):
            if np.isnan(y): continue
            ax.errorbar([i], [y], yerr=[e], color=c, marker=mk, ms=8, capsize=2.5, elinewidth=1.2, mec='white', mew=1.2,
                        zorder=3)
            ax.text(i, y + max(e, 0.2) + 0.35, f'{y:.2f}', ha='center', va='bottom', fontsize=9, color=INK)
        for i in (0, 1):                     # 단계 사이 변화량 (− 기호는 유니코드 마이너스)
            dv = deltas[name][i]
            if ok[i] and ok[i + 1] and not np.isnan(dv):
                ax.text(i + 0.5, (m[i] + m[i + 1]) / 2, f'{dv:+.2f}'.replace('-', '−'), ha='center', va='center',
                        fontsize=8.5, color=INK2, bbox=dict(fc='white', ec='none', pad=1.0))
        if ok.any():
            lo, hi = np.nanmin(m - sd), np.nanmax(m + sd)
            ax.set_ylim(np.floor(lo - 1.0), np.ceil(hi + 1.6))
        ax.set_xticks(range(3)); ax.set_xticklabels(ticks, fontsize=8.5, color=INK2)
        ax.set_xlim(-0.45, 2.45)
        ax.set_title(name, fontsize=10.5, color=INK, loc='left')
        ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_color(AXIS)
        ax.tick_params(colors=INK2, labelsize=8.5)
    axes[0].set_ylabel('Accuracy (%)', fontsize=10, color=INK2)
    fig.suptitle('CDIM, mean of 15 corruptions (severity 5). ±: 3 label subsets (① 3 training seeds)', fontsize=8.5,
                 color=INK2, x=0.01, ha='left', y=0.02, va='bottom')
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    os.makedirs(f'{A.OUT}/figures', exist_ok=True)
    p = f'{A.OUT}/figures/fig8_table_a_datasets.png'
    fig.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    os.makedirs(f'{os.path.dirname(os.path.abspath(__file__))}/figures', exist_ok=True)
    shutil.copy2(p, f'{os.path.dirname(os.path.abspath(__file__))}/figures/fig8_table_a_datasets.png')
    print('그림:', p)


try:
    fig8()
except Exception as e:                      # 그림이 실패해도 표는 이미 저장됨
    print('fig8 실패:', repr(e))
