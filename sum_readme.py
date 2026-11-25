"""README 결과 표 (2026-11-25 사용자 요청: 예시 논문처럼 Table 1 = 방법 × 손상 15종 + 평균, Table 2 = 구성 요소 ablation). GPU 불필요.
  python sum_readme.py                     # CIFAR-10-C
  DATASET=cifar100 python sum_readme.py    # CIFAR-100-C
출력(markdown, experiments/<EXP>/analysis/readme_tables.md):
  Main      : 라벨 없음(zero-shot CLIP, 논문 설정) 7개 방법 / 5만 장 학습 CLIP(①) 7개 방법 / ours(라벨 160장)
  같은 라벨 : 같은 라벨(클래스당 16장)을 쓰는 baseline과 채택하지 않은 변형 (Main에서 분리)
  Ablation  : 구성 요소 조합별 평균 (CLIP 미세조정 / Few-shot DINO head / DINO 적응)
값은 analyze._collect(중복 run은 먼저 기록된 값, 15종이 다 있는 seed만)의 반올림 전 평균을 2자리로."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import analyze as A
from config import CFG

C = CFG['CORR_ALL']
SHORT = ['Gauss.', 'Shot', 'Impul.', 'Defoc.', 'Glass', 'Motion', 'Zoom', 'Snow', 'Frost', 'Fog', 'Brit.', 'Contr.',
         'Elastic', 'Pixel.', 'JPEG']
VENUE = {'source': "ICML'21", 'tent': "ICLR'21", 'eta': "ICML'22", 'sar': "ICLR'23", 'tda': "CVPR'24",
         'batclip': "ICCV'25", 'cdim': "arXiv'26"}
NAME = {'source': 'Source', 'tent': 'TENT', 'eta': 'ETA', 'sar': 'SAR', 'tda': 'TDA', 'batclip': 'BATCLIP', 'cdim': 'CDIM'}
M7 = ['source', 'tent', 'eta', 'sar', 'tda', 'batclip', 'cdim']

dk = A._kfill(A.load('results_kshot.csv'))
s = A._specs()
n16 = 3


def row_cells(r, bold=False):
    if not r['ok']:
        return ['⏳'] * (len(C) + 1)
    b = (lambda v: f'**{v}**') if bold else (lambda v: v)
    return [b(f'{r[c]:.2f}') for c in C] + [b(f"{r['Mean']:.2f}")]


lines = [f"## {CFG['DATASET']} — Main", '',
         '| 라벨 | Method | Venue | ' + ' | '.join(SHORT) + ' | Mean |', '|' + '---|' * (len(C) + 4)]
zs_name = {'source': 'CLIP (zero-shot)'}
groups = [('없음 (zero-shot CLIP)', s['zs'], 1, zs_name), ('5만 장 학습 CLIP', s['full'], 3, {})]
for gname, spec, n, rename in groups:
    t = A._collect(dk, [(gname, rename.get(m, NAME[m]), spec, m, n) for m in M7])
    for (_, r), m in zip(t.iterrows(), M7):
        lines.append(f"| {gname} | {r['Method']} | {VENUE[m]} | " + ' | '.join(row_cells(r)) + ' |')
t = A._collect(dk, [('ours', 'Ours', s['ours'], 'cdim', n16)])
r = t.iloc[0]
lines.append('| **클래스당 16장** | **Ours** (CDIM + Few-shot DINO head + DINO 적응) | | ' + ' | '.join(row_cells(r, True)) + ' |')
ours_mean = r['Mean']

lines += ['', f"## {CFG['DATASET']} — 같은 라벨(클래스당 16장) baseline과 채택하지 않은 변형", '',
          '| 분류 | 방법 | 평균 | ours − 이 방법 |', '|---|---|---|---|']
extra = [('TTA 없음', 'CLIP 미세조정', s['ft'], 'source'),
         ('TTA 없음', 'Tip-Adapter (CLIP 캐시)', s['tc'], 'source'),
         ('TTA 없음', 'DINO 캐시 (Tip 방식)', s['td'], 'source'),
         ('CDIM', 'CDIM + CLIP 미세조정', s['ft'], 'cdim'),
         ('CDIM', 'CDIM + Tip-Adapter', s['tc'], 'cdim'),
         ('CDIM', 'CDIM + DINO 캐시 (Tip 방식)', s['td'], 'cdim'),
         ('채택 안 한 변형', 'CDIM + DINO 적응 (head 없음, 라벨 없음)', s['dln_nohead_zs'], 'cdim'),
         ('채택 안 한 변형', 'CDIM + CLIP 미세조정 + DINO 적응 (head 없음)', s['dln_nohead'], 'cdim'),
         ('채택 안 한 변형', '적응한 DINO를 CDIM 분기에도 (dln)', s['ours_dln'], 'cdim')]
t = A._collect(dk, [(g, n, sp, m, 1 if sp is s['dln_nohead_zs'] else n16) for g, n, sp, m in extra])
for _, r in t.iterrows():
    if not r['ok']:
        lines.append(f"| {r['그룹']} | {r['Method']} | ⏳ | |"); continue
    sd = '' if np.isnan(r['std']) else f" ±{r['std']:.2f}"
    lines.append(f"| {r['그룹']} | {r['Method']} | {r['Mean']:.2f}{sd} | {ours_mean - r['Mean']:+.2f} |")

abl = [('', '', '', s['zs'], 1), ('✓', '', '', s['ft'], n16), ('', '', '✓', s['dln_nohead_zs'], 1),
       ('✓', '', '✓', s['dln_nohead'], n16), ('✓', '✓', '', s['ours_nodino'], n16),
       ('', '✓', '✓', s['ours_zs'], n16), ('✓', '✓', '✓', s['ours'], n16)]
t = A._collect(dk, [('abl', f'{a}|{b}|{c}', sp, 'cdim', n) for a, b, c, sp, n in abl])
lines += ['', f"## {CFG['DATASET']} — Ablation (CDIM 위, 평균)", '',
          '| CLIP 미세조정 (16장) | Few-shot DINO head | DINO 적응 | 평균 |', '|---|---|---|---|']
for _, r in t.iterrows():
    a, b, c = r['Method'].split('|')
    v = '⏳' if not r['ok'] else f"{r['Mean']:.2f}"
    lines.append(f'| {a} | {b} | {c} | {v} |')

md = '\n'.join(lines)
print(md)
os.makedirs(A.OUT, exist_ok=True)
with open(f'{A.OUT}/readme_tables.md', 'w') as f:
    f.write(md + '\n')
print(f'\n저장: {A.OUT}/readme_tables.md')
