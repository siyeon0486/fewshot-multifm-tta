"""㉚ 결과로 ours의 head를 고른다 (2026-11-24 사용자 "결과를 바탕으로 더 높은 쪽을 채택"). GPU 불필요.
비교: 지금 ours(OURS_HEAD = 새 head + dlnh) vs 단순한 head(SIMPLE_HEAD = 기존 head + dlnh), 둘 다 CDIM.
조건 10개(CIFAR-10-C 15종 평균): 16장 미세조정 CLIP, zero-shot CLIP, 5만 장 상한, 라벨 수 곡선 1·2·4·8·79·157·625장.
조건마다 같은 subset(5만 장은 학습 seed)끼리 짝지은 차이(단순 − ours)의 평균을 내고, 조건 평균이 0보다 크면 단순한 head,
아니면(같으면 포함) 지금 ours. 결과는 ours_choice.txt(첫 줄 = 고른 head)와 analysis/ours_choice.md에 쓴다.
확장 실험 stage(config X_*)는 이 파일을 읽는다.  실행: python pick_ours.py   (점검만: python pick_ours.py --dry)"""
import os, sys, time
os.environ['DATASET'] = 'cifar10'            # 대기열이 DATASET=cifar100으로 돌아도 비교는 CIFAR-10 결과로
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A
from config import CFG, OURS_HEAD, SIMPLE_HEAD, PROJECT_DIR

C = CFG['CORR_ALL']
dk = A._kfill(A.load('results_kshot.csv'))
s = A._specs()


def per_seed(filt):
    x = dk['method'] == 'cdim'
    for c, v in filt.items():
        x &= dk[c] == v
    p = dk[x & dk.corruption.isin(C)].groupby(['seed', 'corruption']).acc.mean().unstack('corruption').reindex(columns=C)
    return p.dropna().mean(1)          # 15종이 다 끝난 seed만


conds = [('16장 미세조정 CLIP (Main)', s['ft']), ('zero-shot CLIP', s['zs']), ('5만 장 (상한)', s['full'])] \
    + [(f'곡선 {k}장', dict(s['ft'], shots=k)) for k in (1, 2, 4, 8, 79, 157, 625)]
lines = ['| 조건 | ours | 단순 | 단순 − ours | subset별 |', '|---|---|---|---|---|']
diffs = []
for name, base in conds:
    a = per_seed(dict(base, proto=1.0, proto_src=OURS_HEAD))
    b = per_seed(dict(base, proto=1.0, proto_src=SIMPLE_HEAD))
    d = (b - a).dropna()
    if d.empty:
        lines.append(f'| {name} | {a.mean():.2f} | ⏳ | 빠짐 | |'); continue
    diffs.append(d.mean())
    lines.append(f'| {name} | {a[d.index].mean():.2f} | {b[d.index].mean():.2f} | {d.mean():+.2f} | '
                 + ' / '.join(f'{v:+.2f}' for v in d.values) + ' |')
avg = sum(diffs) / len(diffs) if diffs else 0.0
choice = SIMPLE_HEAD if (diffs and avg > 0) else OURS_HEAD
name = '단순한 head (기존 head + DINO 적응)' if choice == SIMPLE_HEAD else '지금 ours (새 head + DINO 적응)'
summary = (f'조건 {len(diffs)}/{len(conds)}개, 단순 − ours 평균 {avg:+.3f} → **{name}** (`{choice}`) 채택 '
           f'({time.strftime("%Y-%m-%d %H:%M")})')
md = '\n'.join(lines + ['', summary])
print(md)
if '--dry' in sys.argv:                     # 점검용: 파일을 쓰지 않음
    sys.exit(0)
with open(f'{PROJECT_DIR}/ours_choice.txt', 'w') as f:
    f.write(choice + '\n' + summary + '\n')
os.makedirs(A.OUT, exist_ok=True)
with open(f'{A.OUT}/ours_choice.md', 'w') as f:
    f.write(md + '\n')
print(f'\n→ {PROJECT_DIR}/ours_choice.txt, {A.OUT}/ours_choice.md')
