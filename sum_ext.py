"""추가 실험 2·3 정리 (2026-11-25 사용자 "1,3,2 순서로 하자"). GPU 불필요.
  python sum_ext.py --exp s3     # 3. severity 3      (experiments/server_v100_pil_s3,   config Z_base·Z_ours)
  python sum_ext.py --exp cont   # 2. continual TTA   (experiments/server_v100_pil_cont, config Z_base·Z_ours·Z_head_a/b)
Table A 구성(① 5만 장 / ② 16장 / ③ ours / 라벨 없음, continual은 + head만)을 15종 평균·손상 유형별로 만들고,
같은 구성의 기본 실험(severity 5, episodic) 값과 나란히 놓는다. continual은 손상별 (continual − episodic)도 보여 준다.
중복 run은 analyze.py처럼 가장 먼저 기록된 값만 쓴다. 표준편차는 subset(또는 학습 seed) 3개의 표본 표준편차."""
import os, sys, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from config import CFG, FT_ITERS, FULL_ITERS, CHOSEN_HEAD

ROOT, BASE = CFG['EXP_ROOT'], 'server_v100_pil'
CORRS = CFG['CORR_ALL']
GROUPS = {'Noise': CORRS[:3], 'Blur': CORRS[3:7], 'Weather': CORRS[7:11], 'Digital': CORRS[11:]}
KEY = ['head_mode', 'shots', 'iters', 'aug', 'mixup', 'cutmix', 'jsd', 'src_loss', 'align', 'wise', 'cdim_m',
       'proto', 'proto_src', 'seed', 'method', 'corruption', 'cdim_clip_tau_inv']
DEF = dict(aug='none', mixup=0.0, cutmix=0.0, jsd=0.0, src_loss='ce', align=0.0, wise=1.0, cdim_m='batch',
           proto=0.0, proto_src='dino', method='cdim', cdim_clip_tau_inv=CFG['CDIM_CLIP_TAU_INV'])
FT = dict(head_mode='text_init', shots=16, iters=FT_ITERS)
ROWS = [('① 5만 장', dict(head_mode='text_init', shots=5000, iters=FULL_ITERS), (0, 1, 2)),
        ('② 16장', FT, (0, 1, 2)),
        ('③ ② + ours', dict(FT, proto=1.0, proto_src=CHOSEN_HEAD), (0, 1, 2)),
        ('(참고) ② + head만 (DINO 고정)', dict(FT, proto=1.0, proto_src=CHOSEN_HEAD.replace('+dlnh', '')), (0, 1, 2)),
        ('(참고) 라벨 없음 (zero-shot CLIP)', dict(head_mode='zeroshot', shots=16, iters=0), (0,))]


def load(exp):
    fs = sorted(glob.glob(f'{ROOT}/{exp}/results_kshot*.csv'))
    if not fs:
        return None
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    for c in ('proto', 'proto_src'):
        if c not in df.columns:
            df[c] = DEF[c]
    df[['proto', 'proto_src']] = df[['proto', 'proto_src']].fillna({'proto': 0.0, 'proto_src': 'dino'})
    return df.sort_values('time', kind='stable').drop_duplicates(KEY, keep='first')


def pick(df, spec, seeds):
    """seed × corruption 정확도 표 (15종을 다 가진 seed만)."""
    if df is None:
        return pd.DataFrame()
    q = df
    for k, v in {**DEF, **spec}.items():
        q = q[q[k].astype(str) == v] if isinstance(v, str) else q[np.isclose(q[k].astype(float), float(v))]
    piv = q[q.seed.isin(seeds)].pivot_table(index='seed', columns='corruption', values='acc')
    return piv.reindex(columns=CORRS).dropna()


def stats(piv):
    if piv.empty:
        return None
    m = piv.mean(1)
    out = {g: piv[c].mean(1).mean() for g, c in GROUPS.items()}
    out.update(Mean=m.mean(), std=m.std(ddof=1) if len(m) > 1 else np.nan, n=len(m), per=piv.mean(0))
    return out


def fmt(s, key='Mean'):
    if s is None:
        return '⏳'
    v = f'{s[key]:.2f}'
    return v + (f"±{s['std']:.2f}" if key == 'Mean' and not np.isnan(s['std']) else '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', choices=['s3', 'cont'], required=True)
    a = ap.parse_args()
    exp = f'{BASE}_s3' if a.exp == 's3' else f'{BASE}_cont'
    label = 'severity 3, episodic' if a.exp == 's3' else 'severity 5, continual (15종을 리셋 없이 이어서)'
    new, base = load(exp), load(BASE)
    rows = [r for r in ROWS if a.exp == 'cont' or 'head만' not in r[0]]
    S = {n: stats(pick(new, sp, sd)) for n, sp, sd in rows}
    B = {n: stats(pick(base, sp, sd)) for n, sp, sd in rows}
    md = [f'## 추가 실험: {label}', f'결과 `experiments/{exp}`, CDIM, 15종. 비교 열은 기본 실험(severity 5, episodic).', '',
          '| 조건 | ' + ' | '.join(GROUPS) + ' | 평균 | 회 | 기본 실험 평균 | 차이 |', '|' + '---|' * (len(GROUPS) + 5)]
    for n, _, _ in rows:
        s, b = S[n], B[n]
        cells = [fmt(s, g) for g in GROUPS] if s else ['⏳'] * len(GROUPS)
        d = f"{s['Mean'] - b['Mean']:+.2f}" if s and b else ''
        md.append(f"| {n} | " + ' | '.join(cells) + f" | **{fmt(s)}** | {s['n'] if s else 0} | {fmt(b)} | {d} |")
    g = lambda D, n: D[n]['Mean'] if D.get(n) else np.nan
    f1, f2, f3 = (g(S, n) for n in ('① 5만 장', '② 16장', '③ ② + ours'))
    b1, b2, b3 = (g(B, n) for n in ('① 5만 장', '② 16장', '③ ② + ours'))
    md += ['', '| | 이 실험 | 기본 실험 |', '|---|---|---|',
           f'| 감소 ② − ① | {f2 - f1:+.2f} | {b2 - b1:+.2f} |',
           f'| 회복 ③ − ② | {f3 - f2:+.2f} | {b3 - b2:+.2f} |',
           f'| **③ − ①** | **{f3 - f1:+.2f}** | {b3 - b1:+.2f} |']
    if a.exp == 'cont':
        fh = g(S, '(참고) ② + head만 (DINO 고정)')
        bh = g(B, '(참고) ② + head만 (DINO 고정)')
        md += [f'| DINO 적응의 효과 ③ − (head만) | {f3 - fh:+.2f} | {b3 - bh:+.2f} |', '',
               '손상별 정확도 (스트림 순서, subset 평균). 차이 = continual − episodic. 뒤로 갈수록 차이가 작아지면(음수로 가면) 긴 스트림에서 무너지는 것',
               '', '| # | 손상 | ② cont | ② epi | ② 차이 | head만 cont | head만 차이 | ③ cont | ③ epi | ③ 차이 |',
               '|' + '---|' * 10]
        keys = ('② 16장', '(참고) ② + head만 (DINO 고정)', '③ ② + ours')
        for i, c in enumerate(CORRS):
            cells = []
            for k in keys:
                s, b = S.get(k), B.get(k)
                cv = s['per'][c] if s else np.nan
                bv = b['per'][c] if b else np.nan
                cells += ([f'{cv:.2f}'] if k == keys[1] else [f'{cv:.2f}', f'{bv:.2f}']) + [f'{cv - bv:+.2f}']
            md.append(f'| {i + 1} | {c} | ' + ' | '.join(cells) + ' |')
    text = '\n'.join(md)
    print(text)
    out = f'{ROOT}/{exp}/analysis'
    os.makedirs(out, exist_ok=True)
    with open(f'{out}/ext_summary.md', 'w') as f:
        f.write(text + '\n')
    print(f'\n저장: {out}/ext_summary.md')


if __name__ == '__main__':
    main()
