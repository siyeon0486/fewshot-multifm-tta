"""Tip 계열 baseline의 α·β를 D4의 λ와 같은 규칙으로 고른다 (TTA 없음, 결과 CSV에는 기록하지 않음).

배경 (RESULTS.md ⑫): DINO 캐시(Tip 방식)는 D4와 같은 DINO 특징·같은 160장을 쓰는데 clean 91.1% (D4 97.4%).
논문 기본값 α=1에서는 캐시 logit이 CLIP logit(100·cos)에 묻힌다. Tip-Adapter 논문은 α·β를 검증 데이터로 탐색하는데
⑭⑮의 baseline은 기본값을 그대로 썼고, D4의 λ는 탐색용 4종에서 골랐다 → 조정 기회를 같게 맞춘다.

규칙: zero-shot CLIP + head, TTA 없이 탐색용 corruption 4종 평균(mean4)이 가장 높은 값 (subset 3개 평균).
      D4도 같은 표에 넣어, 이 규칙이 실제로 쓴 λ=1을 고르는지 확인한다.
Tip-Adapter (Zhang et al., ECCV 2022): logit = 100·C Wᵀ + α·exp(−β(1 − F Kᵀ))·V. grid는 D4보다 넓게 잡는다(baseline에 유리한 쪽).
검증: α=1, β=5.5 행이 diag_dino.py와 같아야 한다 (mean4 CLIP 캐시 63.44, DINO 캐시 63.41, D4 λ=1 81.55).

GPU 노드에서 (몇 분):  CUDA_VISIBLE_DEVICES=1 python diag_tip.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config import CFG, CLASSES
from core.runtime import init_runtime, R
from core.data import kshot_subset
from core.models import load_models, new_classifier
from core.methods import dino_features, get_corruption, clip_image_feats
from core import source as S
from diag_dino import clip_feats, acc, dino_protos, SEEDS, TAU_D

ALPHAS = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
BETAS = [0.1, 0.25, 0.5, 1, 3, 5.5, 8, 12]   # 첫 실행에서 DINO 캐시가 β=1(끝)에서 최고 → 작은 β 추가
D4_LAMBDAS = [0.25, 0.5, 1.0, 2.0, 4.0]


def main():
    init_runtime(require_gpu=True)
    S.load_clean()
    load_models()
    corrs = CFG['CORR_SWEEP']
    t0 = time.time()
    sets = {'clean': (S.D.Xte, S.D.Yte, dino_features(S.D.Xte).float())}
    for c in corrs:
        x, y, d = get_corruption(c)
        sets[c] = (x, y, d.float())
    clf = new_classifier('zeroshot')[0]
    with torch.no_grad():
        W = clf.class_weights()
        cf = {k: clip_feats(clf, v[0]) for k, v in sets.items()}
        lc = {k: 100.0 * cf[k] @ W.t() for k in sets}
    print(f'특징 준비 {time.time() - t0:.0f}s', flush=True)

    rows = []

    def add(head, a, b, seed, logits):
        accs = {k: acc(logits[k], sets[k][1]) for k in sets}
        rows.append(dict(head=head, alpha=a, beta=b, seed=seed, **{k: round(v, 2) for k, v in accs.items()},
                         mean4=round(float(np.mean([accs[c] for c in corrs])), 2)))

    for s in SEEDS:
        with torch.no_grad():
            idx = kshot_subset(S.D.Ytr, 16, s)
            xs, ys = S.D.Xtr[idx], torch.from_numpy(S.D.Ytr[idx]).to(R.device)
            V = F.one_hot(ys, len(CLASSES)).float()     # CIFAR-100 (2026-11-25 추가 실험 1)에서도 쓰게 클래스 수를 config에서
            keys = {'tip_clip': clip_image_feats(clf, xs), 'tip_dino': dino_features(xs).float().to(R.device)}
            query = {'tip_clip': cf, 'tip_dino': {k: sets[k][2].to(R.device) for k in sets}}
            P = dino_protos(s)
            for lam in D4_LAMBDAS:                     # 같은 규칙의 확인용 (D4 = λ·50·D Pᵀ)
                add('d4', lam, None, s, {k: lc[k] + lam * TAU_D * query['tip_dino'][k] @ P.t() for k in sets})
            for h in ('tip_clip', 'tip_dino'):
                aff = {k: (query[h][k] @ keys[h].t()).float() for k in sets}
                for b in BETAS:
                    cache = {k: torch.exp(-b * (1 - aff[k])) @ V for k in sets}
                    for a in ALPHAS:
                        add(h, a, b, s, {k: lc[k] + a * cache[k] for k in sets})
        print(f'  seed {s} 끝', flush=True)

    df = pd.DataFrame(rows)
    out = f'{S.exp_dir()}/analysis'
    os.makedirs(out, exist_ok=True)
    df.to_csv(f'{out}/diag_tip.csv', index=False)
    cols = ['clean'] + corrs + ['mean4']
    agg = df.groupby(['head', 'alpha', 'beta'], dropna=False)[cols].mean().round(2).reset_index()
    pd.set_option('display.width', 250)

    print('\n=== 확인: 기존 diag_dino.py와 같은 설정 (seed 평균) ===')
    chk = agg[((agg['head'] != 'd4') & (agg.alpha == 1) & (agg.beta == 5.5)) | ((agg['head'] == 'd4') & (agg.alpha == 1))]
    print(chk.to_string(index=False))

    choice = {}
    for h in ('d4', 'tip_clip', 'tip_dino'):
        a = agg[agg['head'] == h].sort_values('mean4', ascending=False)
        print(f'\n=== {h}: mean4 상위 8개 (seed 평균) ===')
        print(a.head(8).to_string(index=False))
        best = a.iloc[0]
        choice[h] = dict(alpha=float(best.alpha), beta=None if pd.isna(best.beta) else float(best.beta),
                         clean=float(best.clean), mean4=float(best.mean4))
        if h != 'd4':
            edge = best.alpha in (min(ALPHAS), max(ALPHAS)) or best.beta in (min(BETAS), max(BETAS))
            if edge:
                print(f'⚠️ {h}: 고른 값이 grid 끝에 있습니다 (α={best.alpha:g}, β={best.beta:g}). grid를 넓혀야 할 수 있습니다.')
    print('\n=== 선택 (규칙: TTA 없는 탐색용 4종 평균 최대) ===')
    for h, c in choice.items():
        print(f"  {h:9s} α(λ)={c['alpha']:g}" + ('' if c['beta'] is None else f" β={c['beta']:g}")
              + f"  clean {c['clean']:.2f}  mean4 {c['mean4']:.2f}")
    with open(f'{out}/diag_tip_choice.json', 'w') as f:
        json.dump(choice, f, indent=1)
    print(f'\n저장: {out}/diag_tip.csv, diag_tip_choice.json')


if __name__ == '__main__':
    main()
