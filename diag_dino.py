"""16-shot 라벨을 DINOv2에 쓰면 clean 분류 성능이 오르는지 진단 (TTA 없음). 결과 CSV(results*.csv)에는 기록하지 않는다.

배경: TTA 후 정확도는 source의 clean 정확도로 거의 정해진다 (RESULTS.md 분석, 모델 22개 R² 0.88).
지금 CDIM에서 DINO는 라벨을 쓰지 않는다. 16-shot 라벨로 만든 DINO 클래스 프로토타입이
clean 정확도를 올릴 여지가 있는지 먼저 본다.

분류기 (모두 TTA 없음, clean test 10,000장 + 탐색용 corruption 4종):
  CLIP              : 100 · C Wᵀ   (W = 텍스트 head 또는 16-shot으로 학습된 head)
  DINO proto        : D Pᵀ          (P = 16-shot DINO 특징의 클래스 평균, 정규화)
  CLIP + DINO (λ)   : 100 · C Wᵀ + λ · 50 · D Pᵀ
CLIP 모델: zero-shot, 16-shot base(200 iter, seed 0·1·2 체크포인트 재사용). DINO 프로토타입은 같은 seed의 subset으로 만든다.
λ를 clean test로 고르면 안 되므로 여러 값을 나란히 보여 주기만 한다 (가능성 진단용).

GPU 노드에서 (약 6분):  CUDA_VISIBLE_DEVICES=0 python diag_dino.py
검증: CLIP 행은 기존 결과와 같아야 한다 (zero-shot clean 90.10 / 4종 61.47, 16-shot base clean 94.27 / 4종 66.13).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config import CFG, J, CLASSES
from core.runtime import init_runtime, R, autocast, clip_input
from core.data import kshot_subset
from core.models import load_models, new_classifier
from core.methods import dino_features, get_corruption, head_logits, clip_image_feats
from core import source as S

SEEDS = [0, 1, 2]
TAU_D = 50.0
LAMBDAS = [0.25, 0.5, 1.0, 2.0]
CLIP_PROTO_LAMBDAS = [1.0, 2.0, 4.0]     # ablation: 같은 방식을 CLIP 특징으로 (F_clipproto stage와 같은 λ)


@torch.no_grad()
def clip_feats(clf, x_u8, bs=500):
    out = []
    for s in range(0, len(x_u8), bs):
        with autocast():
            out.append(clf.image_features(clip_input(x_u8[s:s + bs])))
    return torch.cat(out)


def acc(logits, y):
    return 100.0 * float((logits.argmax(1).cpu().numpy() == y).mean())


@torch.no_grad()
def dino_protos(seed):
    idx = kshot_subset(S.D.Ytr, 16, seed)
    Dtr = dino_features(S.D.Xtr[idx]).float().to(R.device)
    ytr = torch.from_numpy(S.D.Ytr[idx]).to(R.device)
    return F.normalize(torch.stack([Dtr[ytr == k].mean(0) for k in range(len(CLASSES))]), dim=-1)


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
    protos = {s: dino_protos(s) for s in SEEDS}
    print(f'DINO 특징 준비 {time.time() - t0:.0f}s', flush=True)

    rows = []

    def add(model, classifier, seed, accs):
        r = dict(model=model, classifier=classifier, seed=seed, **{k: round(v, 2) for k, v in accs.items()})
        r['mean4'] = round(float(np.mean([accs[c] for c in corrs])), 2)
        rows.append(r)

    for mname, mseed in [('zero-shot', None)] + [('16-shot base', s) for s in SEEDS]:
        t0 = time.time()
        clf = new_classifier('zeroshot')[0] if mseed is None else S.get_source_model_k(J(16, mseed))[0]
        with torch.no_grad():
            W = clf.class_weights()
            cf = {k: clip_feats(clf, v[0]) for k, v in sets.items()}
            lc = {k: 100.0 * cf[k] @ W.t() for k in sets}
        add(mname, 'CLIP', '-' if mseed is None else mseed, {k: acc(lc[k], sets[k][1]) for k in sets})
        for ps in (SEEDS if mseed is None else [mseed]):   # zero-shot CLIP은 seed마다 DINO 프로토타입만 바꿔 조합
            ld = {k: sets[k][2].to(R.device) @ protos[ps].t() for k in sets}
            if mseed is None:                               # DINO 단독은 CLIP 모델과 무관 → seed마다 한 번
                add('-', 'DINO proto', ps, {k: acc(ld[k], sets[k][1]) for k in sets})
            for lam in LAMBDAS:
                add(mname, f'CLIP + DINO λ={lam:g}', ps, {k: acc(lc[k] + lam * TAU_D * ld[k], sets[k][1]) for k in sets})
            if mseed is None:                               # 같은 범주 head: CLIP 프로토타입(ablation), Tip-Adapter 캐시(CLIP·DINO)
                with torch.no_grad():
                    idx = kshot_subset(S.D.Ytr, 16, ps)
                    xs, ys = S.D.Xtr[idx], torch.from_numpy(S.D.Ytr[idx]).to(R.device)
                    Ktr_c = clip_image_feats(clf, xs)
                    Ktr_d = dino_features(xs).float().to(R.device)
                    Pc = F.normalize(torch.stack([Ktr_c[ys == c].mean(0) for c in range(10)]), dim=-1)
                    V = F.one_hot(ys, 10).float()
                    lpc = {k: cf[k] @ Pc.t() for k in sets}
                    add('-', 'CLIP proto', ps, {k: acc(lpc[k], sets[k][1]) for k in sets})
                    for lam in CLIP_PROTO_LAMBDAS:
                        add(mname, f'CLIP + CLIP proto λ={lam:g}', ps,
                            {k: acc(lc[k] + lam * TAU_D * lpc[k], sets[k][1]) for k in sets})
                    add(mname, 'CLIP + Tip-Adapter(CLIP)', ps,
                        {k: acc(lc[k] + head_logits(cf[k], Ktr_c, V, 'tip_clip', 1.0), sets[k][1]) for k in sets})
                    add(mname, 'CLIP + DINO 캐시(Tip)', ps,
                        {k: acc(lc[k] + head_logits(sets[k][2].to(R.device), Ktr_d, V, 'tip_dino', 1.0), sets[k][1])
                         for k in sets})
        print(f'  {mname} seed={mseed}: {time.time() - t0:.0f}s', flush=True)
        del clf

    df = pd.DataFrame(rows)
    out = f'{S.exp_dir()}/analysis'
    os.makedirs(out, exist_ok=True)
    df.to_csv(f'{out}/diag_dino.csv', index=False)
    cols = ['clean'] + corrs + ['mean4']
    agg = df.groupby(['model', 'classifier'], sort=False)[cols].mean().round(2)
    agg['n_seeds'] = df.groupby(['model', 'classifier'], sort=False).seed.nunique()
    pd.set_option('display.width', 250)
    print('\n=== seed 평균 (TTA 없음) ===')
    print(agg.to_string())
    print('\n=== seed별 ===')
    print(df.to_string(index=False))
    print(f'\n저장: {out}/diag_dino.csv')


if __name__ == '__main__':
    main()
