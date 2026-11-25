"""continual 경로 점검 (2026-11-25 추가 실험 2). 손상 2종(gaussian → shot)만 이어서 돌리고 결과는 /tmp CSV에 쓴다.
첫 손상(gaussian)은 시작 상태·순서가 episodic과 같으므로 기존 결과와 같아야 한다(GPU 비결정성으로 ±0.1 안).
세 경로를 본다: 라벨 없음(head 없음) / head만(DINO 고정, head query 캐시를 이어 붙임) / ours(DINO 적응).
  CUDA_VISIBLE_DEVICES=0 python smoke_cont.py      (약 6분)
"""
import os, sys, glob
os.environ['PROTOCOL'] = 'continual'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd

from config import CFG, KSHOT_STAGES
from core.runtime import init_runtime
from core.models import load_models
from core import source as S
import core.runner as RN

OUT = '/tmp/smoke_cont.csv'


def main():
    init_runtime(require_gpu=True)
    S.load_clean()
    load_models()
    CFG['CORR_ALL'] = ['gaussian_noise', 'shot_noise']
    if os.path.exists(OUT):
        os.remove(OUT)
    ref = RN.STORE_K()
    store = RN.Store(OUT, ref.keys, ref.cols, ref.defaults)
    jobs = [('라벨 없음', KSHOT_STAGES['Z_base'][0]), ('head만', KSHOT_STAGES['Z_head'][0]), ('ours', KSHOT_STAGES['Z_ours'][0])]
    for name, j0 in jobs:
        j, methods = RN._job(j0)
        common = dict(stage='smoke', **j, cdim_clip_tau_inv=CFG['CDIM_CLIP_TAU_INV'])
        print(f'== {name}: {j["proto_src"]} proto={j["proto"]:g} iters={j["iters"]} seed={j["seed"]}', flush=True)
        todo = [(c, m) for c in CFG['CORR_ALL'] for m in methods]
        RN._run_job(store, common, todo, lambda j=j: S.get_source_model_k(j), RN._extra_kw_fn(j))

    base = f"{CFG['EXP_ROOT']}/{CFG['CKPT_EXP_NAME']}"
    eps = []
    for p in glob.glob(f'{base}/results_kshot*.csv'):
        df = pd.read_csv(p)
        for c, v in ref.defaults.items():
            if c not in df.columns:
                df[c] = v
        eps.append(df)
    ep = pd.concat(eps)
    ep['k'] = [ref.key(r) for r in ep.to_dict('records')]
    sm = store.load()
    print('\n=== continual(2종) vs episodic ===')
    ok = True
    for r in sm.to_dict('records'):
        e = ep[ep.k == ref.key(r)].acc
        ev = float(e.iloc[0]) if len(e) else float('nan')
        diff = r['acc'] - ev
        flag = ''
        if r['corruption'] == 'gaussian_noise':
            # 기준 0.2: 같은 episodic 설정을 다시 돌려도 손상 1종에서 0.05~0.1씩 흔들린다
            #   (라벨 없음 CDIM gaussian: 11-10 67.46, 11-18 67.41, 11-25 continual 첫 손상 67.56)
            flag = 'OK (첫 손상 = episodic)' if abs(diff) <= 0.2 else '❌ 첫 손상이 episodic과 다름'
            ok &= abs(diff) <= 0.2
        print(f"{r['proto_src']:<22} {r['seed']} {r['corruption']:<15} continual {r['acc']:6.2f} | episodic {ev:6.2f} | "
              f"차이 {diff:+.2f} {flag}")
    print('✅ 점검 통과' if ok else '❌ 점검 실패')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
