"""CDIM / BATCLIP clean-source 실험 실행 스크립트 (GPU 노드에서 실행).

사용법 (README 참고):
  python main.py check                          # 환경·데이터·모델 점검 (짧음)
  python main.py sanity                         # zero-shot TTA를 논문 수치와 비교 (먼저!)
  python main.py sweep  [--plan]                # Phase A: 데이터 비율 sweep
  python main.py fullref                        # k-shot 기준값(full 데이터)만 — sweep 전체를 안 돌릴 때
  python main.py phaseb --ratio 1/32            # Phase B: 기준 비율에서 15개 corruption
  python main.py kshot --stage A_base [--plan]  # 16-shot: A_base / A_aug / A_dino / all
  python main.py times                          # 저장된 source 학습 시간 (GPU 불필요)
"""
import os, sys, glob, argparse
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(line_buffering=True)   # nohup 로그에 바로바로 기록
except Exception:
    pass

from config import CFG, KSHOT_STAGES, PAPER_REF


# 결과 CSV key나 체크포인트 이름에 들어가지 않는 설정. 바뀌면 기존 결과와 섞이므로 실행을 멈춘다.
GUARD_KEYS = ['DATASET', 'RESIZE', 'SEVERITY', 'PROTOCOL', 'LOGIT_SCALE', 'SRC_BATCH', 'SRC_EPOCHS_FULL', 'SRC_LR', 'SRC_WD_HEAD',
              'SRC_WARMUP_FRAC', 'TTA_BATCH', 'TTA_LR', 'TTA_WD', 'STREAM_SHUFFLE', 'STREAM_SEED',
              'CDIM_RHO', 'CDIM_ALPHA', 'CDIM_DINO_TAU_INV', 'CDIM_DETACH_PROJ', 'PROTO_TAU_INV',
              'TIP_ALPHA', 'TIP_BETA', 'TDA_POS_CAP', 'TDA_POS_ALPHA', 'TDA_POS_BETA', 'ETA_D_MARGIN', 'SAR_RHO', 'SAR_RESET',
              'PROTO_TTA_MOM']


def setup(require_gpu=True, need_models=True):
    from core.runtime import init_runtime
    from core.source import load_clean, exp_dir
    import json
    init_runtime(require_gpu)
    os.makedirs(exp_dir(), exist_ok=True)
    snap = f'{exp_dir()}/config_snapshot.json'
    if os.path.exists(snap):
        with open(snap) as f:
            old = json.load(f)
        changed = [k for k in GUARD_KEYS if k in old and old[k] != CFG[k]]
        if changed:
            sys.exit(f"❌ {CFG['EXP_NAME']}의 기존 결과와 설정이 다릅니다: "
                     + ', '.join(f'{k} {old[k]} → {CFG[k]}' for k in changed)
                     + '\n   결과가 섞이지 않게 config.py의 EXP_NAME을 새 이름으로 바꾸세요.')
    tmp = f'{snap}.tmp{os.getpid()}'      # 다른 GPU 프로세스와 동시에 써도 깨지지 않게 원자적으로 교체
    with open(tmp, 'w') as f:
        json.dump(CFG, f, indent=1, default=str)
    os.replace(tmp, snap)
    load_clean()
    if need_models:
        from core.models import load_models
        load_models()
    print('결과 저장 위치:', exp_dir(), flush=True)


def cmd_check(a):
    import numpy as np, torch
    setup()
    from core.data import corruption_dir, stratified_stride_subset, kshot_subset, load_corruption
    from core.source import D, eval_clean
    from core.models import new_classifier
    print(f"데이터셋 {CFG['DATASET']} | 손상 데이터: {corruption_dir()} | 결과: {CFG['EXP_NAME']}")
    print('clean train', D.Xtr.shape, '| test', D.Xte.shape)
    for k in [1, 16]:
        idx = kshot_subset(D.Ytr, k, 0)
        bc = np.bincount(D.Ytr[idx])
        print(f'{k}-shot: n={len(idx)}, 클래스 {len(bc)}개, 클래스당 {bc.min()}~{bc.max()}장')
    for c in CFG['CORR_ALL']:                    # 15종 모두 열리고 라벨이 맞는지
        x, y = load_corruption(c, CFG['SEVERITY'])
        assert x.shape == (10000, 32, 32, 3) and np.bincount(y).min() > 0, (c, x.shape)
    print(f"손상 15종 OK (각 10,000장, 클래스 {len(np.unique(y))}개)")
    clf, _ = new_classifier('zeroshot')
    acc = eval_clean(clf)
    print(f"zero-shot clean {CFG['DATASET']} = {acc:.2f}% (CIFAR-10 단일 템플릿 기준 약 88~90%)")
    print(f'GPU 메모리 최대 사용 {torch.cuda.max_memory_allocated() / 2**30:.1f} GB')
    print('✅ check 완료')


def cmd_sanity(a):
    if a.tau is not None:          # 구현 점검 표의 τ⁻¹=20 행처럼 기본값과 다른 τ로 한 번 돌릴 때
        CFG['CDIM_CLIP_TAU_INV'] = float(a.tau)
    setup()
    import pandas as pd
    from core.runner import run_grid, STORE_A
    corrs = a.corrs or ['gaussian_noise']
    methods = a.methods or ['source', 'tent', 'batclip', 'cdim']
    run_grid('sanity', [('zeroshot', 'none', 1, 0)], methods, corrs)
    df = STORE_A().load()
    s = df[(df.phase == 'sanity') & (df.cdim_clip_tau_inv.astype(float) == CFG['CDIM_CLIP_TAU_INV'])]
    rows = []
    for _, r in s.iterrows():
        ref = PAPER_REF.get(r.corruption, {}).get(r.method)
        diff = None if ref is None else r.acc - ref
        rows.append(dict(corruption=r.corruption, method=r.method, ours=round(r.acc, 2), paper=ref,
                         diff=None if diff is None else round(diff, 2),
                         status='' if diff is None else ('OK' if abs(diff) <= 1.5 else 'CHECK')))
    print(f"\nzero-shot clean CIFAR-10 acc: {s.src_clean_acc.iloc[0]:.2f}%")
    print(pd.DataFrame(rows).to_string(index=False))


def cmd_sweep(a):
    from core.runner import plan_phase_a, run_grid, estimate_grid
    jobs = plan_phase_a()
    if a.plan:
        from core.data import ratio_str
        for j in jobs:
            print(j[0], j[1], ratio_str(j[2]), 'seed', j[3])
        setup(require_gpu=False, need_models=False)
        estimate_grid(jobs, CFG['METHODS_SWEEP'], CFG['CORR_SWEEP'], 'A')
        return
    setup()
    run_grid('A', jobs, CFG['METHODS_SWEEP'], CFG['CORR_SWEEP'])
    print('✅ Phase A 완료')


def cmd_fullref(a):
    """k-shot 회복률의 기준값: full 데이터(text_init · same_epochs · 1/1)만 학습·평가 (Phase A의 첫 job과 동일)."""
    from config import FULL_REF, KSHOT_CORRS, KSHOT_METHODS
    from core.runner import run_grid
    setup()
    run_grid('A', [(FULL_REF['head_mode'], FULL_REF['budget'], FULL_REF['ratio'], 0)], KSHOT_METHODS, KSHOT_CORRS)
    print('✅ full-data 기준값 완료')


def cmd_phaseb(a):
    from core.runner import plan_phase_b, run_grid
    ratio = float(Fraction(a.ratio))
    setup()
    run_grid('B', plan_phase_b(ratio), CFG['METHODS_SWEEP'], CFG['CORR_ALL'])
    print('✅ Phase B 완료')


def cmd_kshot(a):
    from core.runner import run_kshot, estimate_kshot
    stages = list(KSHOT_STAGES) if a.stage == 'all' else [a.stage]
    for st in stages:
        assert st in KSHOT_STAGES, f'stage는 {list(KSHOT_STAGES)} 중 하나'
    if a.plan:
        setup(require_gpu=False, need_models=False)
        for st in stages:
            estimate_kshot(st)
        return
    setup()
    failed = []
    for st in stages:
        f = run_kshot(st)
        failed += [f'{st} | {n}' for n in f]
        print(f'✅ {st} 완료' if not f else f'⚠️ {st} 끝 — 실패 {len(f)}개 (위 traceback 참고)', flush=True)
    if failed:
        sys.exit('❌ 실패한 설정:\n  ' + '\n  '.join(failed))


def cmd_times(a):
    import torch, pandas as pd
    from core.source import exp_dir
    rows = []
    for p in sorted(glob.glob(f'{exp_dir()}/source_ckpt*/*.pt')):
        info = torch.load(p, map_location='cpu', weights_only=False)['info']
        it = info.get('n_iters') or 0
        rows.append(dict(ckpt=os.path.basename(p), n_train=info.get('n_train'), iters=it,
                         minutes=round(info['train_time_s'] / 60, 1),
                         sec_per_iter=round(info['train_time_s'] / it, 3) if it else None))
    print(pd.DataFrame(rows).to_string(index=False) if rows else '체크포인트 없음')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('check')
    s = sub.add_parser('sanity'); s.add_argument('--corrs', nargs='+', help='기본 gaussian_noise. 예: --corrs gaussian_noise defocus_blur snow contrast')
    s.add_argument('--methods', nargs='+', choices=['source', 'tent', 'batclip', 'cdim'], help='기본: 4개 전부')
    s.add_argument('--tau', type=float, help=f"CDIM CLIP τ⁻¹ (기본 {CFG['CDIM_CLIP_TAU_INV']:g}). 예: --tau 20 --methods cdim")
    s = sub.add_parser('sweep'); s.add_argument('--plan', action='store_true', help='실행 계획과 예상 시간만 출력')
    sub.add_parser('fullref')
    s = sub.add_parser('phaseb'); s.add_argument('--ratio', required=True, help='예: 1/32')
    s = sub.add_parser('kshot'); s.add_argument('--stage', required=True, help=f"{list(KSHOT_STAGES)} 또는 all")
    s.add_argument('--plan', action='store_true')
    sub.add_parser('times')
    a = p.parse_args()
    dict(check=cmd_check, sanity=cmd_sanity, sweep=cmd_sweep, fullref=cmd_fullref, phaseb=cmd_phaseb, kshot=cmd_kshot, times=cmd_times)[a.cmd](a)


if __name__ == '__main__':
    main()
