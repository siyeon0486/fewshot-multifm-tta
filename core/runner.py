"""실험 실행과 결과 기록. 한 run이 끝날 때마다 CSV에 한 줄씩 추가 → 중단돼도 이어서 실행."""
import os, time, traceback
import numpy as np
import pandas as pd
import torch

from config import CFG, KSHOT_STAGES, KSHOT_METHODS, KSHOT_CORRS, KSHOT_STAGE_CORRS
from core.runtime import R
from core.models import M, set_grad_ckpt, grad_ckpt_on
from core.data import stream_order, seeds_for, ratio_str, kshot_subset
from core.methods import (run_online, get_corruption, source_projection, dino_prototypes, clip_prototypes, tip_cache,
                          head_base, head_flags, dino_prototypes_v, get_head_query)
from core import source as S


def _norm(v):
    try:
        return f'{float(v):.10g}'
    except (TypeError, ValueError):
        return str(v)


class Store:
    def __init__(self, path, keys, cols, defaults=None):
        self.path, self.keys, self.cols, self.defaults = path, keys, cols, defaults or {}

    def migrate(self):
        """나중에 추가한 열(예: proto)이 기존 CSV에 없으면 백업 후 기본값으로 채워 다시 쓴다 (append 전 1회)."""
        if not os.path.exists(self.path):
            return
        df = pd.read_csv(self.path)
        missing = [c for c in self.cols if c not in df.columns]
        if not missing:
            return
        import shutil
        bak = f"{self.path}.bak_{time.strftime('%m%d_%H%M%S')}"
        shutil.copy2(self.path, bak)
        for c in missing:
            df[c] = self.defaults.get(c, '')
        df[self.cols + [c for c in df.columns if c not in self.cols]].to_csv(self.path, index=False)
        print(f'  CSV 열 추가 {missing} (기본값 {[self.defaults.get(c, "") for c in missing]}), 백업: {bak}', flush=True)

    def load(self):
        if not os.path.exists(self.path):
            return pd.DataFrame(columns=self.cols)
        df = pd.read_csv(self.path)
        for c, v in self.defaults.items():   # migrate 전 CSV도 읽을 수 있게 (예: --plan)
            if c not in df.columns:
                df[c] = v
        return df

    def done(self):
        df = self.load()
        return set(tuple(_norm(v) for v in r) for r in df[self.keys].itertuples(index=False)) if len(df) else set()

    def key(self, row):
        return tuple(_norm(row[k]) for k in self.keys)

    def append(self, row):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        pd.DataFrame([{c: row.get(c, '') for c in self.cols}]).to_csv(
            self.path, mode='a', header=not os.path.exists(self.path), index=False)


KEY_A = ['phase', 'head_mode', 'src_aug', 'budget', 'ratio', 'seed', 'method', 'corruption', 'cdim_clip_tau_inv']
STORE_A = lambda: Store(f'{S.exp_dir()}/results.csv', KEY_A,
                        KEY_A + ['n_train', 'n_iters', 'train_time_s', 'src_clean_acc', 'acc', 'skipped',
                                 'tta_time_s', 'diag_last', 'grad_ckpt', 'gpu', 'time'])
# stage는 key에서 뺐다 (2026-11-19): 같은 설정·방법·corruption은 어느 stage에서 돌았든 한 번만 돈다 (stage 열은 기록용으로 남김).
KEY_K = ['head_mode', 'shots', 'iters', 'aug', 'mixup', 'cutmix', 'jsd', 'src_loss', 'align', 'wise', 'cdim_m',
         'proto', 'proto_src', 'seed', 'method', 'corruption', 'cdim_clip_tau_inv']
# 다른 GPU에서 병렬로 돌릴 때는 RESULTS_TAG를 줘서 결과 파일을 나눈다 (예: RESULTS_TAG=gpu1 → results_kshot_gpu1.csv).
# 이미 돈 run 확인도 자기 파일만 본다 → 두 GPU의 stage는 겹치지 않게 나눠야 한다. analyze.py는 results_kshot*.csv를 합쳐 읽는다.
_TAG = f"_{os.environ['RESULTS_TAG']}" if os.environ.get('RESULTS_TAG') else ''
STORE_K = lambda: Store(f'{S.exp_dir()}/results_kshot{_TAG}.csv', KEY_K,
                        ['stage'] + KEY_K + ['n_train', 'train_time_s', 'src_clean_acc', 'acc', 'skipped', 'tta_time_s',
                                             'diag_last', 'grad_ckpt', 'gpu', 'time'],
                        defaults={'proto': 0.0, 'proto_src': 'dino'})   # proto 11-18, proto_src 11-19 추가 (그 전은 0 / dino)


def _gpu_name():
    return torch.cuda.get_device_name() if R.device == 'cuda' else 'cpu'


def with_oom_retry(fn, label):
    """CUDA OOM이면 gradient checkpointing을 켜고 한 번 더 시도 (끝난 run은 CSV에 있어 다시 돌지 않음)."""
    try:
        return fn()
    except torch.cuda.OutOfMemoryError:
        if grad_ckpt_on(M.clip):
            raise
        print(f'  ⚠️ CUDA OOM ({label}) → gradient checkpointing 켜고 재시도', flush=True)
        torch.cuda.empty_cache()
        set_grad_ckpt(M.clip, True)
        return fn()


CDIM_ONLY_KW = ('M_fixed', 'm_mix', 'P_lab', 'lab_mix')   # D2·확장 B: CDIM에만 넘김. 나머지(D4 proto 등)는 모든 방법에 넘김


def _split_extra(extra):
    cdim_kw = {k: v for k, v in extra.items() if k in CDIM_ONLY_KW}
    shared_kw = {k: v for k, v in extra.items() if k not in CDIM_ONLY_KW}
    hflags = shared_kw.pop('head_flags', None)
    return cdim_kw, shared_kw, hflags


def _head_kw(hflags, corr):
    """head 개선 후보 플래그 → run_online 인자. corr는 head query 캐시를 쓸 손상 이름(DINO 적응이면 쓰지 않음)."""
    if not hflags:
        return {}
    dln = 'all' if 'dln' in hflags else 'head' if 'dlnh' in hflags else None   # DINO도 적응 (2026-11-23)
    return dict(head_q=None if dln else get_head_query(corr, hflags), head_ctr='ctr' in hflags, dino_ln=dln,
                head_feat='cp' if 'cp' in hflags else 'cls', head_qf='qf' in hflags)


def _run_job_continual(store, common, todo, build_model, extra_kw_fn=None):
    """continual TTA (2026-11-25 추가 실험 2, PROTOCOL=continual): 손상 15종(CORR_ALL 순서, CoTTA와 같은 순서)을 이어 붙인
    한 스트림을 리셋 없이 돈다. 모델·optimizer·OLA 사전분포·DINO LayerNorm·head 중심화 평균이 손상이 바뀌어도 이어진다.
    각 손상 안의 순서는 episodic과 같은 stream_order라, 첫 손상의 정확도는 episodic 값과 같아야 한다(검증).
    결과는 손상별 정확도로 나눠 같은 CSV 형식으로 쓴다. tta_time_s는 전체 시간을 손상 수로 나눈 값."""
    corrs = list(CFG['CORR_ALL'])
    methods = list(dict.fromkeys(m for _, m in todo))
    order, n_c = stream_order(10000), 10000
    done_now = store.done()

    def go():
        clf, info = build_model()
        base = clf.state_snapshot()
        cdim_kw, shared_kw, hflags = _split_extra(extra_kw_fn(clf) if extra_kw_fn else {})
        for m in methods:
            rows0 = [dict(common, method=m, corruption=c) for c in corrs]
            if all(store.key(r) in done_now for r in rows0):
                continue
            xs, ys, ds, hqs = [], [], [], []
            for c in corrs:
                x, y, d = get_corruption(c)
                assert len(y) == n_c, (c, len(y))
                xs.append(x); ys.append(y); ds.append(d)
                hk = _head_kw(hflags, c)
                hqs.append(hk.get('head_q'))
            X, Y, Dd = np.concatenate(xs), np.concatenate(ys), torch.cat(ds)
            del xs, ds
            ord_all = np.concatenate([order + i * n_c for i in range(len(corrs))])
            hkw = _head_kw(hflags, corrs[-1])
            if hkw and hkw['head_q'] is not None:
                hkw['head_q'] = torch.cat(hqs)
            del hqs
            need_d = m == 'cdim' or 'proto' in shared_kw
            r = run_online(clf, m, base, X, Y, ord_all, Dd if need_d else None, **shared_kw, **hkw,
                           **(cdim_kw if m == 'cdim' else {}), keep_pred=True)
            pred = r.pop('pred')
            assert (pred >= 0).all()
            accs = []
            for i, (c, row0) in enumerate(zip(corrs, rows0)):
                sl = slice(i * n_c, (i + 1) * n_c)
                acc = 100.0 * float((pred[sl] == Y[sl]).mean())
                accs.append(acc)
                if store.key(row0) in done_now:
                    continue
                row = dict(row0, n_train=info.get('n_train'), n_iters=info.get('n_iters'),
                           train_time_s=info.get('train_time_s'), src_clean_acc=info['src_clean_acc'], acc=acc,
                           skipped=r['skipped'] if i == 0 else 0, tta_time_s=r['tta_time_s'] / len(corrs),
                           diag_last=r.get('diag_last', ''), grad_ckpt=grad_ckpt_on(M.clip), gpu=_gpu_name(),
                           time=time.strftime('%Y-%m-%d %H:%M'))
                store.append(row)
                done_now.add(store.key(row0))
                print(f"   {c:<18} {m:<8} {acc:6.2f}%  (continual)", flush=True)
            warn = f"  ⚠️ fp16 skipped {r['skipped']} steps" if r['skipped'] else ''
            print(f"   == continual {m} 평균 {np.mean(accs):.2f}%  (전체 {r['tta_time_s']:.0f}s){warn}", flush=True)
            del X, Y, Dd, pred
        del clf
    with_oom_retry(go, str(common))
    if R.device == 'cuda':
        torch.cuda.empty_cache()


def _run_job(store, common, todo, build_model, extra_kw_fn=None):
    """extra_kw_fn(clf): source 모델이 준비된 뒤 TTA에 넘길 추가 인자 (D2의 고정 투영 행렬, D4의 DINO 프로토타입)."""
    if CFG['PROTOCOL'] == 'continual':
        return _run_job_continual(store, common, todo, build_model, extra_kw_fn)
    order = stream_order(10000)
    done_now = store.done()

    def go():
        clf, info = build_model()
        base = clf.state_snapshot()
        extra = extra_kw_fn(clf) if extra_kw_fn else {}
        cdim_kw = {k: v for k, v in extra.items() if k in CDIM_ONLY_KW}
        shared_kw = {k: v for k, v in extra.items() if k not in CDIM_ONLY_KW}
        hflags = shared_kw.pop('head_flags', None)       # head 개선 후보: corruption마다 head query 특징을 따로 넘김
        for c, m in todo:
            row0 = dict(common, method=m, corruption=c)
            if store.key(row0) in done_now:
                continue
            x, y, d = get_corruption(c)
            need_d = m == 'cdim' or 'proto' in shared_kw
            hkw = {}
            if hflags:
                dln = 'all' if 'dln' in hflags else 'head' if 'dlnh' in hflags else None   # DINO도 적응 (2026-11-23)
                hkw = dict(head_q=None if dln else get_head_query(c, hflags), head_ctr='ctr' in hflags, dino_ln=dln,
                           head_feat='cp' if 'cp' in hflags else 'cls', head_qf='qf' in hflags)
            r = run_online(clf, m, base, x, y, order, d if need_d else None, **shared_kw, **hkw,
                           **(cdim_kw if m == 'cdim' else {}))
            row = dict(row0, n_train=info.get('n_train'), n_iters=info.get('n_iters'),
                       train_time_s=info.get('train_time_s'), src_clean_acc=info['src_clean_acc'], **r,
                       grad_ckpt=grad_ckpt_on(M.clip), gpu=_gpu_name(), time=time.strftime('%Y-%m-%d %H:%M'))
            store.append(row)
            done_now.add(store.key(row0))
            warn = f"  ⚠️ fp16 skipped {r['skipped']} steps" if r['skipped'] else ''
            print(f"   {c:<18} {m:<8} {r['acc']:6.2f}%  ({r['tta_time_s']:.0f}s){warn}", flush=True)
        del clf
    with_oom_retry(go, str(common))
    if R.device == 'cuda':
        torch.cuda.empty_cache()


# ---------------- sanity / Phase A / Phase B ----------------
def run_grid(phase, jobs, methods, corruptions):
    store = STORE_A()
    for head_mode, budget, ratio, seed in jobs:
        common = dict(phase=phase, head_mode=head_mode, src_aug=CFG['SRC_AUG'], budget=budget,
                      ratio=ratio, seed=seed, cdim_clip_tau_inv=CFG['CDIM_CLIP_TAU_INV'])
        done = store.done()
        todo = [(c, m) for c in corruptions for m in methods
                if store.key(dict(common, method=m, corruption=c)) not in done]
        if not todo:
            continue
        print(f'== {phase} | {head_mode} | {budget} | ratio={ratio_str(ratio)} | seed={seed} | {len(todo)} runs', flush=True)
        _run_job(store, common, todo, lambda: S.get_source_model(head_mode, budget, ratio, seed))


def plan_phase_a():
    jobs, hm = [], CFG['HEAD_MODE_MAIN']
    for r in CFG['RATIOS']:
        for s in seeds_for(r):
            jobs.append((hm, 'same_epochs', r, s))
    if 'same_iters' in CFG['BUDGETS']:
        for r in CFG['SAME_ITERS_RATIOS']:
            jobs.append((hm, 'same_iters', r, 0))
    if CFG['RUN_NORM_ONLY_ABLATION']:
        for r in CFG['RATIOS']:
            jobs.append(('norm_only', 'same_epochs', r, 0))
    return jobs


def plan_phase_b(ratio):
    hm = CFG['HEAD_MODE_MAIN']
    return [(hm, 'same_epochs', 1, 0)] + [(hm, 'same_epochs', ratio, s) for s in seeds_for(ratio)]


# ---------------- k-shot ----------------
def _job(j):
    """KSHOT_STAGES의 job → (결과 key에 들어갈 설정, 돌릴 TTA 방법)."""
    j = dict(j)
    return j, j.pop('methods', None) or KSHOT_METHODS


def _extra_kw_fn(j):
    """TTA 전에 clean 학습 subset(k-shot)으로 미리 계산해 넘기는 것.
    D2: source 모델로 투영 행렬 M 추정(CDIM만).  D4: 라벨로 DINO 클래스 프로토타입(모든 방법)."""
    use_d2 = j['cdim_m'] in ('src', 'mix')
    use_lab = j['cdim_m'] in ('lab', 'lab_mix')          # 확장 B
    use_d4 = j.get('proto', 0.0) > 0
    if not (use_d2 or use_lab or use_d4):
        return None

    def fn(clf):
        idx = kshot_subset(S.D.Ytr, j['shots'], j['seed'])
        out = {}
        if use_d2:
            out.update(M_fixed=source_projection(clf, S.D.Xtr[idx]), m_mix={'src': 0.0, 'mix': 0.5}[j['cdim_m']])
        if use_lab:
            out.update(P_lab=dino_prototypes(S.D.Xtr[idx], S.D.Ytr[idx]), lab_mix={'lab': 1.0, 'lab_mix': 0.5}[j['cdim_m']])
        if use_d4:
            src, xs, ys = j.get('proto_src', 'dino'), S.D.Xtr[idx], S.D.Ytr[idx]
            if src in ('dino', 'dino_tta'):
                P, V = dino_prototypes(xs, ys), None
            elif head_base(src) == 'dino':          # 'dino+cp+ctr+qf+a8' 등 head 개선 후보 (2026-11-20, diag_head2.py)
                P, V = dino_prototypes_v(xs, ys, head_flags(src), j['seed']), None
                out['head_flags'] = head_flags(src)
            elif src == 'clip':
                P, V = clip_prototypes(clf, xs, ys), None
            else:                                   # tip_clip · tip_dino: Tip-Adapter 캐시
                P, V = tip_cache(src, clf, xs, ys)
            out.update(proto=P, proto_vals=V, proto_w=j['proto'], proto_src=src)
        return out
    return fn


def run_kshot(stage):
    """설정 하나가 오류로 멈춰도 traceback만 남기고 다음 설정으로 넘어간다. 실패한 설정 목록을 돌려준다."""
    store, failed = STORE_K(), []
    store.migrate()
    for j0 in KSHOT_STAGES[stage]:
        j, methods = _job(j0)
        common = dict(stage=stage, **j, cdim_clip_tau_inv=CFG['CDIM_CLIP_TAU_INV'])
        done = store.done()
        todo = [(c, m) for c in KSHOT_STAGE_CORRS.get(stage, KSHOT_CORRS) for m in methods
                if store.key(dict(common, method=m, corruption=c)) not in done]
        if not todo:
            continue
        name = (f"{j['shots']}-shot it={j['iters']} aug={j['aug']} mixup={j['mixup']:g} "
                f"cutmix={j['cutmix']:g} jsd={j['jsd']:g} loss={j['src_loss']} align={j['align']:g} "
                f"wise={j['wise']:g} cdim_m={j['cdim_m']} proto={j['proto']:g}({j['proto_src']}) seed={j['seed']}")
        print(f"== {stage} | {name} | {len(todo)} runs", flush=True)
        try:
            _run_job(store, common, todo, lambda j=j: S.get_source_model_k(j), _extra_kw_fn(j))
        except Exception:
            traceback.print_exc()
            print(f'  ❌ 오류로 이 설정을 건너뜀: {name}', flush=True)
            failed.append(name)
            if R.device == 'cuda':
                torch.cuda.empty_cache()
    return failed


def _run_time_table():
    t = {'source': 15.0, 'tent': 45.0, 'batclip': 50.0, 'cdim': 50.0}   # V100 초기 추정치(초), 실측으로 대체
    gpu = _gpu_name()
    for st in (STORE_A(), STORE_K()):
        df = st.load()
        if len(df):
            if gpu != 'cpu' and 'gpu' in df:   # GPU 노드: 같은 GPU 실측만. master(GPU 없음): 서버 실측 전체
                df = df[df.gpu.astype(str) == gpu]
            for m, v in df.groupby('method').tta_time_s.mean().items():
                if not np.isnan(v): t[m] = v
    return t


def estimate_grid(jobs, methods, corruptions, phase):
    store, t = STORE_A(), _run_time_table()
    done = store.done(); secs = 0; n = 0
    for head_mode, budget, ratio, seed in jobs:
        common = dict(phase=phase, head_mode=head_mode, src_aug=CFG['SRC_AUG'], budget=budget,
                      ratio=ratio, seed=seed, cdim_clip_tau_inv=CFG['CDIM_CLIP_TAU_INV'])
        for c in corruptions:
            for m in methods:
                if store.key(dict(common, method=m, corruption=c)) not in done:
                    secs += t.get(m, 50); n += 1
    print(f'{phase}: 남은 run {n}개 ≈ {secs/3600:.1f}시간 (source 학습 시간 제외)')


def estimate_kshot(stage):
    store, t = STORE_K(), _run_time_table()
    done = store.done(); secs = 0; n = 0; new_models = set()   # 경로 기준 (D2처럼 같은 체크포인트를 재사용하는 job은 한 번만)
    for j0 in KSHOT_STAGES[stage]:
        j, methods = _job(j0)
        common = dict(stage=stage, **j, cdim_clip_tau_inv=CFG['CDIM_CLIP_TAU_INV'])
        left = [m for c in KSHOT_STAGE_CORRS.get(stage, KSHOT_CORRS) for m in methods
                if store.key(dict(common, method=m, corruption=c)) not in done]
        n += len(left); secs += sum(t.get(m, 50) for m in left)
        if left and j['iters'] > 0 and not os.path.exists(S.ckpt_path_k(j)):
            new_models.add(S.ckpt_path_k(j))
    print(f'{stage}: 남은 run {n}개 ≈ {secs/3600:.1f}시간 + 새로 학습할 source 모델 {len(new_models)}개')
