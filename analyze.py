"""결과 분석 (GPU 불필요, master에서 실행 가능).
  python analyze.py            → 표는 화면에 출력, 표(csv)·그래프(png)는 experiments/<EXP_NAME>/analysis/ 에 저장
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (CFG, PAPER_REF, KSHOT_CORRS, KSHOT_METHODS, FULL_REF, ITERS_MAIN, ITERS_RQ2, FT_ITERS, FULL_ITERS,
                    NEW_HEAD, OURS_HEAD, CHOSEN_HEAD)

EXP = f"{CFG['EXP_ROOT']}/{CFG['EXP_NAME']}"
OUT = f'{EXP}/analysis'
TAU = CFG['CDIM_CLIP_TAU_INV']
METHOD_STYLE = {   # 고정 순서 categorical 팔레트, 색 + 마커 이중 인코딩
    'source':  ('#2a78d6', 'o', 'Source (no TTA)'),
    'tent':    ('#eb6834', 's', 'TENT'),
    'batclip': ('#1baf7a', '^', 'BATCLIP'),
    'cdim':    ('#eda100', 'D', 'CDIM'),
}


def rstr(r):
    r = float(r); return '1' if r == 1 else f'1/{int(round(1 / r))}'


def load(name, filter_tau=True):
    """name이 results_kshot.csv면 GPU별 파일(results_kshot_gpu1.csv 등)까지 합쳐 읽는다."""
    import glob
    paths = sorted(glob.glob(f'{EXP}/results_kshot*.csv')) if name == 'results_kshot.csv' else [f'{EXP}/{name}']
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        return None
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df[df.cdim_clip_tau_inv.astype(float) == TAU] if filter_tau else df
    if name == 'results_kshot.csv':
        # GPU별 파일(RESULTS_TAG)은 중복 확인을 자기 파일 안에서만 해서, 다른 파일에 이미 있는 설정을 다시 돌릴 수 있다
        # (2026-11-20 L_curve_ours·L_p5에서 발생). 같은 설정·seed·방법·corruption이 여러 번 있으면 가장 먼저 기록된 값만 쓴다
        # → 재실행 때문에 이미 보고한 표의 숫자가 바뀌지 않는다.
        df = _kfill(df)
        key = KCFG + ['seed', 'method', 'corruption', 'cdim_clip_tau_inv']
        n0 = len(df)
        df = df.sort_values('time', kind='stable').drop_duplicates(key, keep='first')
        if len(df) < n0:
            print(f'[load] 중복 run {n0 - len(df)}개는 먼저 기록된 값만 사용')
    return df


def section(t):
    print('\n' + '=' * 70 + f'\n{t}\n' + '=' * 70)


# ---------------- sanity ----------------
def sanity(raw):
    """구현 점검 표. τ⁻¹는 CDIM에만 의미가 있어 CDIM은 τ별로 모두, 나머지 방법은 기본 τ 행만 보여 준다."""
    s = raw[raw.phase == 'sanity'].copy()
    if s.empty: return
    s['tau'] = s.cdim_clip_tau_inv.astype(float)
    s = s[(s.method == 'cdim') | (s.tau == TAU)]
    if s.empty: return
    s['_m'] = s.method.map({m: i for i, m in enumerate(METHOD_STYLE)})
    s['_c'] = s.corruption.map({c: i for i, c in enumerate(CFG['CORR_ALL'])})
    s = s.sort_values(['_m', 'tau', '_c'])
    section('Sanity check — 구현 점검 표 (zero-shot TTA vs CDIM Table 1)')
    print(f"zero-shot clean CIFAR-10 acc: {s.src_clean_acc.iloc[0]:.2f}%")
    rows = []

    def add(corr, m, tau, ours, ref):
        d = None if ref is None else ours - ref
        rows.append(dict(corruption=corr, method=m, tau_inv=f'{tau:g}' if m == 'cdim' else '',
                         ours=round(ours, 2), paper=None if ref is None else round(ref, 2),
                         diff=None if d is None else round(d, 2),
                         status='' if d is None else ('OK' if abs(d) <= 1.5 else 'CHECK')))

    for (_, m, tau), g in s.groupby(['_m', 'method', 'tau'], sort=True):
        for _, r in g.iterrows():
            add(r.corruption, m, tau, r.acc, PAPER_REF.get(r.corruption, {}).get(m))
        if set(PAPER_REF) <= set(g.corruption):          # 탐색용 4종이 다 있으면 평균 행 추가
            g4 = g[g.corruption.isin(PAPER_REF)]
            add(f'mean({len(PAPER_REF)})', m, tau, g4.acc.mean(), float(np.mean([PAPER_REF[c][m] for c in PAPER_REF])))
    t = pd.DataFrame(rows); print(t.to_string(index=False)); t.to_csv(f'{OUT}/sanity.csv', index=False)


# ---------------- Phase A ----------------
def summarize_A(df, phase='A'):
    d = df[df.phase == phase].copy()
    d['ratio'] = d.ratio.astype(float)
    per_seed = d.groupby(['head_mode', 'budget', 'ratio', 'seed', 'method']).agg(
        acc=('acc', 'mean'), n_corr=('corruption', 'nunique'), src_clean_acc=('src_clean_acc', 'first'),
        n_train=('n_train', 'first'), n_iters=('n_iters', 'first')).reset_index()
    agg = per_seed.groupby(['head_mode', 'budget', 'ratio', 'method']).agg(
        mean=('acc', 'mean'), std=('acc', 'std'), n_seeds=('seed', 'nunique'),
        n_corr=('n_corr', 'min'), src_clean=('src_clean_acc', 'mean'),
        n_train=('n_train', 'mean'), n_iters=('n_iters', 'mean')).reset_index()
    agg['std'] = agg['std'].fillna(0.0)
    return agg


def knee(agg, head_mode, budget='same_epochs', methods=('batclip', 'cdim')):
    out = []
    for m in methods:
        a = agg[(agg.head_mode == head_mode) & (agg.budget == budget) & (agg.method == m)].sort_values('ratio', ascending=False)
        if a.empty or 1.0 not in set(a.ratio): continue
        full = a[a.ratio == 1.0]['mean'].iloc[0]; pick = None
        for _, r in a.iterrows():
            drop = full - r['mean']
            if drop >= CFG['KNEE_DROP'] and drop > 2 * r['std']:
                pick = r; break
        out.append(dict(method=m, full=round(full, 2), knee_ratio=None if pick is None else rstr(pick.ratio),
                        knee_acc=None if pick is None else round(pick['mean'], 2),
                        drop=None if pick is None else round(full - pick['mean'], 2)))
    return pd.DataFrame(out)


def plot_A(agg, head_mode):
    budgets = list(agg[agg.head_mode == head_mode].budget.unique())
    fig, axes = plt.subplots(1, len(budgets), figsize=(6 * len(budgets), 4.2), sharey=True, squeeze=False)
    for ax, bd in zip(axes[0], budgets):
        t = agg[(agg.head_mode == head_mode) & (agg.budget == bd)]
        if bd == 'same_iters':   # full 데이터는 두 조건이 동일 → 기준점으로 함께 표시
            t = pd.concat([t, agg[(agg.head_mode == head_mode) & (agg.budget == 'same_epochs') & (agg.ratio == 1.0)]])
        for m, (col, mk, label) in METHOD_STYLE.items():
            a = t[t.method == m].sort_values('ratio')
            if a.empty: continue
            xs = np.log2(a.ratio.values)
            ax.plot(xs, a['mean'], color=col, marker=mk, ms=6, lw=2, label=label)
            ax.fill_between(xs, a['mean'] - a['std'], a['mean'] + a['std'], color=col, alpha=0.15, lw=0)
        ticks = sorted(np.log2(t.ratio.unique()))
        ax.set_xticks(ticks); ax.set_xticklabels([rstr(2 ** v) for v in ticks])
        ax.set_title(f'{head_mode} · {bd}', fontsize=11); ax.set_xlabel('Source training data ratio')
        ax.grid(axis='y', alpha=0.25)
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    axes[0][0].set_ylabel('CIFAR-10-C accuracy (%, mean over corruptions)')
    axes[0][-1].legend(frameon=False, loc='lower right')
    plt.tight_layout(); p = f'{OUT}/phaseA_{head_mode}.png'; plt.savefig(p, dpi=150); plt.close()
    print('  그래프 저장:', p)


def phase_A(df):
    if df[df.phase == 'A'].empty: return
    section('Phase A — 데이터 비율 sweep')
    agg = summarize_A(df); agg.to_csv(f'{OUT}/phaseA_summary.csv', index=False)
    for hm in agg.head_mode.unique():
        for bd in agg[agg.head_mode == hm].budget.unique():
            t = agg[(agg.head_mode == hm) & (agg.budget == bd)].copy()
            t['val'] = t.apply(lambda r: f"{r['mean']:.2f}±{r['std']:.2f}" if r.n_seeds > 1 else f"{r['mean']:.2f}", axis=1)
            tab = t.pivot(index='ratio', columns='method', values='val').sort_index(ascending=False)
            tab.insert(0, 'src_clean', t.groupby('ratio').src_clean.first().sort_index(ascending=False).round(2))
            tab.index = [rstr(r) for r in tab.index]
            print(f'\n### {hm} | {bd}  (corruption {t.n_corr.min()}개 평균)'); print(tab.to_string())
    print('\n### 기준 비율 후보 (main head, same_epochs)')
    print(knee(agg, CFG['HEAD_MODE_MAIN']).to_string(index=False))
    for hm in agg.head_mode.unique():
        plot_A(agg, hm)
    b = df[df.phase == 'B']
    if not b.empty:
        section('Phase B — 15개 corruption')
        aggb = summarize_A(df, 'B'); aggb.to_csv(f'{OUT}/phaseB_summary.csv', index=False)
        print(aggb.round(2).to_string(index=False))


# ---------------- 16-shot (설계 문서 RQ1·RQ3 표) ----------------
KCFG = ['head_mode', 'shots', 'iters', 'aug', 'mixup', 'cutmix', 'jsd', 'src_loss', 'align', 'wise', 'cdim_m', 'proto',
        'proto_src']
KDEFAULT = dict(mixup=0.0, cutmix=0.0, jsd=0.0, src_loss='ce', align=0.0, wise=1.0, cdim_m='batch', proto=0.0,
                proto_src='dino')
METHOD_ORDER = ('source', 'tent', 'eta', 'sar', 'tda', 'batclip', 'cdim')
HEAD_NAME = {'dino': ('Few-shot DINO head', 14), 'clip': ('[ablation] CLIP 프로토타입', 15),   # 'dino' = 예전 이름 D4
             'tip_clip': ('[baseline] Tip-Adapter (CLIP 캐시)', 16), 'tip_dino': ('[baseline] DINO 캐시 (Tip 방식)', 17),
             'dino_tta': ('확장 A: Few-shot DINO head + 프로토타입 TTA 갱신', 19)}
CDIM_M_NAME = {'src': ('D2 source M 고정', 11), 'mix': ('D2 M 혼합', 12),
               'lab': ('확장 B: CDIM DINO 분기 = 라벨 프로토타입', 20), 'lab_mix': ('확장 B: CDIM DINO 분기 + 라벨 프로토타입 혼합', 21)}
AUG_NAME = {'none': ('base', 0), 'basic': ('A1 crop+flip', 1), 'ta': ('A2 safe TrivialAugment', 2),
            'augmix': ('A3 AugMix', 3)}


def _kfill(dk):
    """예전 형식 CSV에 없는 설정 열을 기본값으로 채움."""
    dk = dk.copy()
    for c, v in KDEFAULT.items():
        if c not in dk:
            dk[c] = v
        else:                  # GPU별 CSV를 합칠 때 한쪽에만 있는 열은 NaN → 기본값 (NaN이면 groupby에서 행이 빠짐)
            dk[c] = dk[c].fillna(v)
    return dk


def _is_plain(d):
    """증강·섞기·DINO 변형·WiSE가 없는 16-shot base 설정 (iteration은 따지지 않음).
    열은 반드시 d['...']로 읽는다. d.align은 열이 아니라 DataFrame.align 메서드다."""
    return ((d['shots'] == 16) & (d['aug'] == 'none') & (d['mixup'] == 0) & (d['cutmix'] == 0) & (d['jsd'] == 0)
            & (d['src_loss'] == 'ce') & (d['align'] == 0) & (d['wise'] == 1.0) & (d['cdim_m'] == 'batch')
            & (d['proto'] == 0) & (d['head_mode'] == CFG['HEAD_MODE_MAIN']))


def variant(r):
    """설정 → (문서 표의 행 이름, 정렬 순서). 순서는 문서 표와 같음. r['align']처럼 대괄호로 읽는다(Series.align 충돌)."""
    cdim_m, src_loss, align, aug = r['cdim_m'], r['src_loss'], float(r['align']), r['aug']
    jsd, cutmix, mixup = float(r['jsd']), float(r['cutmix']), float(r['mixup'])
    iters, wise, shots = int(r['iters']), float(r['wise']), int(r['shots'])
    parts = []                      # 증강(A) 부분
    if jsd > 0:
        parts.append(('A3 AugMix + JSD', 4))
    elif cutmix > 0:
        parts.append(('A4 CutMix', 6))
    elif mixup > 0:
        parts.append(('A4 Mixup', 5))
    elif aug != 'none':
        parts.append(AUG_NAME.get(aug, (aug, 9)))
    if src_loss == 'ens':           # DINO 활용(D) 부분
        parts.append(('D1 앙상블 logit 학습', 10))
    if cdim_m != 'batch':
        parts.append(CDIM_M_NAME.get(cdim_m, (cdim_m, 22)))
    if align > 0:
        parts.append((f'D3 관계 정렬 (λ={align:g})', 13))
    proto = float(r['proto'])
    if proto > 0:
        src = str(r['proto_src']); base = src.split('@')[0]      # 'tip_dino@8' = β 8 (core.methods.head_base)
        hname, hg = HEAD_NAME.get(base, (base, 18))
        if base.startswith('tip'):                                 # Tip 캐시: proto = α / TIP_ALPHA
            beta = float(src.split('@')[1]) if '@' in src else CFG['TIP_BETA']
            parts.append((f"{hname} (α={proto * CFG['TIP_ALPHA']:g}, β={beta:g})", hg))
        else:
            parts.append((f'{hname} (λ={proto:g})', hg))
    if not parts:
        name, g = AUG_NAME['none']
    elif len(parts) == 1:
        name, g = parts[0]
    else:                           # 조합(문서 Phase B)은 단일 변형 뒤에
        name, g = ' + '.join(p[0] for p in parts), 20 + max(p[1] for p in parts)
    if iters == 0:                  # CLIP 학습 없음 (zero-shot CLIP 그대로, shots는 프로토타입 등에 쓴 라벨 수)
        name = 'zero-shot CLIP' + ('' if not parts else ' + ' + name)
    elif g == 0 or iters != ITERS_MAIN:
        name += f' (iter {iters})'
    if wise != 1.0:
        name += f' · WiSE {wise:g}'
    if shots >= 5000:
        name = 'full data: ' + name
    elif shots != 16:
        name = f'{shots}-shot ' + name
    return name, (shots != 16, shots, iters == 0, g, iters != ITERS_MAIN, iters, wise)


def kshot(dk, da):
    if dk is None or dk.empty: return
    dk = _kfill(dk)
    section(f'16-shot — RQ1·RQ3 표 (corruption {len(KSHOT_CORRS)}종 평균, Δ = 같은 방법의 base(iter {ITERS_MAIN}) 대비 %p)')
    per_seed = dk[dk.corruption.isin(KSHOT_CORRS)].groupby(KCFG + ['seed', 'method']).agg(
        acc=('acc', 'mean'), n_corr=('corruption', 'nunique'), clean=('src_clean_acc', 'first'),
        train_min=('train_time_s', lambda v: v.iloc[0] / 60)).reset_index()
    per_seed = per_seed[per_seed.n_corr == len(KSHOT_CORRS)]      # 4종이 모두 끝난 것만
    if per_seed.empty:
        print(f'corruption {len(KSHOT_CORRS)}종이 모두 끝난 설정이 아직 없습니다.'); return
    agg = per_seed.groupby(KCFG + ['method']).agg(
        mean=('acc', 'mean'), std=('acc', 'std'), n_seeds=('seed', 'nunique'),
        clean=('clean', 'mean'), train_min=('train_min', 'mean')).reset_index()
    agg['std'] = agg['std'].fillna(0.0)
    # base(iter 200)의 seed별 정확도. Δ는 같은 seed끼리 짝지어 계산한다(subset 차이 상쇄, seed 0만 있는 변형도 공정하게 비교).
    pb = per_seed[_is_plain(per_seed) & (per_seed['iters'] == ITERS_MAIN)].set_index(['seed', 'method'])['acc']
    MS = [m for m in METHOD_ORDER if m in set(per_seed['method'])]   # 결과가 있는 방법만 열로 (TENT는 F_tent 이후)

    ref_rows, full_ref = [], {}
    if da is not None:   # 참고 행: zero-shot(sanity), full 데이터(비율 sweep의 ratio 1)
        a = da[da.corruption.isin(KSHOT_CORRS)]
        full = a[(a.phase == 'A') & (a.head_mode == FULL_REF['head_mode']) & (a.budget == FULL_REF['budget'])
                 & (a.ratio.astype(float) == FULL_REF['ratio'])]
        for name, d in [('zero-shot', a[a.phase == 'sanity']), ('full data (참고)', full)]:
            if d.empty: continue
            row = {'변형': name, 'clean': round(float(d.src_clean_acc.iloc[0]), 2)}
            for m in MS:
                g = d[d.method == m]
                ok = g.corruption.nunique() == len(KSHOT_CORRS)
                row[m] = f'{g.acc.mean():.2f}' if ok else ''
                if ok and d is full:
                    full_ref[m] = g.acc.mean()
            ref_rows.append(row)

    def paired(a, b):   # a, b: seed → acc. 공통 seed에서 a − b의 평균(±표준편차)
        common = [s for s in a.index if s in b.index]
        if not common: return ''
        d = np.array([a[s] - b[s] for s in common])
        return f'{d.mean():+.2f}' + (f'±{d.std(ddof=1):.2f}' if len(d) > 1 else '')

    vrows = []
    for _, g in per_seed.groupby(KCFG, sort=False):
        name, order = variant(g.iloc[0])
        acc = {m: g[g['method'] == m].set_index('seed')['acc'] for m in MS}
        row = {'변형': name, 'clean': round(g['clean'].mean(), 2)}
        for m in MS:
            a = acc[m]
            row[m] = '' if a.empty else _cell({'mean': a.mean(), 'std': a.std() if len(a) > 1 else 0.0, 'n_seeds': len(a)})
        for m in [x for x in MS if x != 'source']:
            base_m = pb.xs(m, level='method') if m in pb.index.get_level_values('method') else pd.Series(dtype=float)
            row[f'Δ{m}'] = paired(acc[m], base_m)
        row['cdim−batclip'] = paired(acc['cdim'], acc['batclip'])
        row['seeds'] = int(g['seed'].nunique())
        row['train_min'] = round(g['train_min'].mean(), 1)
        vrows.append((order, row))
    out = pd.DataFrame(ref_rows + [r for _, r in sorted(vrows, key=lambda t: t[0])]).fillna('')
    print(out.to_string(index=False)); out.to_csv(f'{OUT}/kshot_summary.csv', index=False)

    it = agg[_is_plain(agg)]     # base의 학습량 곡선 (50·200·400·782 iter)
    if it.iters.nunique() >= 2:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for m, (col, mk, label) in METHOD_STYLE.items():
            d = it[it.method == m].sort_values('iters')
            if d.empty: continue
            xs = np.log2(d.iters.values.astype(float))
            ax.plot(xs, d['mean'], color=col, marker=mk, ms=6, lw=2, label=label)
            ax.fill_between(xs, d['mean'] - d['std'], d['mean'] + d['std'], color=col, alpha=0.15, lw=0)
            if m in full_ref:
                ax.axhline(full_ref[m], color=col, lw=1, ls='--', alpha=0.7)
        ticks = np.log2(np.sort(it.iters.unique().astype(float)))
        ax.set_xticks(ticks); ax.set_xticklabels([f'{int(round(2 ** t))}' for t in ticks])
        ax.set_xlabel('16-shot source training iterations (dashed = full-data reference)')
        ax.set_ylabel('CIFAR-10-C accuracy (%)')
        ax.grid(axis='y', alpha=0.25); ax.legend(frameon=False, loc='lower right')
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
        plt.tight_layout(); p = f'{OUT}/kshot_iters.png'; plt.savefig(p, dpi=150); plt.close()
        print('  그래프 저장:', p)


# ---------------- 최종 비교 (corruption 15종) ----------------
def final15(dk):
    """corruption 15종이 모두 끝난 설정만 (H_final · G_fewshot · G_tta · I_ext). 방법마다 seed 평균±표준편차."""
    if dk is None or dk.empty: return
    dk = _kfill(dk)
    C = CFG['CORR_ALL']
    d = dk[dk.corruption.isin(C)]
    per_seed = d.groupby(KCFG + ['seed', 'method']).agg(
        acc=('acc', 'mean'), n_corr=('corruption', 'nunique'), clean=('src_clean_acc', 'first')).reset_index()
    per_seed = per_seed[per_seed.n_corr == len(C)]
    if per_seed.empty: return
    section(f'최종 비교 — corruption {len(C)}종 평균 (severity {CFG["SEVERITY"]}, 15종이 다 끝난 설정만)')
    MS = [m for m in METHOD_ORDER if m in set(per_seed['method'])]
    rows = []
    for _, g in per_seed.groupby(KCFG, sort=False):
        name, order = variant(g.iloc[0])
        row = {'변형': name, 'clean': round(g['clean'].mean(), 2)}
        for m in MS:
            a = g[g['method'] == m]['acc']
            row[m] = '' if a.empty else _cell({'mean': a.mean(), 'std': a.std() if len(a) > 1 else 0.0, 'n_seeds': len(a)})
        row['seeds'] = int(g['seed'].nunique())
        rows.append((order, row))
    out = pd.DataFrame([r for _, r in sorted(rows, key=lambda t: t[0])]).fillna('')
    print(out.to_string(index=False)); out.to_csv(f'{OUT}/final15_summary.csv', index=False)

    # corruption별: zero-shot CLIP 그대로 vs + Few-shot DINO head (같은 TTA 방법끼리, seed 평균)
    done = per_seed[KCFG + ['seed', 'method']]
    d = d.merge(done, on=KCFG + ['seed', 'method'])
    z = d[(d['iters'] == 0) & (d['shots'] == 16) & (d['cdim_m'] == 'batch')
          & ((d['proto'] == 0) | ((d['proto'] == 1.0) & (d['proto_src'] == 'dino')))]
    if z['proto'].nunique() < 2: return
    t = z.groupby(['corruption', 'method', 'proto']).acc.mean().unstack(['method', 'proto'])
    cols, names = [], []
    for m in MS:
        for p, tag in ((0.0, ''), (1.0, '+head')):
            if (m, p) in t.columns:
                cols.append((m, p)); names.append(m + tag)
    t = t[cols]; t.columns = names
    t = t.reindex([c for c in C if c in t.index])
    t.loc['mean'] = t.mean()
    print(f'\n### corruption별 — zero-shot CLIP 그대로 vs + Few-shot DINO head (λ=1)')
    print(t.round(2).to_string())
    t.round(2).to_csv(f'{OUT}/final15_per_corruption.csv')
    wins = {m: int((t[m + '+head'].drop('mean') > t[m].drop('mean')).sum()) for m in MS
            if m in t and m + '+head' in t}
    print('head가 이긴 corruption 수 (/15): ' + ', '.join(f'{m} {w}' for m, w in wins.items()))


# ---------------- 논문 형식 main table (행 = 방법, 열 = corruption) ----------------
CORR_SHORT = {'gaussian_noise': 'Gauss.', 'shot_noise': 'Shot', 'impulse_noise': 'Impul.', 'defocus_blur': 'Defoc.',
              'glass_blur': 'Glass', 'motion_blur': 'Motion', 'zoom_blur': 'Zoom', 'snow': 'Snow', 'frost': 'Frost',
              'fog': 'Fog', 'brightness': 'Brit.', 'contrast': 'Contr.', 'elastic_transform': 'Elastic',
              'pixelate': 'Pixel.', 'jpeg_compression': 'JPEG'}
CORR_GROUPS = {'Noise': ['gaussian_noise', 'shot_noise', 'impulse_noise'],          # Hendrycks & Dietterich (2019) 분류
               'Blur': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'],
               'Weather': ['snow', 'frost', 'fog', 'brightness'],
               'Digital': ['contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']}


# 2026-11-19 이름 확정 (코드·예전 기록의 D4 = Few-shot DINO head). 2026-11-20 사용자 결정으로 ours를 바꿈:
#   16장 CLIP 미세조정(FT_ITERS) + 새 head(NEW_HEAD = CLS+patch 특징 · test 특징 중심화 · test 좌우반전) + CDIM.
#   이전 ours(zero-shot CLIP + 기존 head)는 ablation의 참고 행으로 남긴다.
#   2026-11-23 사용자 결정("그래 넣자")으로 다시 바꿈: + TTA 중 DINOv2 LayerNorm 적응(적응한 특징은 head에만, OURS_HEAD = NEW_HEAD+dlnh).
#   DINO 고정판(86.35)은 s['ours_fix']로 ablation·Table A 참고 행에 남긴다.
#   2026-11-24: ours의 head는 pick_ours.py가 고른 config.CHOSEN_HEAD(파일이 없으면 OURS_HEAD). 표·fig7이 이를 따른다.
OURS = 'CDIM + CLIP 미세조정 + Few-shot DINO head + DINO 적응 (ours)'
_FLAG_NAME = {'cp': 'patch 평균 특징', 'ctr': 'test 특징 중심화', 'qf': 'test 좌우반전 평균'}


def head_desc(src):
    """'dino+ctr+qf+dlnh' → 'CLS + test 특징 중심화 + test 좌우반전 평균' (DINO 적응 플래그는 뺌)."""
    fl = [f for f in src.split('+')[1:] if f in _FLAG_NAME]
    return 'CLS만 (기존 head)' if not fl else ' + '.join(['CLS'] + [_FLAG_NAME[f] for f in fl])
OLD_OURS = '이전 ours: CDIM + Few-shot DINO head (zero-shot CLIP, 기존 head)'


def _specs():
    """표에 쓰는 설정들. 모든 행이 같은 160장 subset 3개·같은 TTA 프로토콜 (RESULTS.md ★ 조건).
    조건 통일(2026-11-20): 16장 CLIP 미세조정 = FT_ITERS(782 iter), full = FULL_ITERS(10 epoch), 반복 3회."""
    from config import TIP_TUNED
    base = dict(KDEFAULT, aug='none', shots=16)
    s = dict(zs=dict(base, iters=0, head_mode='zeroshot'),
             ft=dict(base, iters=FT_ITERS, head_mode=CFG['HEAD_MODE_MAIN']),
             full=dict(base, shots=5000, iters=FULL_ITERS, head_mode=CFG['HEAD_MODE_MAIN']))

    def tip(src):
        a, b = TIP_TUNED[src]
        return dict(s['zs'], proto=a / CFG['TIP_ALPHA'], proto_src=src if b == CFG['TIP_BETA'] else f'{src}@{b:g}')
    s.update(tc=tip('tip_clip'), td=tip('tip_dino'), head=dict(s['zs'], proto=1.0, proto_src='dino'))
    s.update(fthead=dict(s['ft'], proto=1.0, proto_src='dino'),               # 16장 미세조정 + 기존 head
             ours_fix=dict(s['ft'], proto=1.0, proto_src=NEW_HEAD),           # 새 head, DINO 고정 (2026-11-20 ours, 86.35)
             ours_zs_fix=dict(s['zs'], proto=1.0, proto_src=NEW_HEAD),        # 위에서 CLIP 학습 없음
             ours_full_fix=dict(s['full'], proto=1.0, proto_src=CHOSEN_HEAD.replace('+dlnh', '')),   # 5만 장 + ours head, DINO 고정
             ours=dict(s['ft'], proto=1.0, proto_src=CHOSEN_HEAD),            # ours (2026-11-23 + DINO 적응, head는 11-24 선택)
             ours_zs=dict(s['zs'], proto=1.0, proto_src=CHOSEN_HEAD),         # ours에서 CLIP 학습 없음
             ours_full=dict(s['full'], proto=1.0, proto_src=CHOSEN_HEAD),     # 5만 장 10 epoch + ours 방식 (상한)
             ours_nodino=dict(s['ft'], proto=1.0, proto_src=CHOSEN_HEAD.replace('+dlnh', '')),   # ours − DINO 적응
             ours_dln=dict(s['ft'], proto=1.0, proto_src=CHOSEN_HEAD.replace('dlnh', 'dln')),   # 적응한 DINO를 CDIM 분기에도
             # baseline "CDIM + DINO LN 적응" (head 없음: λ=1e-6으로 head 항을 끔, 적응한 DINO는 CDIM 분기에)
             dln_nohead=dict(s['ft'], proto=1e-6, proto_src=NEW_HEAD + '+dln'),
             dln_nohead_zs=dict(s['zs'], proto=1e-6, proto_src=NEW_HEAD + '+dln'))
    return s


def _main_rows():
    """Main table: 기존 방법 + ours 하나. (그룹, 행 이름, 설정, 방법, 기대 seed 수)"""
    s = _specs()
    zs, L0, L16, L16T = s['zs'], '라벨 없음 (zero-shot CLIP)', '클래스당 16장, TTA 없음', '클래스당 16장 + TTA'
    return [
        (L0, 'CLIP zero-shot (TTA 없음)', zs, 'source', 1),
        (L0, 'TENT', zs, 'tent', 1),
        (L0, 'ETA', zs, 'eta', 1),
        (L0, 'SAR', zs, 'sar', 1),
        (L0, 'TDA', zs, 'tda', 1),
        (L0, 'BATCLIP', zs, 'batclip', 1),
        (L0, 'CDIM', zs, 'cdim', 1),
        (L0, 'CDIM + DINO LN 적응', s['dln_nohead_zs'], 'cdim', 1),
        (L16, 'CLIP 미세조정', s['ft'], 'source', 3),
        (L16, 'Tip-Adapter', s['tc'], 'source', 3),
        (L16, 'DINO 캐시 (Tip 방식)', s['td'], 'source', 3),
        (L16T, 'CDIM + CLIP 미세조정', s['ft'], 'cdim', 3),
        (L16T, 'CDIM + CLIP 미세조정 + DINO LN 적응', s['dln_nohead'], 'cdim', 3),
        (L16T, 'CDIM + Tip-Adapter', s['tc'], 'cdim', 3),
        (L16T, 'CDIM + DINO 캐시 (Tip 방식)', s['td'], 'cdim', 3),
        (L16T, OURS, s['ours'], 'cdim', 3),
        ('참고: full 데이터 5만 장 (10 epoch, 학습 seed 3개)', 'CDIM + CLIP 미세조정 (5만 장)', s['full'], 'cdim', 3),
    ]


def _ablation_rows():
    """Ablation: Few-shot DINO head가 들어간 변형 (corruption 15종)."""
    s = _specs()
    zs, h, o = s['zs'], s['head'], s['ours']
    ft_h = lambda src: dict(s['ft'], proto=1.0, proto_src=src + '+dlnh')     # head 구성 바꾸기: DINO 적응은 유지
    # ours와 다른 head 구성 (㉙ R_comp, 16장 × CDIM으로 잰 것들). ours가 어느 쪽이든 나머지를 모두 보여 준다
    variants = [v for v in ('dino+cp+ctr+qf', 'dino+cp+ctr', 'dino+cp+qf', 'dino+ctr+qf', 'dino')
                if v + '+dlnh' != CHOSEN_HEAD]
    nodino = ('11-20 ours, DINO 고정' if CHOSEN_HEAD == NEW_HEAD + '+dlnh' else
              '② + 기존 head, DINO 고정' if CHOSEN_HEAD == 'dino+dlnh' else 'DINO 고정')
    return [
        ('구성 요소', OURS, o, 'cdim', 3),
        ('구성 요소', f'− DINO 적응 (= {nodino})', s['ours_nodino'], 'cdim', 3),
        ('구성 요소', '− head (= CDIM + CLIP 미세조정 + DINO LN 적응)', s['dln_nohead'], 'cdim', 3),
        ('구성 요소', '− head − DINO 적응 (= CDIM + CLIP 미세조정)', s['ft'], 'cdim', 3),
        ('구성 요소', 'TTA 없음 (= CLIP 미세조정 + head)', o, 'source', 3),
        ('구성 요소', 'CLIP 학습 없음 (= zero-shot CLIP + head + DINO 적응)', s['ours_zs'], 'cdim', 3),
        ('DINO 적응 방식', '적응한 DINO를 CDIM 분기에도 (dln)', s['ours_dln'], 'cdim', 3),
    ] + [('head 구성 바꾸기 (DINO 적응 유지)', head_desc(v), ft_h(v), 'cdim', 3) for v in variants] + [
        ('TTA 방법 교체 (head 고정)', 'TENT + head', o, 'tent', 3),
        ('TTA 방법 교체 (head 고정)', 'ETA + head', o, 'eta', 3),
        ('TTA 방법 교체 (head 고정)', 'SAR + head †', o, 'sar', 3),
        ('TTA 방법 교체 (head 고정)', 'TDA + head', o, 'tda', 3),
        ('TTA 방법 교체 (head 고정)', 'BATCLIP + head', o, 'batclip', 3),
        ('이전 ours와 확장 (zero-shot CLIP + 기존 head 기준)', OLD_OURS, h, 'cdim', 3),
        ('이전 ours와 확장 (zero-shot CLIP + 기존 head 기준)', 'A: 프로토타입을 TTA 중 갱신',
         dict(zs, proto=1.0, proto_src='dino_tta'), 'cdim', 3),
        ('이전 ours와 확장 (zero-shot CLIP + 기존 head 기준)', 'B: CDIM DINO 분기를 라벨 프로토타입으로 교체 (head 없음)',
         dict(zs, cdim_m='lab'), 'cdim', 3),
        ('이전 ours와 확장 (zero-shot CLIP + 기존 head 기준)', 'B: 교체 0.5 + 원래 0.5 혼합 (head 없음)',
         dict(zs, cdim_m='lab_mix'), 'cdim', 3),
        ('이전 ours와 확장 (zero-shot CLIP + 기존 head 기준)', 'B 교체 + 기존 head', dict(h, cdim_m='lab'), 'cdim', 3),
        ('참고: full 데이터 5만 장 (10 epoch, 학습 seed 3개)', 'ours 방식 (5만 장, 상한)', s['ours_full'], 'cdim', 3),
        ('참고: full 데이터 5만 장 (10 epoch, 학습 seed 3개)', 'ours 방식, DINO 고정 (5만 장)', s['ours_full_fix'], 'cdim', 3),
    ]


def _appendix_rows():
    """부록: 160장 라벨의 사용 방식(행) × TTA 방법(열), 15종 평균. 모두 같은 조건(782 iter, 10 epoch, 3회)."""
    s = _specs()
    return [('사용 안 함 (zero-shot CLIP)', s['zs'], 1),
            (f'CLIP 미세조정 ({FT_ITERS} iter)', s['ft'], 3),
            ('Tip-Adapter (CLIP 캐시)', s['tc'], 3),
            ('DINO 캐시 (Tip 방식)', s['td'], 3),
            ('기존 Few-shot DINO head (λ=1)', s['head'], 3),
            ('CLIP 미세조정 + 기존 head', s['fthead'], 3),
            ('CLIP 미세조정 + 새 head (DINO 고정)', s['ours_fix'], 3),
            (f'**CLIP 미세조정 + head({head_desc(CHOSEN_HEAD)}) + DINO 적응 (ours)**', s['ours'], 3),
            ('참고 (조건 다름): full 데이터 CLIP 미세조정 (10 epoch)', s['full'], 3)]


def appendix_table(dk):
    rows = _appendix_rows()
    specs = [(name, m, f, m, n) for name, f, n in rows for m in METHOD_ORDER]
    t = _collect(dk, specs)
    lines = ['| 160장 라벨의 사용 | ' + ' | '.join(METHOD_ORDER) + ' |', '|---|' + '---|' * len(METHOD_ORDER)]
    for name, _, _ in rows:
        cells = []
        for m in METHOD_ORDER:
            r = t[(t['그룹'] == name) & (t['Method'] == m)].iloc[0]
            cells.append('⏳' if not r['ok'] else
                         f"{r['Mean']:.2f}" + ('' if np.isnan(r['std']) else f"±{r['std']:.2f}"))
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    md = '\n'.join(lines)
    with open(f'{OUT}/appendix_table.md', 'w') as fh:
        fh.write(md + '\n')
    print('\n### 부록 표 — 라벨 사용 방식 × TTA 방법 (15종 평균)\n' + md)


def _collect(dk, specs):
    C = CFG['CORR_ALL']
    rows = []
    for grp, name, filt, m, n_exp in specs:
        sel = dk['method'] == m
        for c, v in filt.items():
            sel &= dk[c] == v
        d = dk[sel & dk.corruption.isin(C)]
        ps = d.groupby(['seed', 'corruption']).acc.mean().unstack('corruption').dropna()   # 15종이 다 끝난 seed만
        r = dict(그룹=grp, Method=name, seeds=len(ps), ok=len(ps) >= n_exp and set(C) <= set(ps.columns))
        if r['ok']:
            for c in C:
                r[c] = ps[c].mean()
            for g, cs in CORR_GROUPS.items():
                r[g] = ps[cs].mean(1).mean()
            pm = ps[C].mean(1)
            r['Mean'], r['std'] = pm.mean(), (pm.std() if len(pm) > 1 else np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def _md(t, cols, heads, path, bold_best=True, delta_ref=None):
    """굵게: 참고 행을 뺀 열 최고값(bold_best) 또는 ours 행. delta_ref가 있으면 Mean − ours 열 추가."""
    main = t[t.ok & ~t['그룹'].str.startswith('참고')]
    best = {c: main[c].max() for c in cols} if bold_best else {}
    extra = [] if delta_ref is None else ['Δ vs ours']
    lines = ['| 그룹 | Method | ' + ' | '.join(heads + extra) + ' |', '|' + '---|' * (len(heads) + len(extra) + 2)]
    for _, r in t.iterrows():
        cells = []
        for c in cols:
            if not r['ok']:
                cells.append('⏳'); continue
            v = f'{r[c]:.2f}'
            if c == 'Mean' and not np.isnan(r['std']):
                v += f'±{r["std"]:.2f}'
            if (bold_best and not r['그룹'].startswith('참고') and round(r[c], 2) == round(best[c], 2)) or \
               (not bold_best and r['Method'] == OURS):
                v = f'**{v}**'
            cells.append(v)
        if delta_ref is not None:
            cells.append('' if (not r['ok'] or r['Method'] == OURS) else f"{r['Mean'] - delta_ref:+.2f}")
        name = f"**{r['Method']}**" if r['Method'] == OURS else r['Method']
        lines.append(f"| {r['그룹']} | {name} | " + ' | '.join(cells) + ' |')
    s = '\n'.join(lines)
    with open(path, 'w') as f:
        f.write(s + '\n')
    print(s)


def main_table(dk):
    if dk is None or dk.empty: return
    dk = _kfill(dk)
    C, G = CFG['CORR_ALL'], list(CORR_GROUPS)
    t = _collect(dk, _main_rows())
    section('Main table — 논문 형식 (행 = 방법, 열 = corruption, 15종, severity 5, 동일 조건은 RESULTS.md ★)')
    t.drop(columns=['ok']).round(2).to_csv(f'{OUT}/main_table.csv', index=False)
    print('\n### Table 1. corruption별 (굵게 = 참고 행을 뺀 열 최고)')
    _md(t, C + ['Mean'], [CORR_SHORT[c] for c in C] + ['Mean'], f'{OUT}/main_table.md')
    print('\n### Table 2. 손상 유형별 평균 (Noise 3 · Blur 4 · Weather 4 · Digital 4)')
    _md(t, G + ['Mean'], G + ['Mean'], f'{OUT}/main_table_groups.md')

    a = _collect(dk, _ablation_rows())
    a.drop(columns=['ok']).round(2).to_csv(f'{OUT}/ablation_table.csv', index=False)
    ours = a[(a.Method == OURS) & a.ok]
    print('\n### Table 3. Ablation — Few-shot DINO head 변형 (15종, Δ = Mean − ours, † SAR 붕괴 감지가 매 배치 초기화)')
    _md(a, G + ['Mean'], G + ['Mean'], f'{OUT}/ablation_table.md', bold_best=False,
        delta_ref=None if ours.empty else float(ours['Mean'].iloc[0]))
    appendix_table(dk)
    table_a(dk)


# ---------------- Table A: Full 5만 장 → 16장 → 16장 + ours (교수님 면담 서사, RESULTS.md ★★) ----------------
FULL_EPOCHS = {782: 2, 1955: 5, 3910: 10}     # 5만 장, batch 128 → 391 iter/epoch
FULL_REF_ITERS = FULL_ITERS                     # Table A의 ① (P2 수렴 확인 결과로 10 epoch)


def table_a(dk):
    s = _specs()
    # 수렴 비교(full{e}ep)는 세 학습량을 같은 seed 0으로 비교. 표의 ①('full')은 학습 seed 3개 평균
    conds = [(f'full{e}ep', dict(s['full'], iters=it, seed=0), 1) for it, e in FULL_EPOCHS.items()] + [
        ('full', s['full'], 3),
        ('16shot', s['ft'], 3), ('16shot200', dict(s['ft'], iters=ITERS_MAIN), 3),
        ('16shot+head', s['fthead'], 3),        # ② + 기존 head
        ('zs+head', s['head'], 3),              # 처음 ours
        ('ours_fix', s['ours_fix'], 3),         # ② + 새 head, DINO 고정 (2026-11-20 ours)
        ('nohead', s['dln_nohead'], 3),         # ② + DINO 적응만 (head 없음)
        ('ours', s['ours'], 3),                 # ③ = ② + 새 head + DINO 적응 (2026-11-23 새 ours)
        ('zs+newhead', s['ours_zs'], 3)]        # 새 ours에서 CLIP 학습 없음
    specs = [(c, m, f, m, n) for m in METHOD_ORDER for c, f, n in conds]
    t = _collect(dk, specs).rename(columns={'그룹': 'cond', 'Method': 'method'})
    if not t.ok.any(): return
    t[['cond', 'method', 'seeds', 'ok'] + list(CORR_GROUPS) + ['Mean', 'std']].round(2).to_csv(
        f'{OUT}/table_a_long.csv', index=False)
    M = t.set_index(['method', 'cond'])
    section('Table A — Full 5만 장 → 16장 → 16장 + Few-shot DINO head (15종 평균)')

    def g(m, c, k='Mean'):
        return M.loc[(m, c), k] if M.loc[(m, c), 'ok'] else np.nan

    clean = dk[(dk.shots == 5000) & (dk.head_mode == CFG['HEAD_MODE_MAIN']) & (dk.proto == 0)].groupby(
        'iters').src_clean_acc.first()
    print('full clean (TTA 없음): ' + ', '.join(f'{FULL_EPOCHS.get(int(i), i)} epoch {v:.2f}' for i, v in clean.items()))
    print('\n(a) full 학습량별 15종 평균')
    print('| TTA | ' + ' | '.join(f'{e} epoch' for e in FULL_EPOCHS.values()) + ' |')
    print('|---|' + '---|' * len(FULL_EPOCHS))
    for m in METHOD_ORDER:
        print(f'| {m} | ' + ' | '.join(f'{g(m, f"full{e}ep"):.2f}' for e in FULL_EPOCHS.values()) + ' |')

    ref = 'full'

    def sd(m, c):
        v, e = g(m, c), g(m, c, 'std')
        return '⏳' if np.isnan(v) else f'{v:.2f}' + ('' if np.isnan(e) else f'±{e:.2f}')

    dif = lambda x, y: '⏳' if np.isnan(x) or np.isnan(y) else f'{x - y:+.2f}'
    lines = [f'| TTA 방법 | ① Full 5만 장 ({FULL_EPOCHS[FULL_REF_ITERS]} epoch) | ② 16장 ({FT_ITERS} iter) | 감소 ②−① | '
             '③ ② + head + DINO 적응 (ours) | 회복 ③−② | ③−① | (참고) ② + 새 head, DINO 고정 | (참고)−① |',
             '|---|---|---|---|---|---|---|---|---|']
    rows = []
    for m in METHOD_ORDER:
        f, a, b, z = g(m, ref), g(m, '16shot'), g(m, 'ours'), g(m, 'ours_fix')
        rows.append(dict(method=m, full=f, shot16=a, ours=b, shot16_fixhead=z, drop=a - f, recover=b - a, vs_full=b - f,
                         fixhead_vs_full=z - f))
        lines.append(f'| {m} | {sd(m, ref)} | {sd(m, "16shot")} | {dif(a, f)} | {sd(m, "ours")} | {dif(b, a)} | '
                     f'{dif(b, f)} | {sd(m, "ours_fix")} | {dif(z, f)} |')
    pd.DataFrame(rows).round(2).to_csv(f'{OUT}/table_a_all_methods.csv', index=False)
    md = '\n'.join(lines)
    with open(f'{OUT}/table_a_all_methods.md', 'w') as fh:
        fh.write(md + '\n')
    print(f'\n(b) 부록: 7개 TTA 방법 (① = {FULL_EPOCHS[FULL_REF_ITERS]} epoch)\n' + md)

    # (c) Table A 본표: 회복 서사는 CDIM만 본다 (2026-11-19 사용자 결정). ③ = Main table의 ours (zero-shot CLIP + head)
    #     ② = 16장 782 iter (2026-11-20 사용자 결정: 감소폭이 가장 작은 쪽 = 가장 강한 16장 기준선). 200 iter는 참고 행
    G = list(CORR_GROUPS)
    #     2026-11-20 새 ours: ③ = ② + 새 head (같은 CLIP 모델에 head만 더함). 이전 ③(zero-shot CLIP + 기존 head)은 참고 행
    rows_c = [('① Full', '5만 장', f'{FULL_EPOCHS[FULL_REF_ITERS]} epoch', ref),
              ('② 16장', '160장', f'{FT_ITERS} iter', '16shot'),
              ('③ ② + Few-shot DINO head + DINO 적응 (ours)', '160장', f'{FT_ITERS} iter', 'ours'),
              ('(참고) ② + DINO 적응만 (head 없음)', '160장', f'{FT_ITERS} iter', 'nohead'),
              ('(참고) ② + 새 head, DINO 고정 (11-20 ours)', '160장', f'{FT_ITERS} iter', 'ours_fix'),
              ('(참고) CLIP 학습 없이 ours (head + DINO 적응)', '160장', '없음', 'zs+newhead'),
              ('(참고) ② + 기존 head', '160장', f'{FT_ITERS} iter', '16shot+head'),
              ('(참고) 처음 ours: CLIP 학습 없이 기존 head', '160장', '없음', 'zs+head'),
              ('(참고) 라벨 없음', '0장', '없음', None),
              ('(참고) 16장, 짧게 학습 (설계 문서 고정값)', '160장', f'{ITERS_MAIN} iter', '16shot200')]
    zs = _collect(dk, [('', '', s['zs'], 'cdim', 1)]).iloc[0]
    out = ['| 조건 | 라벨 | CLIP 학습 | ' + ' | '.join(G) + ' | Mean | vs ① | vs ② |', '|' + '---|' * (len(G) + 6)]
    f, a = g('cdim', ref), g('cdim', '16shot')
    for name, lab, tr, c in rows_c:
        r = zs if c is None else M.loc[('cdim', c)]
        if not r['ok']:
            out.append(f'| {name} | {lab} | {tr} | ' + ' | '.join(['⏳'] * (len(G) + 3)) + ' |'); continue
        mean = f"{r['Mean']:.2f}" + (f"±{r['std']:.2f}" if not np.isnan(r['std']) else '')
        out.append(f'| {name} | {lab} | {tr} | ' + ' | '.join(f'{r[k]:.2f}' for k in G) +
                   f" | {mean} | {r['Mean'] - f:+.2f} | {r['Mean'] - a:+.2f} |")
    md = '\n'.join(out)
    with open(f'{OUT}/table_a.md', 'w') as fh:
        fh.write(md + '\n')
    print(f"\n(c) Table A — CDIM (① seed {int(M.loc[('cdim', ref), 'seeds'])}개)\n" + md)


# ---------------- P4: 클래스당 라벨 수 곡선 (CDIM, 15종 평균) ----------------
CURVE_SHOTS = (1, 2, 4, 8, 16, 79, 157, 625, 5000)    # config.CURVE_K와 같음. 625장 이상은 1회, 그 아래는 subset 3개


def label_curve(dk):
    """baseline = CLIP 미세조정(FT_ITERS iter; 5만 장만 FULL_ITERS) + CDIM, ours = 같은 CLIP 모델 + 같은 k장으로 만든 새 head + CDIM
    (2026-11-20 새 ours. 5만 장 점은 학습 seed 3개)."""
    s = _specs()
    n = lambda k: 1 if k >= 625 else 3
    specs = ([('base', k, dict(s['ft'], shots=k), 'cdim', n(k)) for k in CURVE_SHOTS if k < 5000]
             + [('base', 5000, s['full'], 'cdim', 3)]
             + [('ours', k, dict(s['ours'], shots=k), 'cdim', n(k)) for k in CURVE_SHOTS if k < 5000]
             + [('ours', 5000, s['ours_full'], 'cdim', 3)])
    t = _collect(dk, specs).rename(columns={'그룹': 'series', 'Method': 'shots'})
    zs = _collect(dk, [('', '', s['zs'], 'cdim', 1)]).iloc[0]
    if not t.ok.any() or not zs['ok']: return
    t['shots'] = t['shots'].astype(int)
    t[['series', 'shots', 'seeds', 'ok'] + list(CORR_GROUPS) + ['Mean', 'std']].round(2).to_csv(
        f'{OUT}/label_curve.csv', index=False)
    b = t[(t.series == 'base') & t.ok].set_index('shots')
    o = t[(t.series == 'ours') & t.ok].set_index('shots')
    cell = lambda d, k: '⏳' if k not in d.index else (
        f"{d.loc[k, 'Mean']:.2f}" + ('' if np.isnan(d.loc[k, 'std']) else f"±{d.loc[k, 'std']:.2f}") + f" ({d.loc[k, 'seeds']}회)")
    lines = ['| 클래스당 라벨 | baseline: CLIP 미세조정 + CDIM | ours | ours − baseline |', '|---|---|---|---|']
    for k in sorted(CURVE_SHOTS, reverse=True):
        d = (f"{o.loc[k, 'Mean'] - b.loc[k, 'Mean']:+.2f}" if k in b.index and k in o.index else '⏳')
        lines.append(f"| {k}{' (전부, 10 epoch)' if k == 5000 else ''} | {cell(b, k)} | {cell(o, k)} | {d} |")
    lines.append(f"| 0 (zero-shot) | {zs['Mean']:.2f} | | |")
    md = '\n'.join(lines)
    with open(f'{OUT}/label_curve.md', 'w') as fh:
        fh.write(md + '\n')
    section('P4 — 클래스당 라벨 수 곡선 (CDIM, 15종 평균)')
    print(md)

    # 그림: 정적 PNG(보고서용). 2개 계열 = categorical 1·2번 색 + 마커 모양, 범례 + 직접 라벨, 기준선 2개는 회색
    BLUE, ORANGE, INK, INK2, MUTED, GRID, AXIS = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    full = b.loc[5000, 'Mean'] if 5000 in b.index else np.nan
    ax.axhline(zs['Mean'], color=MUTED, lw=1.2, ls=(0, (5, 3)), zorder=1)
    ax.text(3600, zs['Mean'] - 0.12, f"No labels: zero-shot CLIP + CDIM ({zs['Mean']:.2f})", ha='right', va='top',
            fontsize=8.5, color=INK2)
    if not np.isnan(full):
        ax.axhline(full, color=MUTED, lw=1.2, ls=(0, (1.5, 2.5)), zorder=1)
        ax.text(70, full + 0.12, f'All 50k labels: fine-tuned\nCLIP 10 epochs + CDIM ({full:.2f})', ha='left',
                va='bottom', fontsize=8, color=INK2, linespacing=1.25)   # 79~625 구간: ours 곡선 아래·기준선 위의 빈 곳
    bf = b[b.index < 5000]
    ax.errorbar(bf.index, bf.Mean, yerr=bf['std'].fillna(0), color=BLUE, marker='o', ms=6.5, lw=2, capsize=2.5,
                elinewidth=1.2, mec='white', mew=1.2, zorder=3,
                label=f'Baseline: CLIP fine-tuned on the labels\n({FT_ITERS} iter; 5000 = 10 epochs) + CDIM')
    if 5000 in b.index and len(bf):   # 5만 장 점은 학습량이 달라(10 epoch) 점선으로 잇는다
        k0 = bf.index.max()
        ax.plot([k0, 5000], [bf.loc[k0, 'Mean'], full], color=BLUE, lw=1.5, ls=(0, (4, 2.5)), zorder=2)
        ax.plot([5000], [full], color=BLUE, marker='o', ms=6.5, mec='white', mew=1.2, zorder=3)
    of = o[o.index < 5000]
    ax.errorbar(of.index, of.Mean, yerr=of['std'].fillna(0), color=ORANGE, marker='s', ms=6.5, lw=2, capsize=2.5,
                elinewidth=1.2, mec='white', mew=1.2, zorder=4,
                label='Ours: baseline + Few-shot DINO head,\nDINO also adapted at test time (same labels)')
    if 5000 in o.index and len(of):   # ours도 5만 장 점은 CLIP을 10 epoch 학습한 모델이라 점선으로 잇는다
        k0 = of.index.max()
        ax.plot([k0, 5000], [of.loc[k0, 'Mean'], o.loc[5000, 'Mean']], color=ORANGE, lw=1.5, ls=(0, (4, 2.5)), zorder=2)
        ax.plot([5000], [o.loc[5000, 'Mean']], color=ORANGE, marker='s', ms=6.5, mec='white', mew=1.2, zorder=4)
    if 625 in b.index and 625 in o.index:   # 직접 라벨
        ax.text(625, o.loc[625, 'Mean'] + 0.25, 'Ours', ha='center', va='bottom', fontsize=9.5, color=INK, weight='bold')
        ax.text(625, b.loc[625, 'Mean'] - 0.3, 'Baseline', ha='center', va='top', fontsize=9.5, color=INK, weight='bold')
    y_note = None                           # 핵심 주장 설명의 y (그릴 때만)
    if 16 in b.index and 16 in o.index:     # 16장: 회복 폭
        y0, y1 = b.loc[16, 'Mean'], o.loc[16, 'Mean']
        ax.annotate('', xy=(16, y1 - 0.4), xytext=(16, y0 + 0.22),
                    arrowprops=dict(arrowstyle='<->', color=INK2, lw=1.1, shrinkA=0, shrinkB=0), zorder=5)
        # 회복 폭 라벨: 5만 장 점선과 겹치지 않게, 점선이 화살표 구간 안에 있으면 그 아래 반쪽 가운데에
        ym = (y0 + full) / 2 if (not np.isnan(full) and y0 < full < y1) else (y0 + y1) / 2
        ax.text(17.5, ym, f'{y1 - y0:+.2f}%p\n(16 per class)', ha='left', va='center', fontsize=8.5, color=INK2)
        if not np.isnan(full) and y1 > full:   # 핵심 주장 강조: 160장 ours > 5만 장 baseline (링 + 왼쪽 위 빈 곳의 설명)
            ax.plot([16], [y1], marker='o', ms=15, mfc='none', mec=ORANGE, mew=1.4, zorder=5)
            lo16 = of[of.index <= 16]          # 설명은 16장 이하 ours 곡선(error bar 포함)보다 위에
            y_note = max((lo16.Mean + lo16['std'].fillna(0)).max(), y1) + 0.6
            ax.annotate(f'160 labels beat all 50k labels\n{y1:.2f} vs {full:.2f} (+{y1 - full:.2f}%p)',
                        xy=(16, y1), xytext=(1.0, y_note), ha='left', va='bottom', fontsize=9, color=INK,
                        linespacing=1.3, zorder=6,
                        arrowprops=dict(arrowstyle='-', color=INK2, lw=0.9, shrinkA=3, shrinkB=8,
                                        relpos=(0.9, 0)))
    ax.set_xscale('log')
    ax.set_xticks(list(CURVE_SHOTS))
    ax.set_xticklabels([str(k) if k < 5000 else '5000\n(all)' for k in CURVE_SHOTS])
    ax.minorticks_off()
    ax.set_xlim(0.8, 6500)
    lo = np.floor(min(zs['Mean'], (t.Mean - t['std'].fillna(0)).min()) - 0.5)   # error bar가 잘리지 않게
    hi = np.ceil(t.Mean.max() + 0.8)
    if y_note is not None:                  # 위 설명(2줄)이 잘리지 않게
        hi = max(hi, np.ceil(y_note + 1.6))
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Labeled images per class (log scale)', fontsize=10, color=INK2)
    ax.set_ylabel('Accuracy (%)', fontsize=10, color=INK2)
    ax.set_title('CIFAR-10-C (severity 5), mean of 15 corruptions, TTA = CDIM', fontsize=10.5, color=INK, loc='left')
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=8.5, frameon=False,
              handlelength=2.2, columnspacing=2.0)   # 그림 아래: 곡선·기준선 라벨과 겹치지 않게
    os.makedirs(f'{OUT}/figures', exist_ok=True)
    fig.savefig(f'{OUT}/figures/fig7_label_curve.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'그림: {OUT}/figures/fig7_label_curve.png')


# ---------------- 설계 문서용 표 ----------------
def _cell(d):
    return f"{d['mean']:.2f}±{d['std']:.2f}" if d['n_seeds'] > 1 else f"{d['mean']:.2f}"


def doc_tables(da, dk):
    """설계 문서의 Motivation 수치, Preliminary 표, RQ2 표 (text_init · corruption 4종 평균 · 현재 τ)."""
    if da is None: return
    a = da[(da.phase == 'A') & (da.head_mode == CFG['HEAD_MODE_MAIN']) & da.corruption.isin(KSHOT_CORRS)]
    if a.empty: return
    agg = summarize_A(a)
    n_corr = len(KSHOT_CORRS)

    def get(budget, ratio, m):   # corruption 4종이 모두 끝난 경우만
        r = agg[(agg.budget == budget) & np.isclose(agg.ratio, ratio) & (agg.method == m) & (agg.n_corr == n_corr)]
        return None if r.empty else r.iloc[0]

    section(f'설계 문서용 표 ({CFG["HEAD_MODE_MAIN"]} · corruption {n_corr}종 평균 · CDIM τ⁻¹={TAU:g})')
    full = {m: get('same_epochs', 1.0, m) for m in ('source', 'batclip', 'cdim')}
    if full['source'] is not None:
        c, s = full['source']['src_clean'], full['source']['mean']
        print(f'[Motivation] full source: clean {c:.2f}% → CIFAR-10-C {s:.2f}% ({s - c:+.2f}%p)')

    rows = []
    for ratio, budget, label in [(1.0, 'same_epochs', '2 epochs'), (1 / 8, 'same_epochs', 'same epochs'),
                                 (1 / 8, 'same_iters', 'same iters'), (1 / 64, 'same_epochs', 'same epochs'),
                                 (1 / 64, 'same_iters', 'same iters')]:
        cells = {m: get(budget, ratio, m) for m in full}
        if all(v is None for v in cells.values()): continue
        any_r = next(v for v in cells.values() if v is not None)
        row = dict(ratio=rstr(ratio), budget=label, iters=int(any_r['n_iters']), clean=round(any_r['src_clean'], 2))
        for m, r in cells.items():
            row[m] = '' if r is None else _cell(r)
        for m, r in cells.items():   # full 대비 하락 (Summary·Motivation의 "x%p 떨어졌다")
            row[f'Δ{m}'] = '' if (ratio == 1.0 or r is None or full[m] is None) else round(r['mean'] - full[m]['mean'], 2)
        rows.append(row)
    if rows:
        t = pd.DataFrame(rows); print('\n### Preliminary — source 데이터 양'); print(t.to_string(index=False))
        t.to_csv(f'{OUT}/doc_preliminary.csv', index=False)

    rows = []
    for ratio, budget in [(1.0, 'same_epochs')] + [(r, 'same_iters') for r in CFG['SAME_ITERS_RATIOS']]:
        b, c = get(budget, ratio, 'batclip'), get(budget, ratio, 'cdim')
        if b is None and c is None: continue
        any_r = b if b is not None else c
        rows.append(dict(data=rstr(ratio), per_class=int(round(any_r['n_train'] / 10)), iters=int(any_r['n_iters']),
                         batclip='' if b is None else _cell(b), cdim='' if c is None else _cell(c),
                         cdim_minus_batclip='' if (b is None or c is None) else round(c['mean'] - b['mean'], 2)))
    if dk is not None and not dk.empty:   # 16-shot base, 비율 행들과 같은 학습량(ITERS_RQ2 = full 2 epoch)만
        k = _kfill(dk)
        k = k[_is_plain(k) & (k.iters == ITERS_RQ2) & k.corruption.isin(KSHOT_CORRS)]
        for it, g in k.groupby('iters'):
            cells = {}
            for m in ('batclip', 'cdim'):
                ps = g[g.method == m].groupby('seed').agg(acc=('acc', 'mean'), n=('corruption', 'nunique'))
                ps = ps[ps.n == n_corr]
                if len(ps):
                    cells[m] = {'mean': ps.acc.mean(), 'std': ps.acc.std() if len(ps) > 1 else 0.0, 'n_seeds': len(ps)}
            if not cells: continue
            b, c = cells.get('batclip'), cells.get('cdim')
            rows.append(dict(data='16-shot', per_class=16, iters=int(it),
                             batclip='' if b is None else _cell(b), cdim='' if c is None else _cell(c),
                             cdim_minus_batclip='' if (b is None or c is None) else round(c['mean'] - b['mean'], 2)))
    if rows:
        t = pd.DataFrame(rows); print('\n### RQ2 — 데이터 양에 따른 BATCLIP vs CDIM'); print(t.to_string(index=False))
        t.to_csv(f'{OUT}/doc_rq2.csv', index=False)


def main():
    os.makedirs(OUT, exist_ok=True)
    raw, dk = load('results.csv', filter_tau=False), load('results_kshot.csv')
    if raw is None and dk is None:
        print(f'결과가 없습니다: {EXP}'); return
    da = None if raw is None else raw[raw.cdim_clip_tau_inv.astype(float) == TAU]
    if raw is not None:
        sanity(raw); phase_A(da)
    kshot(dk, da)
    final15(dk)
    main_table(dk)
    label_curve(dk)
    doc_tables(da, dk)
    print(f'\n표·그래프 저장 위치: {OUT}')


if __name__ == '__main__':
    main()
