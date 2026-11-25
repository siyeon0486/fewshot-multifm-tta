"""Online TTA: Source / TENT / ETA / SAR / TDA / BATCLIP / CDIM.
모든 방법은 배치마다 forward 1번 → 그 forward로 예측 → loss로 1 step 업데이트 (TENT 프로토콜, SAR만 SAM이라 forward 2번).
업데이트 대상·optimizer는 모든 방법이 같다 (CLIP LayerNorm, AdamW). corruption마다 source 상태로 리셋 (episodic).
ETA·SAR·TDA는 이 프로토콜에 맞춘 재구현이다 (논문 원 설정과 optimizer·업데이트 층이 다를 수 있음)."""
import os, json, time, math
import numpy as np
import torch
import torch.nn.functional as F

from config import CFG, CLASSES
from core.runtime import R, autocast, make_scaler, optimizer_step, clip_input, dino_input
from core.models import M, freeze_all, layernorm_params
from core.data import load_corruption

K = len(CLASSES)          # CIFAR-10: 10, CIFAR-100: 100 (config.DATASET)
E0 = 0.4 * math.log(K)   # ETA·SAR의 신뢰 샘플 기준: 엔트로피 < 0.4·ln C (Niu et al.)


def entropy_from_logits(z):
    return -(z.softmax(1) * z.log_softmax(1)).sum(1)


# ---------------- source 쪽 head (D4와 같은 범주: 라벨 있는 few-shot 이미지로 만든 분류 head) ----------------
def head_base(src):
    """'tip_dino@8' → 'tip_dino', 'dino+cp+ctr' → 'dino'. Tip 캐시는 '@β'로 β를 따로 줄 수 있다(없으면 TIP_BETA).
    proto_src 문자열이 결과 key라 설정이 구분된다."""
    return src.split('@')[0].split('+')[0]


def head_flags(src):
    """'dino+cp+ctr+qf+a8' → {'cp', 'ctr', 'qf', 'a8'}. Few-shot DINO head 개선 후보 (2026-11-20, diag_head2.py):
      cp : 특징 = CLS와 patch 평균을 각각 정규화해 이어 붙임 (1536차원, DINOv2 linear probe 방식). 없으면 CLS(768)
      ctr: 중심화. 프로토타입은 라벨 160장 특징 평균 μ_s를 빼고, test 특징은 지금까지 본 test 특징 평균 μ_t(online,
           현재 batch 포함, corruption마다 초기화)를 뺀 뒤 다시 정규화
      qf : test 특징 = 원본과 좌우반전 특징의 평균 (DINO forward 2배)
      a8 : 프로토타입 = 라벨 이미지의 원본·반전·crop+flip 6개, 8개 view 특징 평균
      dln : (2026-11-23) TTA 중에 DINOv2의 LayerNorm도 같은 손실·optimizer로 업데이트. 적응한 DINO 특징을 head와
            CDIM의 DINO 분기 모두에 쓴다. 배치마다 DINO forward(gradient)가 필요해 특징 캐시를 쓰지 않는다
      dlnh: dln과 같지만 적응한 DINO 특징은 head에만 쓰고, CDIM의 DINO 분기는 원래(고정) 특징 그대로"""
    return set(src.split('@')[0].split('+')[1:])


def tip_beta(src):
    return float(src.split('@')[1]) if '@' in src else CFG['TIP_BETA']


def head_logits(Fq, keys, vals, src, w):
    """gradient 없는 head logit. Fq: query 특징(정규화, DINO 또는 detach한 CLIP).
    dino·clip : 클래스 프로토타입(nearest class mean)  w·PROTO_TAU_INV·F Pᵀ
    tip_*     : Tip-Adapter 캐시(Zhang et al., ECCV 2022)  w·TIP_ALPHA·exp(−β·(1 − F Kᵀ))·V  (α = w·TIP_ALPHA, β = tip_beta(src))"""
    if head_base(src) in ('dino', 'clip', 'dino_tta'):
        return w * CFG['PROTO_TAU_INV'] * Fq @ keys.t()
    return w * CFG['TIP_ALPHA'] * torch.exp(-tip_beta(src) * (1 - Fq @ keys.t())) @ vals


# ---------------- ETA / SAR / TDA (같은 범주의 TTA baseline, 재구현) ----------------
def eta_loss(logits, m):
    """ETA = EATA(Niu et al., ICML 2022)에서 Fisher 정규화를 뺀 것.
    신뢰(엔트로피 < E0)이면서 비중복(지금까지 평균 예측 m과 |cos| < ETA_D_MARGIN)인 샘플만, 가중치 exp(E0 − H)로 엔트로피 최소화."""
    ent, p = entropy_from_logits(logits), logits.softmax(1)
    sel = ent < E0
    ent1, p1 = ent[sel], p[sel]
    if m is not None and len(p1):
        keep = F.cosine_similarity(m[None], p1.detach(), dim=1).abs() < CFG['ETA_D_MARGIN']
        ent1, p1 = ent1[keep], p1[keep]
    if len(p1):
        mp = p1.detach().mean(0)
        m = mp if m is None else 0.9 * m + 0.1 * mp
    loss = (ent1 * torch.exp(E0 - ent1.detach())).mean() if len(ent1) else None
    return loss, m, dict(kept=int(len(ent1)))


def sar_update(clf, params, opt, scaler, loss1, x, sel, extra_fn):
    """SAR (Niu et al., ICLR 2023): 신뢰 샘플로 SAM 2단계 업데이트.
    1) w에서 gradient → ε = ρ·g/‖g‖  2) w+ε에서 다시 forward, 여전히 신뢰인 샘플의 엔트로피로 gradient  3) w로 되돌린 뒤 step.
    반환: (fp16 overflow로 건너뜀, 둘째 loss 값)."""
    scale = scaler.get_scale() if scaler is not None else 1.0
    grads = torch.autograd.grad(loss1 * scale, params, allow_unused=True)
    grads = [torch.zeros_like(p) if g is None else g / scale for g, p in zip(grads, params)]
    gnorm = torch.norm(torch.stack([g.norm() for g in grads]))
    if not torch.isfinite(gnorm):
        if scaler is not None:
            scaler.update(scale / 2)
        return True, None
    eps = [CFG['SAR_RHO'] * g / (gnorm + 1e-12) for g in grads]
    with torch.no_grad():
        for p, e in zip(params, eps): p.add_(e)
    with torch.enable_grad():
        with autocast():
            C2 = clf.image_features(x)
        logits2 = clf.logit_scale * C2 @ clf.class_weights().t() + extra_fn(C2)
        ent2 = entropy_from_logits(logits2)[sel]
        ent2 = ent2[ent2.detach() < E0]
        loss2 = ent2.mean() if len(ent2) else None
    opt.zero_grad(set_to_none=True)
    if loss2 is not None:
        (loss2 if scaler is None else scaler.scale(loss2)).backward()
    with torch.no_grad():
        for p, e in zip(params, eps): p.sub_(e)
    skipped = False
    if loss2 is not None:
        if scaler is None:
            opt.step()
        else:
            before = scaler.get_scale(); scaler.step(opt); scaler.update(); skipped = scaler.get_scale() < before
    return skipped, (None if loss2 is None else loss2.item())


class TDACache:
    """TDA (Karmanov et al., CVPR 2024)의 positive cache만 재구현 (학습 없음, negative cache는 제외).
    클래스(의사 라벨)마다 엔트로피가 가장 낮은 test 특징 TDA_POS_CAP개를 저장하고,
    logit에 α·exp(−β(1 − C Kᵀ))·V를 더한다 (α=2, β=5: 논문 ImageNet 설정). 배치를 먼저 cache에 넣고 예측한다."""

    def __init__(self):
        self.cap, self.alpha, self.beta = CFG['TDA_POS_CAP'], CFG['TDA_POS_ALPHA'], CFG['TDA_POS_BETA']
        self.items = {c: [] for c in range(K)}      # 클래스 → [(엔트로피, 특징)]

    def update(self, C, pl, ent):
        for f, c, e in zip(C, pl.tolist(), ent.tolist()):
            it = self.items[c]
            if len(it) < self.cap:
                it.append((e, f))
            else:
                j = max(range(len(it)), key=lambda i: it[i][0])
                if e < it[j][0]:
                    it[j] = (e, f)

    def logits(self, C):
        labels = [c for c in range(K) for _ in self.items[c]]
        if not labels:
            return 0.0
        keys = torch.stack([f for c in range(K) for _, f in self.items[c]])
        vals = F.one_hot(torch.tensor(labels, device=C.device), K).float()
        return self.alpha * torch.exp(-self.beta * (1 - C @ keys.t())) @ vals


def batclip_loss(logits, C, W):
    """BATCLIP Eq.2-6. C:(B,d) 정규화 이미지 feature, W:(K,d) 정규화 텍스트/헤드."""
    l_ent = entropy_from_logits(logits).mean()
    yhat = logits.argmax(1).detach()
    onehot = F.one_hot(yhat, W.shape[0]).float()
    counts = onehot.sum(0)
    present = counts > 0
    protos = (onehot.t() @ C)[present] / counts[present, None]          # Eq.2
    l_pm = (protos * W[present]).sum(1).mean()                          # Eq.3
    if protos.shape[0] > 1:
        Pn = F.normalize(protos, dim=1)
        l_sp = (1 - Pn @ Pn.t()).sum()                                  # Eq.4 (대각은 0)
    else:
        l_sp = logits.new_zeros(())
    return l_ent - l_pm - l_sp, dict(l_ent=l_ent.item(), l_pm=l_pm.item(), l_sp=float(l_sp.detach()))


def cdim_ensemble_logits(C, D, W, clip_tau_inv, dino_tau_inv, detach_proj=False, M_fixed=None, m_mix=0.0,
                         P_lab=None, lab_mix=1.0):
    """CDIM Eq.1-3. M_fixed가 있으면 D2: M = m_mix·M_batch + (1 − m_mix)·M_fixed (m_mix=0이면 완전히 고정).
    P_lab(K×768, 정규화)이 있으면 확장 B: DINO 분기의 분류기 T_D(라벨 없이 CLIP에서 투영)를 라벨 프로토타입으로
    교체(lab_mix=1) 또는 섞음 T_D = normalize((1−lab_mix)·T_D + lab_mix·Pᵀ)."""
    B = C.shape[0]
    Mx = (C.detach() if detach_proj else C).t() @ D / B
    if M_fixed is not None:
        Mx = M_fixed if m_mix == 0 else m_mix * Mx + (1 - m_mix) * M_fixed
    T_D = F.normalize(Mx.t() @ W.t(), dim=0)
    if P_lab is not None:
        T_D = P_lab.t() if lab_mix == 1.0 else F.normalize((1 - lab_mix) * T_D + lab_mix * P_lab.t(), dim=0)
    return 0.5 * (clip_tau_inv * C @ W.t() + dino_tau_inv * D @ T_D)


def gt_im_loss(Z, rho):
    """CDIM Eq.4-6 (gating은 Algorithm 1 구현)."""
    B = Z.shape[0]
    p = Z.softmax(1)
    E = -torch.logsumexp(Z, 1)
    k = max(1, int(B * rho))
    thr = torch.topk(E.detach(), k).values[-1]
    G = (E.detach() < thr).float()
    ent = -(p * torch.log(p + 1e-8)).sum(1)
    cond = (ent * G).sum() / (G.sum() + 1e-8)
    pm = p.mean(0)
    neg_marg = (pm * torch.log(pm + 1e-8)).sum()
    return cond + neg_marg, p, dict(cond=cond.item(), marg=-neg_marg.item(), kept=int(G.sum().item()))


def ola_predict(Z, p, ema_prior, alpha):
    """CDIM Eq.8-9."""
    ema_prior = alpha * ema_prior + (1 - alpha) * p.detach().mean(0)
    return (Z.detach() - torch.log(ema_prior + 1e-8)).argmax(1), ema_prior


def tta_params(clf, method):
    if method in ('source', 'tda'):    # TDA는 학습 없음 (cache만)
        return [], False
    ps = list(clf.visual_ln())
    update_text = method in ('batclip', 'cdim') and clf.head is None
    if update_text:
        ps += clf.text_ln()
    return ps, update_text


def run_online(clf, method, base_snap, x_u8, y, order, dino_feats=None, M_fixed=None, m_mix=0.0,
               proto=None, proto_w=0.0, proto_src='dino', proto_vals=None, P_lab=None, lab_mix=1.0,
               head_q=None, head_ctr=False, dino_ln=None, head_feat='cls', head_qf=False, keep_pred=False):
    """proto가 있으면 source 쪽 head logit P_l = head_logits(...)을 모든 방법의 logit에 더한다 (gradient 없음).
      proto_src: dino(D4, DINO 클래스 프로토타입) | clip(ablation, CLIP 프로토타입) | tip_clip·tip_dino(Tip-Adapter 캐시, proto=키, proto_vals=one-hot)
      query 특징은 dino·tip_dino면 DINO 특징 D, clip·tip_clip이면 현재 CLIP 특징 C를 detach한 것.
    Source·TENT·ETA·SAR·TDA·BATCLIP은 CLIP logit(100·C Wᵀ)에 그대로, CDIM은 앙상블이 CLIP logit을 0.5배로 쓰므로 0.5·P_l을 더한다
    (CLIP : head 비율이 diag_dino.py와 같음).
    proto_src='dino_tta'(확장 A): D4 프로토타입을 TTA 중에 갱신한다. 매 배치 예측 뒤, 확신 있는(엔트로피 < E0) 샘플이 있는 클래스마다
      P_c ← normalize((1−η)·P_c + η·normalize(그 샘플들의 DINO 특징 평균)), η = PROTO_TTA_MOM. 모든 클래스가 같은 속도로 움직여
      손상 공통 성분이 클래스 사이에서 상쇄된다. (처음 버전은 특징을 개수만큼 누적해 많이 예측된 클래스로 쏠려 smoke test에서
      Gaussian 1,000장 CDIM 62.0 → 52.4로 무너졌다.) gradient 없음. corruption마다 source 프로토타입에서 다시 시작한다.
    P_lab(확장 B): CDIM의 DINO 분기 분류기를 라벨 프로토타입으로 교체·혼합 (cdim_ensemble_logits 참고).
    head_q: head에 쓸 test 특징(head_flags의 cp·qf). None이면 CDIM과 같은 DINO CLS 특징 D. CDIM의 DINO 분기는 항상 D.
    head_ctr: head query에서 지금까지 본 test 특징 평균을 뺀다(head_flags의 ctr). gradient 없음.
    dino_ln: None | 'all' | 'head' (head_flags의 dln·dlnh). DINOv2 LayerNorm도 CLIP과 같은 optimizer로 업데이트한다
      (lr = CFG DINO_TTA_LR, 없으면 TTA_LR). head query는 배치마다 DINO로 새로 계산한다(head_feat·head_qf, 반전 view는 gradient 없음).
      'all'이면 CDIM의 DINO 분기 D도 적응한 CLS 특징. 끝나면 DINO를 원래 상태로 되돌린다(episodic, 다른 캐시 계산 보호).
    keep_pred: True면 샘플별 최종 예측(x 순서, 길이 n)을 out['pred']로 돌려준다 (fig_model.py 예시용, 결과 CSV에는 안 씀)."""
    proto_tta = proto is not None and proto_src == 'dino_tta'
    if proto_tta:
        proto = proto.clone()
    clf.load_snapshot(base_snap)
    freeze_all(clf)
    params, update_text = tta_params(clf, method)
    for p in params: p.requires_grad_(True)
    dino_params = layernorm_params(M.dino) if (dino_ln and params) else []
    dino_snap = [p.detach().clone() for p in dino_params]
    for p in dino_params: p.requires_grad_(True)
    groups = [{'params': params}] + ([{'params': dino_params, 'lr': CFG.get('DINO_TTA_LR', CFG['TTA_LR'])}]
                                     if dino_params else [])
    opt = torch.optim.AdamW(groups, lr=CFG['TTA_LR'], weight_decay=CFG['TTA_WD']) if params else None
    scaler = make_scaler()
    ema_prior = torch.full((K,), 1.0 / K, device=R.device)
    head_on_dino = proto is not None and head_base(proto_src) in ('dino', 'tip_dino', 'dino_tta')
    head_on_clip = proto is not None and head_base(proto_src) in ('clip', 'tip_clip')
    clip_head = (lambda Cq: head_logits(Cq.detach(), proto, proto_vals, proto_src, proto_w)) if head_on_clip else None
    eta_m, sar_ema, n_reset = None, None, 0
    h_sum, h_n = 0.0, 0                      # head_ctr: 지금까지 본 test 특징 합·개수
    tda = TDACache() if method == 'tda' else None
    correct, skipped, n, B = 0, 0, len(y), CFG['TTA_BATCH']
    pred_all = np.full(n, -1, dtype=np.int64) if keep_pred else None
    diag = []
    if R.device == 'cuda': torch.cuda.synchronize()
    t0 = time.time()
    try:
        for s in range(0, n, B):
            idx = order[s:s + B]
            x = clip_input(x_u8[idx])
            yb = torch.from_numpy(y[idx])
            D = dino_feats[torch.from_numpy(idx)].to(R.device).float() if dino_feats is not None else None
            if head_on_dino and dino_ln:                 # dln·dlnh: 지금 DINO(적응 중)로 head query를 새로 계산
                Dh, cls_live = dino_live_feats(x_u8[idx], head_feat, head_qf, grad=bool(dino_params))
                if dino_ln == 'all':
                    D = cls_live
                if head_ctr:
                    h_sum, h_n = h_sum + Dh.detach().sum(0), h_n + len(Dh)
                    Dh = F.normalize(Dh - h_sum / h_n, dim=-1)
                P_l = head_logits(Dh, proto, proto_vals, proto_src, proto_w)

                def head_live(_Cq, xb=x_u8[idx], mu=(h_sum / h_n if head_ctr else None)):
                    """SAR 2단계: 섭동된 DINO로 head logit을 다시 계산 (gradient 있음, μ_t는 이번 배치까지 그대로)."""
                    Dq, _ = dino_live_feats(xb, head_feat, head_qf, grad=True)
                    if mu is not None:
                        Dq = F.normalize(Dq - mu, dim=-1)
                    return head_logits(Dq, proto, proto_vals, proto_src, proto_w)
            elif head_on_dino:
                Dh = D if head_q is None else head_q[torch.from_numpy(idx)].to(R.device).float()
                if head_ctr:
                    h_sum, h_n = h_sum + Dh.sum(0), h_n + len(Dh)
                    Dh = F.normalize(Dh - h_sum / h_n, dim=-1)
                P_l = head_logits(Dh, proto, proto_vals, proto_src, proto_w)
            else:
                P_l = 0.0
            loss = None
            with torch.set_grad_enabled(opt is not None):
                with autocast():
                    C = clf.image_features(x)
                if head_on_clip:
                    P_l = clip_head(C)
                W = clf.class_weights(text_grad=update_text)
                if method == 'cdim':
                    Z = cdim_ensemble_logits(C, D, W, CFG['CDIM_CLIP_TAU_INV'], CFG['CDIM_DINO_TAU_INV'],
                                             CFG['CDIM_DETACH_PROJ'], M_fixed, m_mix, P_lab, lab_mix) + 0.5 * P_l
                    loss, p, d = gt_im_loss(Z, CFG['CDIM_RHO']); diag.append(d)
                    pred, ema_prior = ola_predict(Z, p, ema_prior, CFG['CDIM_ALPHA'])
                    final_logits = Z
                else:
                    logits = clf.logit_scale * C @ W.t() + P_l
                    pred = logits.argmax(1)
                    final_logits = logits
                    if method == 'source':
                        pass
                    elif method == 'tda':
                        z = logits.detach()
                        tda.update(C.detach(), z.argmax(1), entropy_from_logits(z))
                        pred = (z + tda.logits(C.detach())).argmax(1)
                    elif method == 'tent':
                        loss = entropy_from_logits(logits).mean()
                    elif method == 'eta':
                        loss, eta_m, d = eta_loss(logits, eta_m); diag.append(d)
                    elif method == 'sar':
                        ent = entropy_from_logits(logits)
                        sar_sel = ent.detach() < E0
                        loss = ent[sar_sel].mean() if sar_sel.any() else None
                    elif method == 'batclip':
                        loss, d = batclip_loss(logits, C, W); diag.append(d)
                    else:
                        raise ValueError(method)
            correct += (pred.cpu() == yb).sum().item()
            if keep_pred:
                pred_all[idx] = pred.cpu().numpy()
            if proto_tta:                                # 확장 A: 다음 배치부터 쓸 프로토타입 갱신 (클래스마다 같은 속도의 EMA)
                with torch.no_grad():
                    conf = entropy_from_logits(final_logits.detach()) < E0
                    pl, eta, new = pred.to(R.device), CFG['PROTO_TTA_MOM'], proto.clone()
                    for c in range(K):
                        msk = conf & (pl == c)
                        if msk.any():
                            new[c] = F.normalize((1 - eta) * proto[c] + eta * F.normalize(D[msk].mean(0), dim=0), dim=0)
                    proto = new
            if opt is None or loss is None:
                continue
            if method == 'sar':
                if dino_params:                          # dln·dlnh: SAM 섭동·gradient를 DINO LayerNorm에도 (2026-11-23)
                    sk, l2 = sar_update(clf, params + dino_params, opt, scaler, loss, x, sar_sel, head_live)
                else:
                    P_d = P_l.detach() if torch.is_tensor(P_l) else P_l
                    sk, l2 = sar_update(clf, params, opt, scaler, loss, x, sar_sel,
                                        clip_head if head_on_clip else (lambda Cq: P_d))
                skipped += sk
                if l2 is not None:                       # SAR model recovery: 손실 EMA가 너무 작으면(붕괴) 리셋
                    sar_ema = l2 if sar_ema is None else 0.9 * sar_ema + 0.1 * l2
                    if sar_ema < CFG['SAR_RESET']:
                        clf.load_snapshot(base_snap); opt.state.clear(); sar_ema = None; n_reset += 1
                        with torch.no_grad():            # DINO도 원래 상태로
                            for p, v in zip(dino_params, dino_snap): p.copy_(v)
            else:
                skipped += optimizer_step(loss, opt, scaler)
    finally:
        for p in params: p.requires_grad_(False)
        if dino_params:                                   # DINO를 원래 상태로 (다음 corruption·다른 job·특징 캐시 계산 보호)
            with torch.no_grad():
                for p, v in zip(dino_params, dino_snap):
                    p.copy_(v); p.requires_grad_(False)
    if R.device == 'cuda': torch.cuda.synchronize()
    out = dict(acc=100.0 * correct / n, skipped=skipped, tta_time_s=time.time() - t0)
    if keep_pred:
        out['pred'] = pred_all
    if method == 'sar':
        diag.append(dict(resets=n_reset))
    if diag:
        out['diag_last'] = json.dumps(diag[-1])
    return out


# ---------------- DINO feature 캐시 ----------------
@torch.no_grad()
def dino_features(x_u8, bs=250):
    out = []
    for s in range(0, len(x_u8), bs):
        x = dino_input(x_u8[s:s + bs])
        with autocast():
            f = M.dino(x)
        out.append(F.normalize(f.float(), dim=-1).half().cpu())
    return torch.cat(out)


@torch.no_grad()
def dino_prototypes(x_u8, y):
    """D4: 라벨 있는 source 이미지(16-shot subset)의 DINO 특징으로 클래스 프로토타입 (정규화 특징의 클래스 평균, 다시 정규화)."""
    Dtr = dino_features(x_u8).float().to(R.device)
    yt = torch.from_numpy(y).to(R.device)
    return F.normalize(torch.stack([Dtr[yt == c].mean(0) for c in range(K)]), dim=-1)


@torch.no_grad()
def dino_head_features(x_u8, feat='cls', flip=False, bs=250):
    """head용 DINO 특징 (정규화, fp16 cpu). cls = CLS 토큰(dino_features와 같음), cp = CLS와 patch 평균을 각각 정규화해
    이어 붙인 1536차원. flip이면 좌우반전한 이미지의 특징."""
    out = []
    for s in range(0, len(x_u8), bs):
        xb = x_u8[s:s + bs]
        if flip:
            xb = np.ascontiguousarray(np.asarray(xb)[:, :, ::-1])
        with autocast():
            o = M.dino.forward_features(dino_input(xb))
        f = F.normalize(o['x_norm_clstoken'].float(), dim=-1)
        if feat == 'cp':
            f = torch.cat([f, F.normalize(o['x_norm_patchtokens'].float().mean(1), dim=-1)], 1)
        out.append(F.normalize(f, dim=-1).half().cpu())
    return torch.cat(out)


def dino_live_feats(x_u8, feat='cls', qf=False, grad=False):
    """dln·dlnh: 지금 상태의 DINO로 배치 특징을 계산 (dino_head_features와 같은 식, device float).
    돌려주는 것: (head query 특징, 정규화한 CLS 특징). grad면 원본 view만 gradient를 남기고 반전 view는 no_grad."""
    def one(xb, g):
        with torch.set_grad_enabled(g), autocast():
            o = M.dino.forward_features(dino_input(xb))
        cls = F.normalize(o['x_norm_clstoken'].float(), dim=-1)
        f = cls if feat == 'cls' else F.normalize(
            torch.cat([cls, F.normalize(o['x_norm_patchtokens'].float().mean(1), dim=-1)], 1), dim=-1)
        return f, cls
    f, cls = one(x_u8, grad)
    if qf:
        f2, _ = one(np.ascontiguousarray(np.asarray(x_u8)[:, :, ::-1]), False)
        f = F.normalize(f + f2, dim=-1)
    return f, cls


@torch.no_grad()
def dino_prototypes_v(x_u8, y, flags, seed):
    """head 개선 후보(head_flags)의 프로토타입. cp: CLS+patch 특징, a8: 8개 view 평균, ctr: 160장 평균 μ_s를 빼고 정규화.
    diag_head2.py와 같은 계산 (a8의 crop 난수 = RandomState(1000 + subset seed))."""
    feat = 'cp' if 'cp' in flags else 'cls'
    views = [dino_head_features(x_u8, feat)]
    if 'a8' in flags:
        from core.source import crop_flip_u8
        rng = np.random.RandomState(1000 + seed)
        views += [dino_head_features(x_u8, feat, flip=True)]
        views += [dino_head_features(crop_flip_u8(np.asarray(x_u8), rng), feat) for _ in range(6)]
    Fs = F.normalize(torch.stack([v.float() for v in views]).mean(0), dim=-1).to(R.device)
    yt = torch.from_numpy(y).to(R.device)
    Pm = torch.stack([Fs[yt == c].mean(0) for c in range(K)])
    if 'ctr' in flags:
        Pm = Pm - Fs.mean(0)
    return F.normalize(Pm, dim=-1)


@torch.no_grad()
def clip_image_feats(clf, x_u8, bs=250):
    Cs = []
    for s in range(0, len(x_u8), bs):
        with autocast():
            Cs.append(clf.image_features(clip_input(x_u8[s:s + bs])))
    return torch.cat(Cs)


@torch.no_grad()
def clip_prototypes(clf, x_u8, y):
    """D4 ablation: 같은 방식으로 CLIP 이미지 특징(TTA 전 source 모델)의 클래스 프로토타입."""
    C = clip_image_feats(clf, x_u8)
    yt = torch.from_numpy(y).to(R.device)
    return F.normalize(torch.stack([C[yt == c].mean(0) for c in range(K)]), dim=-1)


@torch.no_grad()
def tip_cache(src, clf, x_u8, y):
    """Tip-Adapter 캐시: 라벨 있는 이미지 특징 전부(키) + one-hot 라벨(값). src = tip_clip | tip_dino."""
    keys = clip_image_feats(clf, x_u8) if head_base(src) == 'tip_clip' else dino_features(x_u8).float().to(R.device)
    vals = F.one_hot(torch.from_numpy(y).to(R.device), K).float()
    return keys, vals


@torch.no_grad()
def source_projection(clf, x_u8, bs=250):
    """D2: clean source 이미지로 CDIM 투영 행렬 M = Cᵀ D / N을 추정 (TTA 전 source 모델 기준, 이후 고정)."""
    Cs, Ds = [], []
    for s in range(0, len(x_u8), bs):
        xb = x_u8[s:s + bs]
        with autocast():
            Cs.append(clf.image_features(clip_input(xb)))
        Ds.append(dino_features(xb).float().to(R.device))
    C, D = torch.cat(Cs), torch.cat(Ds)
    return C.t() @ D / len(C)


_CORR_CACHE = {}


def _cached_feats(name, suffix, compute):
    """corruption별 DINO 특징 디스크 캐시. 다른 GPU 프로세스가 동시에 만들어도 반쯤 쓴 파일을 읽지 않게 원자적으로 교체."""
    cache_dir = f"{CFG['DATA_DIR']}/dino_cache"
    os.makedirs(cache_dir, exist_ok=True)
    tag = '' if CFG['RESIZE'] == 'torch_bicubic' else f"_{CFG['RESIZE']}"   # 전처리가 다르면 다른 캐시
    tag += '' if CFG['DATASET'] == 'cifar10' else f"_{CFG['DATASET']}"       # 데이터셋이 다르면 다른 캐시 (CIFAR-10 이름은 예전 그대로)
    path = f"{cache_dir}/{name}_s{CFG['SEVERITY']}_dinov2_vitb14{tag}{suffix}.pt"
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    d = compute()
    tmp = f'{path}.tmp{os.getpid()}'
    torch.save(d, tmp)
    os.replace(tmp, path)
    return d


def get_corruption(name):
    """(x_u8, y, dino_feats). DINOv2는 frozen이라 corruption마다 한 번만 계산해 디스크에 저장."""
    if name not in _CORR_CACHE:
        _CORR_CACHE.clear()
        x, y = load_corruption(name, CFG['SEVERITY'])
        d = _cached_feats(name, '', lambda: dino_features(x))
        assert d.shape[0] == len(y)
        _CORR_CACHE[name] = (x, y, d)
    return _CORR_CACHE[name]


_HEADQ_CACHE = {}


def get_head_query(name, flags):
    """head query 특징 (head_flags의 cp·qf). 둘 다 없으면 None (= CDIM과 같은 CLS 특징 D를 그대로 씀)."""
    feat = 'cp' if 'cp' in flags else 'cls'
    if feat == 'cls' and 'qf' not in flags:
        return None
    key = (name, feat, 'qf' in flags)
    if key not in _HEADQ_CACHE:
        _HEADQ_CACHE.clear()
        x, _, d = get_corruption(name)
        q = d if feat == 'cls' else _cached_feats(name, '_cp', lambda: dino_head_features(x, 'cp'))
        if 'qf' in flags:
            qf = _cached_feats(name, f'_{feat}_flip', lambda: dino_head_features(x, feat, flip=True))
            q = F.normalize(q.float() + qf.float(), dim=-1).half()
        assert q.shape[0] == len(x)
        _HEADQ_CACHE[key] = q
    return _HEADQ_CACHE[key]
