"""실험 설정. 조건은 이 파일에서만 바꾸세요.
결과 CSV의 key에 조건이 모두 기록되므로, 조건을 바꿔도 기존 결과와 섞이지 않습니다."""
import os

HOME = os.path.expanduser('~')
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# 데이터셋 (2026-11-24 확장 실험). 환경 변수로 고른다: 없으면 CIFAR-10(지금까지의 모든 결과),
#   DATASET=cifar100이면 CIFAR-100 / CIFAR-100-C. 결과·체크포인트·DINO 특징 캐시가 모두 따로 저장된다.
DATASET = os.environ.get('DATASET', 'cifar10')
assert DATASET in ('cifar10', 'cifar100'), f'DATASET은 cifar10 또는 cifar100: {DATASET}'
# 추가 실험 (2026-11-25 사용자 "1,3,2 순서로 하자"): 환경 변수로 손상 강도와 TTA 프로토콜을 고른다. 없으면 지금까지와 같다.
#   SEVERITY=3          → 약한 손상. 결과는 <기본 실험>_s3
#   PROTOCOL=continual  → 손상 15종을 리셋 없이 이어서 (core.runner._run_job_continual). 결과는 <기본 실험>_cont
#   CLIP 체크포인트는 테스트 손상·프로토콜과 무관하므로 기본 실험 폴더(CKPT_EXP_NAME)의 것을 같이 쓴다.
SEVERITY = int(os.environ.get('SEVERITY', '5'))
PROTOCOL = os.environ.get('PROTOCOL', 'episodic')
assert SEVERITY in (1, 2, 3, 4, 5), f'SEVERITY는 1~5: {SEVERITY}'
assert PROTOCOL in ('episodic', 'continual'), f'PROTOCOL은 episodic 또는 continual: {PROTOCOL}'
_BASE_EXP = 'server_v100_pil' if DATASET == 'cifar10' else 'server_v100_pil_c100'   # v1(torch bicubic)과도 분리

CFG = dict(
    DATASET   = DATASET,
    # ---- 경로 (홈은 master·gpu01·gpu02가 NFS로 공유 → 어느 노드에서든 같은 경로) ----
    DATA_DIR  = f'{HOME}/cdim_data',                 # CIFAR-10(-C), CIFAR-100(-C), CLIP·DINOv2 가중치
    EXP_ROOT  = f'{PROJECT_DIR}/experiments',
    EXP_NAME  = _BASE_EXP + ('' if SEVERITY == 5 else f'_s{SEVERITY}') + ('' if PROTOCOL == 'episodic' else '_cont'),
    CKPT_EXP_NAME = _BASE_EXP,
    PROTOCOL  = PROTOCOL,

    # ---- 데이터 감소 (Phase A: 비율 sweep) ----
    RATIOS    = [1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64],
    SEEDS_MANY = [0, 1, 2],
    MANY_SEEDS_BELOW = 1/16,                         # ratio <= 이 값이면 SEEDS_MANY, 아니면 [0]

    # ---- source 학습 ----
    HEAD_MODE_MAIN = 'text_init',                    # norm + 텍스트 초기화 head
    RUN_NORM_ONLY_ABLATION = True,
    SRC_AUG   = 'none',                              # Phase A용. k-shot은 job마다 따로 지정
    SRC_BATCH = 128,
    SRC_EPOCHS_FULL = 2,
    SRC_LR    = 1e-4,
    SRC_WD_HEAD = 0.0,
    SRC_WARMUP_FRAC = 0.05,
    BUDGETS   = ['same_epochs', 'same_iters'],
    SAME_ITERS_RATIOS = [1/8, 1/32, 1/64],
    LOGIT_SCALE = 100.0,

    # ---- 테스트 / TTA (BATCLIP·CDIM 논문 조건) ----
    RESIZE    = 'pil_bicubic',                       # 32→224: CLIP 공식 preprocess(PIL bicubic). zero-shot Source가 논문과
                                                     # 4종 평균 −0.01%p로 일치. 'torch_bicubic'(v1)은 −4.06%p (RESULTS.md)
    SEVERITY  = SEVERITY,                            # 기본 5. 위의 환경 변수 SEVERITY
    CORR_SWEEP = ['gaussian_noise', 'defocus_blur', 'snow', 'contrast'],
    CORR_ALL  = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast',
                 'elastic_transform', 'pixelate', 'jpeg_compression'],
    METHODS_SWEEP = ['source', 'batclip', 'cdim'],
    TTA_BATCH = 200,
    TTA_LR    = 1e-3,
    TTA_WD    = 0.0,
    STREAM_SHUFFLE = True,
    STREAM_SEED = 1,

    # ---- CDIM ----
    CDIM_RHO  = 0.05,
    CDIM_ALPHA = 0.9,
    CDIM_CLIP_TAU_INV = 100.0,                       # 설계 문서 결정: 20이면 Gaussian 재현 −8.71%p, 100이면 −0.81%p
                                                     # 20 비교는 `sanity --tau 20 --methods cdim`으로만
    CDIM_DINO_TAU_INV = 20.0,
    CDIM_DETACH_PROJ = False,
    PROTO_TAU_INV = 50.0,       # D4: 16-shot DINO 프로토타입 logit 온도 (diag_dino.py와 같은 값). logit = λ·50·D Pᵀ
    # ---- 같은 범주 baseline (재구현). 값은 각 논문의 기본값 ----
    TIP_ALPHA = 1.0, TIP_BETA = 5.5,                 # Tip-Adapter (Zhang et al., ECCV 2022) 학습 없는 버전, ImageNet 기본값
    TDA_POS_CAP = 3, TDA_POS_ALPHA = 2.0, TDA_POS_BETA = 5.0,   # TDA (Karmanov et al., CVPR 2024) positive cache, ImageNet 설정
    ETA_D_MARGIN = 0.4,     # EATA (Niu et al., ICML 2022) 중복 샘플 기준. 0.05는 ImageNet(1000클래스)용이라 10클래스에서는
                            # 첫 배치 이후 모든 샘플이 제외됨(smoke test kept=0) → CIFAR-10 설정으로 알려진 0.4 (원 논문 확인 필요)
    SAR_RHO = 0.05, SAR_RESET = 0.2,                 # SAR (Niu et al., ICLR 2023) SAM 반경, model recovery 기준
    PROTO_TTA_MOM = 0.05,       # 확장 A: 프로토타입 TTA 갱신의 클래스별 EMA 속도 (배치마다 5%씩 test 쪽으로)

    # ---- 실행 환경 (V100 32GB) ----
    GRAD_CKPT = False,          # 32GB면 보통 불필요. OOM이 나면 자동으로 켜고 그 job을 다시 시도
    CPU_THREADS = 4,            # 매뉴얼 8.3: GPU 노드 CPU는 공용 → 4~8로 제한
    FP16_INIT_SCALE = 2.0 ** 10,
    KNEE_DROP = 3.0,
)

# ---- 16-shot 실험 (설계 문서 Method · Results의 RQ1·RQ3 표) ----
ITERS_MAIN = 200   # 문서 결정: 모든 16-shot 변형을 200 iteration으로 고정해 학습량 효과를 분리
ITERS_RQ2 = 782    # RQ2 표의 16-shot 행: 비율 sweep(full 2 epoch = same iters)과 같은 학습량


def J(shots, seed, iters=None, aug='none', mixup=0.0, cutmix=0.0, jsd=0.0, src_loss='ce', align=0.0,
      wise=1.0, cdim_m='batch', head_mode='text_init', methods=None, proto=0.0, proto_src='dino'):
    """k-shot job 하나 = source 모델 하나 (+ 학습 후 단계인 WiSE 계수·D2 투영 방식·D4 프로토타입).
    aug: none | basic(A1) | ta(A2) | augmix(A3),  mixup·cutmix: A4 Beta(α,α),  jsd: A3 JSD 가중치,
    src_loss: ce | ens(D1),  align: D3 λ,  cdim_m: batch(기본) | src(D2 고정) | mix(D2 0.5:0.5)
    proto: D4 λ. 16-shot DINO 클래스 프로토타입 logit λ·PROTO_TAU_INV·D Pᵀ를 TTA logit에 더함 (0이면 끔, 재학습 없음)
    proto_src: dino(D4) | clip(ablation: 같은 방식을 CLIP 이미지 특징으로) | tip_clip·tip_dino(Tip-Adapter 캐시 baseline)
    subset은 클래스별 seed 고정 무작위 추출 (CoOp 방식, seed가 같으면 1⊂2⊂4⊂8⊂16)."""
    j = dict(head_mode=head_mode, shots=shots, iters=ITERS_MAIN if iters is None else iters, aug=aug,
             mixup=float(mixup), cutmix=float(cutmix), jsd=float(jsd), src_loss=src_loss, align=float(align),
             wise=float(wise), cdim_m=cdim_m, proto=float(proto), proto_src=proto_src, seed=seed)
    if methods:
        j['methods'] = methods     # 결과 key에는 들어가지 않음 (돌릴 TTA 방법만 지정)
    return j


KSHOT_STAGES = {
    # 문서 Phase A — 탐색 (subset 1개 = seed 0, corruption 4종)
    # base: 학습량 곡선. 200이 기준, 782는 RQ2 표의 16-shot 행으로도 사용
    'A_base': [J(16, 0, iters=it) for it in (ITERS_MAIN, 50, 400, ITERS_RQ2)],
    # (a) 증강: A1 crop+flip, A2 A1+safe TrivialAugment, A3 A1+AugMix (+JSD), A4 A1+Mixup / CutMix
    'A_aug': [J(16, 0, aug='basic'), J(16, 0, aug='ta'), J(16, 0, aug='augmix'), J(16, 0, aug='augmix', jsd=12),
              J(16, 0, aug='basic', mixup=0.2), J(16, 0, aug='basic', cutmix=1.0)],
    # (b) DINO 활용: D1 앙상블 logit 학습, D2 source M 고정·혼합(base 모델 재사용, CDIM만), D3 관계 정렬
    'A_dino': [J(16, 0, src_loss='ens'),
               J(16, 0, cdim_m='src', methods=['cdim']), J(16, 0, cdim_m='mix', methods=['cdim']),
               J(16, 0, align=1.0)],
    # 문서 Phase C — Phase A 상위 변형을 subset 3개(seed 0·1·2)로 검증. seed 0은 위 stage에 있으므로 1·2만 추가.
    # seed는 subset 추출과 학습 난수를 함께 바꾼다. 순서 주의: D2 job은 같은 seed의 base·AugMix 체크포인트를 재사용하므로 그 뒤에 둔다.
    'C_subsets': [J(16, s, iters=it) for s in (1, 2) for it in (ITERS_MAIN, ITERS_RQ2)]   # base 200, RQ2용 782
                 + [J(16, s, aug='augmix') for s in (1, 2)]                                  # A3 AugMix
                 + [J(16, s, src_loss='ens') for s in (1, 2)]                                # D1
                 + [J(16, s, cdim_m='src', methods=['cdim']) for s in (1, 2)]                # D2 source M 고정
                 + [J(16, s, aug='augmix', cdim_m='src', methods=['cdim']) for s in (0, 1, 2)],  # 조합: AugMix + D2
    # WiSE-FT: 학습된 visual LN·head를 zero-shot 값과 섞음 θ = (1−α)·θ_zs + α·θ_ft (재학습 없음, base 체크포인트 재사용).
    # 근거: TTA 후 정확도는 clean 정확도로 거의 정해지는데 zero-shot만 +2.3%p 위에 있음 → 미세조정에서 잃는 강건성을 되찾는지 확인.
    'C_wise': [J(16, s, wise=a) for a in (0.5, 0.75) for s in (0, 1, 2)],
    # D4: 16장의 라벨로 만든 DINO 클래스 프로토타입 logit을 TTA logit에 더함 (Source·BATCLIP·CDIM 모두, base 체크포인트 재사용).
    # 근거: diag_dino.py에서 TTA 없이 clean 97.5%, C 4종 81.6% (RESULTS.md ⑨)
    'D_proto': [J(16, s, proto=lam) for lam in (1.0, 0.5) for s in (0, 1, 2)],
    # D4 해석. iters=0 + head_mode='zeroshot' = CLIP 학습 없음(zero-shot 그대로), shots는 프로토타입에 쓰는 라벨 수.
    'E_interp': [J(16, 0, iters=0, head_mode='zeroshot')]                                    # 검증: sanity zero-shot 값과 같아야 함
                + [J(16, s, iters=0, head_mode='zeroshot', proto=1.0) for s in (0, 1, 2)]    # (a) zero-shot CLIP + D4
                + [J(k, s, iters=0, head_mode='zeroshot', proto=1.0, methods=['source', 'cdim'])
                   for k in (1, 2, 4, 8) for s in (0, 1, 2)]                                   # (b) 라벨 수 곡선
                + [J(5000, 0, iters=ITERS_RQ2), J(5000, 0, iters=ITERS_RQ2, proto=1.0)],      # (c) full 데이터 ± D4 (상한)
    # TENT + D4: TTA 방법이 하나 더 있어도 D4 효과가 일관된지 (TENT 열만 추가, 다른 방법 결과는 기존 것과 합쳐 봄)
    'F_tent': [J(16, 0, iters=0, head_mode='zeroshot', methods=['tent'])]
              + [J(16, s, iters=0, head_mode='zeroshot', proto=1.0, methods=['tent']) for s in (0, 1, 2)]
              + [J(16, s, methods=['tent']) for s in (0, 1, 2)]
              + [J(16, s, proto=1.0, methods=['tent']) for s in (0, 1, 2)],
    # ablation: 프로토타입을 CLIP 특징으로 (zero-shot CLIP, 같은 16장). DINO라서 좋은지 확인 → "다수의 파운데이션 모델"의 근거
    # CLIP 특징은 클래스 간 유사도 차이가 작아 λ를 넓게 봄. Source = TTA 없음
    'F_clipproto': [J(16, s, iters=0, head_mode='zeroshot', proto=lam, proto_src='clip', methods=['source', 'cdim'])
                    for lam in (1.0, 2.0, 4.0) for s in (0, 1, 2)],
}

# ---- 최종 검증과 같은 범주 baseline: corruption 15종, 모두 같은 조건 (subset 3개, 같은 source 모델, 같은 TTA 프로토콜) ----
FINAL_METHODS = ['source', 'tent', 'batclip', 'cdim']


def _final_configs(**kw):
    """최종 표의 행: zero-shot, zero-shot + D4, 16-shot base, 16-shot + D4, full 데이터 ± D4 (참고)."""
    return ([J(16, 0, iters=0, head_mode='zeroshot', **kw)]
            + [J(16, s, iters=0, head_mode='zeroshot', proto=1.0, **kw) for s in (0, 1, 2)]
            + [J(16, s, **kw) for s in (0, 1, 2)]
            + [J(16, s, proto=1.0, **kw) for s in (0, 1, 2)]
            + [J(5000, 0, iters=ITERS_RQ2, **kw), J(5000, 0, iters=ITERS_RQ2, proto=1.0, **kw)])


KSHOT_STAGES.update({
    # 최종 검증 (문서 계획: 15종). 탐색 4종에서 이미 돈 run은 다시 돌지 않는다 (key에 stage 없음)
    'H_final': _final_configs(methods=FINAL_METHODS),
    # 같은 범주 baseline (1) few-shot head: Tip-Adapter(160장 CLIP 특징 캐시), DINO 캐시(같은 방식, CaFo 계열).
    # D4와 같은 zero-shot CLIP·같은 16장·같은 TTA 방법들. 학습 없음
    'G_fewshot': [J(16, s, iters=0, head_mode='zeroshot', proto=1.0, proto_src=src, methods=FINAL_METHODS)
                  for src in ('tip_clip', 'tip_dino') for s in (0, 1, 2)],
    # 같은 범주 baseline (2) TTA: ETA, SAR, TDA(positive cache) — 최종 표의 모든 행에서 (full 참고 행 제외)
    'G_tta': _final_configs(methods=['eta', 'sar', 'tda'])[:-2],
    # 확장 A: D4 프로토타입을 TTA 중에 확신 있는 test DINO 특징으로 갱신 (gradient 없음, Source에 붙이면 학습 없는 TTA)
    # 확장 B: CDIM의 DINO 분기 분류기(라벨 없는 투영 T_D)를 라벨 프로토타입으로 교체(lab)·혼합(lab_mix), 그리고 교체 + D4 head
    'I_ext': [J(16, s, iters=0, head_mode='zeroshot', proto=1.0, proto_src='dino_tta', methods=FINAL_METHODS)
              for s in (0, 1, 2)]
             + [J(16, s, iters=0, head_mode='zeroshot', cdim_m=cm, methods=['cdim'])
                for cm in ('lab', 'lab_mix') for s in (0, 1, 2)]
             + [J(16, s, iters=0, head_mode='zeroshot', proto=1.0, cdim_m='lab', methods=['cdim']) for s in (0, 1, 2)],
})

# 같은 범주 baseline (1)의 재조정 (2026-11-19). G_fewshot은 Tip-Adapter 논문 기본값(α=1, β=5.5)이라 캐시 logit이 CLIP logit에 묻혔다.
# D4의 λ와 같은 규칙(zero-shot CLIP + head, TTA 없는 탐색용 4종 평균 최대, subset 3개)으로 고른 값 — diag_tip.py, analysis/diag_tip.csv.
# 이 규칙은 D4에서 실제로 쓴 λ=1을 고른다. Tip grid(α 10개 × β 8개)는 D4(λ 5개)보다 넓다.
#   CIFAR-100(2026-11-25 추가 실험 1): 같은 규칙으로 diag_tip.py가 고른 값을 읽는다(없으면 빈 dict → Tip stage가 비어 있음).
def _tip_tuned():
    if DATASET == 'cifar10':
        return {'tip_clip': (2.0, 8.0), 'tip_dino': (10.0, 1.0)}     # (α, β)
    import json
    p = f"{CFG['EXP_ROOT']}/{_BASE_EXP}/analysis/diag_tip_choice.json"
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        ch = json.load(f)
    return {h: (float(ch[h]['alpha']), float(ch[h]['beta'])) for h in ('tip_clip', 'tip_dino')}


TIP_TUNED = _tip_tuned()
KSHOT_STAGES['G_fewshot_tuned'] = [
    J(16, s, iters=0, head_mode='zeroshot', proto=a / CFG['TIP_ALPHA'], methods=FINAL_METHODS,
      proto_src=src if b == CFG['TIP_BETA'] else f'{src}@{b:g}')      # '@β' = core.methods.tip_beta
    for src, (a, b) in TIP_TUNED.items() for s in (0, 1, 2)]
# main table의 빈칸 채우기: 조정된 Tip 계열 head × ETA·SAR·TDA (D4 행과 같은 TTA 방법 7개를 모두 갖추게)
KSHOT_STAGES['G_tta_tip'] = [dict(j, methods=['eta', 'sar', 'tda']) for j in KSHOT_STAGES['G_fewshot_tuned']]

# 교수님 면담 (2026-11-19): "Full 5만 장 → 16장 → 16장 + 제안" 서사의 Full 기준선 (RESULTS.md ★★)
#   P1 K_full_fill : 기존 full 모델(2 epoch = 782 iter)의 빈칸 ETA·SAR·TDA
#   P2 K_full_ep5/10: full을 5·10 epoch(1955·3910 iter)까지 학습. 수렴 판단은 clean 정확도로(손상 test로 고르지 않음).
#                     어느 쪽으로 정해지든 Table A가 바로 채워지도록 7개 방법 × 15종을 모두 돌린다
ALL_METHODS = ['source', 'tent', 'eta', 'sar', 'tda', 'batclip', 'cdim']
KSHOT_STAGES['K_full_fill'] = [J(5000, 0, iters=ITERS_RQ2, methods=['eta', 'sar', 'tda'])]
KSHOT_STAGES['K_full_ep5'] = [J(5000, 0, iters=1955, methods=ALL_METHODS)]
KSHOT_STAGES['K_full_ep10'] = [J(5000, 0, iters=3910, methods=ALL_METHODS)]
#   P3 K_full_rep_s1/s2: 기준선 ①(10 epoch)의 반복. 데이터는 5만 장 그대로, seed는 학습 순서만 바꾼다.
#                        회복 서사는 CDIM만 본다(사용자 결정 2026-11-19) → CDIM만. GPU 2장에 seed 하나씩
KSHOT_STAGES['K_full_rep_s1'] = [J(5000, 1, iters=3910, methods=['cdim'])]
KSHOT_STAGES['K_full_rep_s2'] = [J(5000, 2, iters=3910, methods=['cdim'])]
#   P5 L_p5      : Table A ②의 수렴판. 16장 782 iter(⑦에서 학습한 체크포인트, 4종 결과도 있음) × CDIM × 15종
#   P4 L_curve_* : 라벨 수 곡선 (CDIM만, 15종). baseline = 그 라벨로 CLIP 학습(782 iter, 5000장은 ① 10 epoch 재사용),
#                  ours = zero-shot CLIP + 같은 이미지로 만든 Few-shot DINO head. 두 줄이 같은 이미지(kshot_subset)를 쓴다.
#                  seed 수는 sweep 규칙(MANY_SEEDS_BELOW = 1/16): 클래스당 625장은 1개, 그 이하는 3개.
#                  5000장 ours는 subset이 seed와 무관(전체)하고 학습이 없어 1회
CURVE_K = (5000, 625, 157, 79, 16, 8, 4, 2, 1)
_curve_seeds = lambda k: [0] if k >= 625 else [0, 1, 2]
KSHOT_STAGES['L_p5'] = [J(16, s, iters=ITERS_RQ2, methods=['cdim']) for s in (0, 1, 2)]
KSHOT_STAGES['L_curve_ours'] = [J(k, s, iters=0, head_mode='zeroshot', proto=1.0, methods=['cdim'])
                                for k in CURVE_K for s in _curve_seeds(k)]
KSHOT_STAGES['L_curve_base_a'] = [J(k, s, iters=ITERS_RQ2, methods=['cdim']) for k in (625, 157) for s in _curve_seeds(k)]
KSHOT_STAGES['L_curve_base_b'] = [J(k, s, iters=ITERS_RQ2, methods=['cdim']) for k in (79, 8, 4, 2, 1)
                                  for s in _curve_seeds(k)]
# 조건 통일 (2026-11-20 사용자: "전체적으로 실험 조건을 동일하게"). 보고서의 모든 표가 같은 조건을 쓴다:
#   16장 CLIP 미세조정 = FT_ITERS(782 iter, 측정한 학습량 중 최고), full = FULL_ITERS(10 epoch), 반복 3회(subset 또는 학습 seed)
#   M_main  : Main table·ablation 칸 — 16장 782 source(TTA 없음), 16장 782 + head × CDIM, full 10 epoch + head × CDIM(상한)
#   M_ft    : 부록 — 16장 782 × 나머지 TTA 방법 (CDIM은 P5, source는 M_main)
#   M_fthead: 부록 — 16장 782 + head × 7개 방법
#   M_full  : 부록 — full 10 epoch 학습 seed 1·2 × CDIM 외 6개 방법 (CDIM은 P3)
FT_ITERS = ITERS_RQ2
FULL_ITERS = 3910
KSHOT_STAGES['M_main'] = ([J(16, s, iters=FT_ITERS, methods=['source']) for s in (0, 1, 2)]
                          + [J(16, s, iters=FT_ITERS, proto=1.0, methods=['cdim']) for s in (0, 1, 2)]
                          + [J(5000, s, iters=FULL_ITERS, proto=1.0, methods=['cdim']) for s in (0, 1, 2)])
KSHOT_STAGES['M_ft'] = [J(16, s, iters=FT_ITERS, methods=['tent', 'eta', 'sar', 'tda', 'batclip']) for s in (0, 1, 2)]
KSHOT_STAGES['M_fthead'] = [J(16, s, iters=FT_ITERS, proto=1.0, methods=ALL_METHODS) for s in (0, 1, 2)]
KSHOT_STAGES['M_full'] = [J(5000, s, iters=FULL_ITERS, methods=[m for m in ALL_METHODS if m != 'cdim']) for s in (1, 2)]
# head 개선 후보 (2026-11-20, 사용자 "조금이라도 더 올릴 수 있는 방법"). 라벨 160장·CLIP 학습 없음은 그대로, head만 바꾼다.
#   diag_head2.py(TTA 없음, 탐색용 4종)에서 고른 후보를 CDIM × 탐색용 4종 × subset 3개로 확인 → 가장 좋은 것만 15종 (N_head15)
#   proto_src 플래그(core.methods.head_flags): cp = CLS + patch 평균 특징, ctr = 특징 중심화(test는 online 평균), qf = test 좌우반전 평균,
#   a8 = 프로토타입 8 view 평균
HEAD_EXPLORE = ['dino+ctr', 'dino+cp', 'dino+cp+ctr', 'dino+cp+ctr+qf', 'dino+cp+ctr+qf+a8']
_zh = lambda src, s, **kw: J(16, s, iters=0, head_mode='zeroshot', proto=1.0, proto_src=src, **kw)
KSHOT_STAGES['N_head_a'] = [_zh(src, s, methods=['cdim']) for src in HEAD_EXPLORE[:3] for s in (0, 1, 2)]
KSHOT_STAGES['N_head_b'] = [_zh(src, s, methods=['cdim']) for src in HEAD_EXPLORE[3:] for s in (0, 1, 2)]
# 4종 결과(CDIM, 기존 ours 대비 같은 subset 차이): ctr +0.34, cp +0.30, cp+ctr +0.41, cp+ctr+qf +0.56, +a8 +0.56 (a8는 효과 없음)
# → 15종: cp+ctr(추가 계산 없음)과 cp+ctr+qf(DINO 2배), 각각 zero-shot CLIP과 16장 782 iter CLIP 위에서.
#   4종 결과가 있는 GPU의 태그 파일에서 이어서 돈다 (_a = head_gpu0, _b = head_gpu1)
KSHOT_STAGES['N_head15_a'] = ([_zh('dino+cp+ctr', s, methods=['cdim']) for s in (0, 1, 2)]
                              + [J(16, s, iters=FT_ITERS, proto=1.0, proto_src='dino+cp+ctr', methods=['cdim'])
                                 for s in (0, 1, 2)])
KSHOT_STAGES['N_head15_b'] = ([_zh('dino+cp+ctr+qf', s, methods=['cdim']) for s in (0, 1, 2)]
                              + [J(16, s, iters=FT_ITERS, proto=1.0, proto_src='dino+cp+ctr+qf', methods=['cdim'])
                                 for s in (0, 1, 2)])
# 새 ours (2026-11-20 사용자 결정): 16장 CLIP 미세조정(FT_ITERS) + 새 head(NEW_HEAD) + CDIM = 86.35 (N_head15_b).
#   O_tta  : 부록·ablation — 새 ours의 TTA 방법 교체(Source = TTA 없음 포함)
#   O_comp : ablation — 새 head 구성 요소 하나씩 빼기 (cp 빼기 = dino+ctr+qf, ctr 빼기 = dino+cp+qf. qf 빼기·전부 빼기는 이미 있음)
#   O_full : ablation 상한·라벨 수 곡선 5000장 점 — full 10 epoch + 새 head (학습 seed 3개)
#   O_curve: 라벨 수 곡선(P4)의 ours — 클래스당 k장으로 CLIP 미세조정(782 iter, baseline 곡선 체크포인트 재사용) + 같은 k장으로 새 head
NEW_HEAD = 'dino+cp+ctr+qf'
KSHOT_STAGES['O_tta'] = [J(16, s, iters=FT_ITERS, proto=1.0, proto_src=NEW_HEAD,
                           methods=['source', 'tent', 'eta', 'sar', 'tda', 'batclip']) for s in (0, 1, 2)]
KSHOT_STAGES['O_comp'] = [J(16, s, iters=FT_ITERS, proto=1.0, proto_src=src, methods=['cdim'])
                          for src in ('dino+cp+qf', 'dino+ctr+qf') for s in (0, 1, 2)]
KSHOT_STAGES['O_full'] = [J(5000, s, iters=FULL_ITERS, proto=1.0, proto_src=NEW_HEAD, methods=['cdim']) for s in (0, 1, 2)]
KSHOT_STAGES['O_curve'] = [J(k, s, iters=FT_ITERS, proto=1.0, proto_src=NEW_HEAD, methods=['cdim'])
                           for k in (625, 157, 79, 8, 4, 2, 1) for s in _curve_seeds(k)]
# 1장 진단 (2026-11-22): 새 ours 1장 subset 2만 77.57로 하락. TTA 없이는 재현 안 됨(diag_shot1.py) → CDIM에서 CLIP과 head 중
#   무엇 때문인지 2×2를 채운다. 이미 있음: (미세조정 + 새 head) = ours, (zero-shot + 기존 head) = 이전 ours(L_curve_ours).
#   P_shot1_a: 미세조정 CLIP + 기존 head / P_shot1_b: zero-shot CLIP + 새 head
KSHOT_STAGES['P_shot1_a'] = [J(1, s, iters=FT_ITERS, proto=1.0, proto_src='dino', methods=['cdim']) for s in (0, 1, 2)]
KSHOT_STAGES['P_shot1_b'] = [J(1, s, iters=0, head_mode='zeroshot', proto=1.0, proto_src=NEW_HEAD, methods=['cdim'])
                             for s in (0, 1, 2)]
# 2~8장에서도 같은지 (2026-11-23 사용자 요청): zero-shot CLIP + 새 head. 미세조정 CLIP + 새 head는 O_curve에 있음.
_zs_new = lambda k, s: J(k, s, iters=0, head_mode='zeroshot', proto=1.0, proto_src=NEW_HEAD, methods=['cdim'])
KSHOT_STAGES['P_lowk_a'] = [_zs_new(2, s) for s in (0, 1, 2)] + [_zs_new(4, 0)]
KSHOT_STAGES['P_lowk_b'] = [_zs_new(4, s) for s in (1, 2)] + [_zs_new(8, s) for s in (0, 1, 2)]
# ours를 더 올리기 (2026-11-23 사용자 요청): TTA 중 DINOv2 LayerNorm도 같은 손실·optimizer(lr = TTA_LR, 튜닝 없음)로 적응.
#   선택 규칙대로 먼저 탐색용 4종(KSHOT_CORRS)에서 CDIM. smoke_dln.py: Gaussian 앞 2,000장 70.80 → 76.00.
#   Q_dln_a: 적응한 DINO를 head와 CDIM DINO 분기 모두에 / Q_dln_b: head에만
KSHOT_STAGES['Q_dln_a'] = [J(16, s, iters=FT_ITERS, proto=1.0, proto_src=NEW_HEAD + '+dln', methods=['cdim'])
                           for s in (0, 1, 2)]
KSHOT_STAGES['Q_dln_b'] = [J(16, s, iters=FT_ITERS, proto=1.0, proto_src=NEW_HEAD + '+dlnh', methods=['cdim'])
                           for s in (0, 1, 2)]
# 4종 결과(18:07): ours 87.87 → dlnh 92.13, dln 92.15 (subset 모두 +4.1~4.4). 후보 = dlnh로 미리 정함(차이 0.02로 같은 수준,
#   CDIM은 그대로 두고 적응한 DINO는 head에만 → 이득을 head에 귀속). dln은 ablation.
#   Q15_dlnh·Q15_dln: 15종 검증 (4종은 이미 있어 11종만 돈다: 같은 RESULTS_TAG 파일로 돌릴 것 — dlnh는 dln_gpu1, dln은 dln_gpu0)
#   Q_dln_nohead: head 없이 DINO만 적응 (λ = 1e-6으로 head를 사실상 끔, 적응한 DINO는 CDIM DINO 분기에만), 탐색용 4종
KSHOT_STAGES['Q15_dlnh'] = KSHOT_STAGES['Q_dln_b']
KSHOT_STAGES['Q15_dln'] = KSHOT_STAGES['Q_dln_a']
KSHOT_STAGES['Q_dln_nohead'] = [J(16, s, iters=FT_ITERS, proto=1e-6, proto_src=NEW_HEAD + '+dln', methods=['cdim'])
                                for s in (0, 1, 2)]
# 새 ours 2차 (2026-11-23 사용자 결정 "그래 넣자"): ours = 16장 CLIP 미세조정(FT_ITERS) + 새 head + DINO LN 적응(head에만, dlnh)
#   + CDIM = 90.66 (Q15_dlnh). 표를 다시 채우는 stage (모두 15종):
#   R_nohead : Main baseline "CDIM + DINO LN 적응"(head 없음: λ=1e-6, 적응한 DINO는 CDIM 분기에, dln) — 16장 미세조정 CLIP
#              subset 3개(4종은 Q_dln_nohead에 있어 dln_gpu1 파일에서 이어 돈다) + zero-shot CLIP 1회
#   R_zs     : zero-shot CLIP + 새 head + dlnh × CDIM
#   R_full   : 5만 장 10 epoch + 새 head + dlnh × CDIM (상한, 라벨 수 곡선 5000장 점, 학습 seed 3개)
#   R_comp   : 구성 요소 하나씩 빼기 (dlnh는 유지): −cp, −ctr, −qf, 셋 다 빼기
#   R_tta    : TTA 방법 교체. Source·TDA는 gradient가 없어 DINO가 적응하지 않으므로 O_tta와 같아야 한다(점검용)
#   R_curve_a/b: 라벨 수 곡선(P4)의 ours — 클래스당 k장 CLIP 미세조정 + 같은 k장 새 head + dlnh
OURS_HEAD = NEW_HEAD + '+dlnh'
def _nohead(seed, **kw):
    return J(16, seed, proto=1e-6, proto_src=NEW_HEAD + '+dln', methods=['cdim'], **kw)


KSHOT_STAGES['R_nohead'] = ([_nohead(s, iters=FT_ITERS) for s in (0, 1, 2)]
                            + [_nohead(0, iters=0, head_mode='zeroshot')])
KSHOT_STAGES['R_zs'] = [J(16, s, iters=0, head_mode='zeroshot', proto=1.0, proto_src=OURS_HEAD, methods=['cdim'])
                        for s in (0, 1, 2)]
KSHOT_STAGES['R_full'] = [J(5000, s, iters=FULL_ITERS, proto=1.0, proto_src=OURS_HEAD, methods=['cdim']) for s in (0, 1, 2)]
KSHOT_STAGES['R_comp'] = [J(16, s, iters=FT_ITERS, proto=1.0, proto_src=src + '+dlnh', methods=['cdim'])
                          for src in ('dino+ctr+qf', 'dino+cp+qf', 'dino+cp+ctr', 'dino') for s in (0, 1, 2)]
KSHOT_STAGES['R_tta'] = [dict(j, proto_src=OURS_HEAD) for j in KSHOT_STAGES['O_tta']]
KSHOT_STAGES['R_curve_a'] = [J(k, s, iters=FT_ITERS, proto=1.0, proto_src=OURS_HEAD, methods=['cdim'])
                             for k in (625, 157, 79, 8) for s in _curve_seeds(k)]
KSHOT_STAGES['R_curve_b'] = [J(k, s, iters=FT_ITERS, proto=1.0, proto_src=OURS_HEAD, methods=['cdim'])
                             for k in (4, 2, 1) for s in _curve_seeds(k)]
# 단순한 head 후보 (2026-11-24 사용자 "일단 바꿔보고, 결과를 보고 괜찮은 걸 하자"): ㉙에서 DINO 적응 아래 새 head의 세부(cp·ctr·qf)가
#   효과 없음(기존 head + dlnh 90.62 vs ours 90.66). ours의 head를 기존 head(CLS 프로토타입만)로 바꾼 버전을 같은 표 전체로 잰다.
#   16장 미세조정 CLIP × CDIM은 R_comp에 이미 있음. T_zs·T_full·T_tta·T_dln·T_curve_a/b는 R_* 와 같은 구성에서 head만 바꾼 것.
SIMPLE_HEAD = 'dino+dlnh'
KSHOT_STAGES['T_zs'] = [J(16, s, iters=0, head_mode='zeroshot', proto=1.0, proto_src=SIMPLE_HEAD, methods=['cdim'])
                        for s in (0, 1, 2)]
KSHOT_STAGES['T_full'] = [J(5000, s, iters=FULL_ITERS, proto=1.0, proto_src=SIMPLE_HEAD, methods=['cdim'])
                          for s in (0, 1, 2)]
KSHOT_STAGES['T_tta'] = [dict(j, proto_src=SIMPLE_HEAD) for j in KSHOT_STAGES['O_tta']]
KSHOT_STAGES['T_dln'] = [J(16, s, iters=FT_ITERS, proto=1.0, proto_src='dino+dln', methods=['cdim']) for s in (0, 1, 2)]
KSHOT_STAGES['T_curve_a'] = [J(k, s, iters=FT_ITERS, proto=1.0, proto_src=SIMPLE_HEAD, methods=['cdim'])
                             for k in (625, 157, 79, 8, 4) for s in _curve_seeds(k)]
KSHOT_STAGES['T_curve_b'] = [J(k, s, iters=FT_ITERS, proto=1.0, proto_src=SIMPLE_HEAD, methods=['cdim'])
                             for k in (2, 1) for s in _curve_seeds(k)]
# ---- 확장 실험: CIFAR-100-C (2026-11-24 사용자 "더 높은 쪽을 채택하고 확장 실험을 돌릴 수 있게") ----
#   DATASET=cifar100으로 실행한다(결과는 experiments/server_v100_pil_c100). 하이퍼파라미터는 CIFAR-10에서 정한 값 그대로(재조정 없음).
#   ours의 head는 pick_ours.py가 ㉚ 결과(CIFAR-10, CDIM 조건 10개의 짝지은 차이 평균)로 골라 ours_choice.txt에 쓴다.
#   파일이 없으면 미리 정한 OURS_HEAD.
def _chosen_head():
    p = f'{PROJECT_DIR}/ours_choice.txt'
    if os.path.exists(p):
        with open(p) as f:
            v = f.readline().strip()
        assert v in (OURS_HEAD, SIMPLE_HEAD), f'ours_choice.txt 첫 줄이 이상함: {v}'
        return v
    return OURS_HEAD


CHOSEN_HEAD = _chosen_head()
_X_ALL = ['source', 'tent', 'eta', 'sar', 'tda', 'batclip', 'cdim']
_ours = lambda k, s, **kw: J(k, s, proto=1.0, proto_src=CHOSEN_HEAD, methods=['cdim'], **kw)
#   GPU 2장으로 나눠 돌리려고(사용자 2026-11-24 "그냥 gpu1까지 써줘") head 선택과 무관한 것(_base)과 선택 뒤에 도는 것(_ours)을 나눴다.
#   X_zs       : 라벨 없음 — zero-shot CLIP × 7개 방법, + CDIM + DINO LN 적응(head 없음)             (Main 위쪽 행)
#   X_base16   : 16장(1,600장) subset 3개 — ② CLIP 미세조정 × Source·CDIM, + DINO LN 적응(head 없음)  (Table A ②, Main)
#   X_basefull : 5만 장 10 epoch 학습 seed 3개 — ① CDIM                                             (Table A ①)
#   X_basetta  : 부록 — 16장 미세조정 × 나머지 TTA 5개
#   X_ours16   : ③ ours, ours − DINO 적응(head만, 2×2용), CLIP 학습 없이 ours (X_base16의 체크포인트 재사용)
#   X_oursfull : 상한 = ① + ours 방식 (X_basefull의 체크포인트 재사용)
#   X_ourstta_a/b : 부록 — ours의 TTA 방법 교체 6개 (subset 0·1 / 2로 나눔, GPU 2장)
KSHOT_STAGES['X_zs'] = [J(16, 0, iters=0, head_mode='zeroshot', methods=_X_ALL),
                        _nohead(0, iters=0, head_mode='zeroshot')]
KSHOT_STAGES['X_base16'] = ([J(16, s, iters=FT_ITERS, methods=['source', 'cdim']) for s in (0, 1, 2)]
                            + [_nohead(s, iters=FT_ITERS) for s in (0, 1, 2)])
KSHOT_STAGES['X_basefull'] = [J(5000, s, iters=FULL_ITERS, methods=['cdim']) for s in (0, 1, 2)]
KSHOT_STAGES['X_basetta'] = [J(16, s, iters=FT_ITERS, methods=['tent', 'eta', 'sar', 'tda', 'batclip']) for s in (0, 1, 2)]
KSHOT_STAGES['X_ours16'] = ([_ours(16, s, iters=FT_ITERS) for s in (0, 1, 2)]
                            + [dict(_ours(16, s, iters=FT_ITERS), proto_src=CHOSEN_HEAD.replace('+dlnh', ''))
                               for s in (0, 1, 2)]
                            + [_ours(16, s, iters=0, head_mode='zeroshot') for s in (0, 1, 2)])
KSHOT_STAGES['X_oursfull'] = [_ours(5000, s, iters=FULL_ITERS) for s in (0, 1, 2)]
_ours_tta = lambda s: dict(_ours(16, s, iters=FT_ITERS), methods=[m for m in _X_ALL if m != 'cdim'])
KSHOT_STAGES['X_ourstta_a'] = [_ours_tta(s) for s in (0, 1)]
KSHOT_STAGES['X_ourstta_b'] = [_ours_tta(2)]
# ㉚의 T_tta(단순한 head의 TTA 방법 교체)도 subset으로 나눔. 단순한 head가 채택됐을 때만 돌린다(run_if_simple.py).
#   T_tta_a(subset 0·1)는 GPU 1에서 이미 끝난 6개 run이 있는 simple_gpu1 파일로, T_tta_b(subset 2)는 simple_gpu0 파일로.
KSHOT_STAGES['T_tta_a'] = [j for j in KSHOT_STAGES['T_tta'] if j['seed'] in (0, 1)]
KSHOT_STAGES['T_tta_b'] = [j for j in KSHOT_STAGES['T_tta'] if j['seed'] == 2]
# ---- 추가 실험 (2026-11-25 사용자 "1,3,2 순서로 하자") ----
#   1. Y_tip_a/b (DATASET=cifar100): 같은 1,600장을 쓰는 few-shot head baseline. CIFAR-10 G_fewshot_tuned와 같은 구성
#      (zero-shot CLIP + Tip 캐시, α·β는 diag_tip.py가 탐색용 4종으로 고른 값) × Source(TTA 없음)·CDIM. _a = CLIP 캐시, _b = DINO 캐시
_tip_job = lambda src, s: J(16, s, iters=0, head_mode='zeroshot', proto=TIP_TUNED[src][0] / CFG['TIP_ALPHA'],
                            proto_src=src if TIP_TUNED[src][1] == CFG['TIP_BETA'] else f'{src}@{TIP_TUNED[src][1]:g}',
                            methods=['source', 'cdim'])
KSHOT_STAGES['Y_tip_a'] = [_tip_job('tip_clip', s) for s in (0, 1, 2)] if TIP_TUNED else []
KSHOT_STAGES['Y_tip_b'] = [_tip_job('tip_dino', s) for s in (0, 1, 2)] if TIP_TUNED else []
#   3·2. Z_* (SEVERITY=3 또는 PROTOCOL=continual, CIFAR-10): Table A의 ①②③과 라벨 없음, 모두 CDIM × 15종.
#      Z_base = 라벨 없음 1회 + ② 16장 미세조정 subset 3개 + ① 5만 장 학습 seed 3개 / Z_ours = ③ ours subset 3개
#      Z_head = ours − DINO 적응(head만, DINO 고정) subset 3개: continual에서 DINO 적응이 긴 스트림에서 버티는지 비교용
KSHOT_STAGES['Z_base'] = ([J(16, 0, iters=0, head_mode='zeroshot', methods=['cdim'])]
                          + [J(16, s, iters=FT_ITERS, methods=['cdim']) for s in (0, 1, 2)]
                          + [J(5000, s, iters=FULL_ITERS, methods=['cdim']) for s in (0, 1, 2)])
KSHOT_STAGES['Z_ours'] = [_ours(16, s, iters=FT_ITERS) for s in (0, 1, 2)]
KSHOT_STAGES['Z_head'] = [dict(_ours(16, s, iters=FT_ITERS), proto_src=CHOSEN_HEAD.replace('+dlnh', '')) for s in (0, 1, 2)]
KSHOT_STAGES['Z_head_a'] = KSHOT_STAGES['Z_head'][:2]     # GPU 2장에 나눠 돌리기 (subset 0·1 / 2)
KSHOT_STAGES['Z_head_b'] = KSHOT_STAGES['Z_head'][2:]
KSHOT_STAGE_CORRS = {st: CFG['CORR_ALL'] for st in ('H_final', 'G_fewshot', 'G_tta', 'I_ext', 'G_fewshot_tuned',
                                                    'G_tta_tip', 'K_full_fill', 'K_full_ep5', 'K_full_ep10',
                                                    'K_full_rep_s1', 'K_full_rep_s2', 'L_p5', 'L_curve_ours',
                                                    'L_curve_base_a', 'L_curve_base_b',
                                                    'M_main', 'M_ft', 'M_fthead', 'M_full',
                                                    'N_head15_a', 'N_head15_b',
                                                    'O_tta', 'O_comp', 'O_full', 'O_curve',
                                                    'P_shot1_a', 'P_shot1_b', 'P_lowk_a', 'P_lowk_b',
                                                    'Q15_dlnh', 'Q15_dln',
                                                    'R_nohead', 'R_zs', 'R_full', 'R_comp', 'R_tta',
                                                    'R_curve_a', 'R_curve_b',
                                                    'T_zs', 'T_full', 'T_tta', 'T_dln',
                                                    'T_curve_a', 'T_curve_b',
                                                    'X_zs', 'X_base16', 'X_basefull', 'X_basetta',
                                                    'X_ours16', 'X_oursfull', 'X_ourstta_a', 'X_ourstta_b',
                                                    'T_tta_a', 'T_tta_b',
                                                    'Y_tip_a', 'Y_tip_b', 'Z_base', 'Z_ours', 'Z_head',
                                                    'Z_head_a', 'Z_head_b')}   # 나머지는 KSHOT_CORRS(4종)
KSHOT_METHODS = ['source', 'batclip', 'cdim']
KSHOT_CORRS = CFG['CORR_SWEEP']
FULL_REF = dict(head_mode='text_init', budget='same_epochs', ratio=1.0)   # k-shot 회복률 기준

CLASSES_C10 = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# CIFAR-100 fine label (torchvision CIFAR100.classes 순서, '_'를 공백으로). core.data가 로드할 때 순서를 확인한다.
CLASSES_C100 = [
    'apple', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak tree', 'orange', 'orchid', 'otter', 'palm tree', 'pear', 'pickup truck', 'pine tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow tree', 'wolf', 'woman', 'worm']
CLASSES = CLASSES_C10 if DATASET == 'cifar10' else CLASSES_C100

# CDIM Table 1 (CIFAR10-C, ViT-B/16, severity 5) — sanity check 기준. CIFAR-100은 논문 값을 맞추지 않으므로 비움(상대 비교 방침)
PAPER_REF = {
    'gaussian_noise': dict(source=37.9, tent=15.4, batclip=61.9, cdim=67.8),
    'defocus_blur':   dict(source=71.7, tent=81.4, batclip=80.1, cdim=83.2),
    'snow':           dict(source=73.9, tent=83.6, batclip=82.5, cdim=86.2),
    'contrast':       dict(source=62.4, tent=80.7, batclip=81.2, cdim=87.0),
} if DATASET == 'cifar10' else {}
