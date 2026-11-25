# 라벨이 적을 때의 다중 파운데이션 모델 TTA: Few-shot DINO head + DINO 적응

**Few-shot labels for multi-foundation-model test-time adaptation (CLIP + DINOv2) on corrupted images**

> 클래스당 16장(160장)의 라벨만으로, 5만 장 전부로 학습한 모델보다 손상 이미지에서 더 정확합니다.
> CIFAR-10-C 90.66% vs 85.50% (+5.16%p), CIFAR-100-C 72.96% vs 63.51% (+9.45%p).

English summary. With only 16 labeled images per class, we attach a training-free DINOv2 prototype head to CLIP + CDIM test-time adaptation and also adapt DINOv2's LayerNorm during TTA. On CIFAR-10-C (severity 5, 15 corruptions) this beats a model trained on all 50k labels by +5.16%p, works with 7 different TTA methods, and scales to CIFAR-100-C (+9.45%p) without retuning.

![model](figures/fig6_model_ours.png)

---

## 핵심 결과

1. **라벨 160장으로 5만 장을 넘습니다.** 클래스당 16장만 쓴 ours가 5만 장 전부로 학습한 모델보다 +5.16%p 높습니다(90.66% vs 85.50%).
2. **라벨은 CLIP 학습보다 DINO 쪽에 쓸 때 훨씬 효과가 큽니다.** 같은 160장을 CLIP 미세조정에 쓰면 +1.91%p, Few-shot DINO head + DINO 적응에 쓰면 +10.75%p입니다(라벨 없는 CDIM 79.81% 기준).
3. **두 파운데이션 모델을 함께 적응시키면 효과가 더해지는 것보다 큽니다.** DINO 적응만 +1.02%p, head만 +4.63%p, 둘 다 +8.93%p로 상호작용이 +3.28%p입니다.
4. **어떤 TTA 방법에 붙여도 5만 장 모델을 넘습니다.** 7개 방법(Source, TENT, ETA, SAR, TDA, BATCLIP, CDIM) 모두 +5.16 ~ +15.22%p입니다.
5. **클래스당 2장(총 20장)부터** 라벨 subset 3개 모두 5만 장 모델을 넘습니다(88.27% vs 85.50%).
6. **클래스 100개(CIFAR-100-C)에서는 효과가 더 큽니다.** CIFAR-10 하이퍼파라미터를 그대로 썼는데 5만 장 모델보다 +9.45%p(72.96% vs 63.51%), 같은 라벨을 쓰는 가장 강한 baseline보다 +9.79%p 높고, 상호작용은 +6.88%p입니다.

---

## 문제 설정

- **분포 이동:** 5만 장으로 학습한 CLIP 모델은 깨끗한 CIFAR-10 test에서 97.43%지만, 손상 이미지(CIFAR-10-C 15종, severity 5)에서는 73.07%로 떨어집니다.
- **Test-Time Adaptation (TTA):** 테스트 이미지가 들어오는 동안 라벨 없이 모델 일부를 고칩니다. online, 배치 200장, 배치마다 예측 후 1 step 업데이트, 손상 종류가 바뀌면 처음 상태로 돌아갑니다(episodic).
- **다중 파운데이션 모델 TTA:** CDIM은 CLIP과 DINOv2를 함께 쓰는 TTA 방법입니다(연구실 원고 단계).
- **이 프로젝트의 질문:** source 라벨이 **클래스당 16장뿐일 때** 다중 파운데이션 모델 TTA는 어떻게 되는가, 그리고 그 적은 라벨을 어디에 써야 하는가.
- **출발점이 된 관찰:** source 모델 22개를 보면 TTA 후 정확도는 source의 clean 정확도로 거의 정해집니다(R² = 0.88). 같은 16장으로 만든 DINOv2 클래스 프로토타입은 clean 96.05%로, 같은 16장으로 미세조정한 CLIP(94.52%)보다 높았습니다.

## 방법

기존 모델은 **16장으로 미세조정한 CLIP + CDIM**입니다. 여기에 두 가지를 더했습니다(그림에서 분홍색).

1. **Few-shot DINO head:** 라벨 이미지의 DINOv2 특징으로 클래스 프로토타입을 만들고, test 특징과의 코사인 유사도를 logit에 더합니다. 학습할 파라미터가 없습니다.
   - 특징: CLS 토큰 ⊕ patch 평균(1536차원), 중심화(프로토타입은 라벨 평균, test는 지금까지 본 test 특징의 누적 평균을 뺌), test 이미지와 좌우반전 이미지의 평균
2. **DINO 적응:** TTA 중 CLIP뿐 아니라 DINOv2의 LayerNorm도 같은 손실·같은 optimizer로 업데이트합니다. 적응한 DINO 특징은 head에만 쓰고, CDIM의 DINO 분기는 원래 DINO를 그대로 씁니다.

**식** (배치 B = 200, C: CLIP 이미지 특징, D: DINOv2 CLS 특징, W: CLIP 분류기, 모두 정규화)

CDIM (기존 방법)
- 교차공분산 투영: $M = C^\top D / B$, $T_D = \mathrm{normalize}_{\text{col}}(M^\top W^\top)$
- 앙상블 logit: $Z = \tfrac{1}{2}\left(\tau_c^{-1}\, C W^\top + \tau_d^{-1}\, D\, T_D\right)$, $\tau_c^{-1} = 100$, $\tau_d^{-1} = 20$
- Gated Info-Max 손실: 에너지 $E = -\mathrm{logsumexp}(Z)$가 가장 높은 5%를 빼고, $L = \frac{1}{|G|}\sum_{i \in G} H(p_i) - H(\bar p)$
- OLA 예측: $\pi \leftarrow 0.9\,\pi + 0.1\,\bar p$, $\hat y = \arg\max(Z - \log \pi)$

Ours (더한 부분)
- $P = \lambda\, \tau_p^{-1}\, q\, P_{\text{proto}}^\top$ ($\lambda = 1$, $\tau_p^{-1} = 50$, $q$: 적응한 DINOv2의 test 특징)
- $Z_{\text{ours}} = Z + \tfrac{1}{2} P$. 손실과 OLA 예측은 $Z_{\text{ours}}$로 계산
- $L$의 gradient로 CLIP 이미지 인코더 LayerNorm과 adapted DINOv2 LayerNorm을 함께 업데이트

| 부분 | 오프라인 (라벨 160장) | TTA 중 (라벨 없음) |
|---|---|---|
| CLIP 이미지 인코더 LayerNorm | 미세조정 | 업데이트 |
| CLIP 분류기 W (텍스트로 초기화) | 미세조정 | 고정 |
| DINOv2 (source, CDIM 분기) | 고정 | 고정 |
| DINOv2 (adapted, head용) | 고정 | **LayerNorm 업데이트** |
| 프로토타입 | 클래스 평균 (gradient 없음) | 고정 |

## 실험 설정

- **모델:** OpenAI CLIP ViT-B/16(프롬프트 `a photo of a {class}.`), DINOv2 ViT-B/14
- **데이터:** source는 CIFAR-10 train에서 클래스당 16장(라벨 subset 3개, seed 0·1·2). target은 CIFAR-10-C severity 5, 15종 × 10,000장. 32×32를 224×224로 PIL bicubic 리사이즈(CLIP 공식 전처리)
- **CLIP 미세조정:** 이미지 LayerNorm + 분류기, AdamW lr 1e-4, batch 128, 782 iteration(16장) / 10 epoch(5만 장)
- **TTA:** online batch 200, AdamW lr 1e-3, 손상마다 초기화. 모든 방법이 같은 프로토콜과 같은 test 순서
- **선택 규칙:** λ, head 구성, DINO 적응 방식은 탐색용 손상 4종(Gaussian, Defocus, Snow, Contrast)으로만 골랐습니다. DINO 적응의 이득은 탐색에 안 쓴 11종에서 +4.32%p로, 탐색용 4종(+4.26%p)과 같습니다.
- **환경:** NVIDIA V100 32GB 1장. 손상 1종(10,000장) TTA에 CDIM 34초 / ours 81초, 메모리 13.8GB / 25.7GB

## 결과 (CDIM, severity 5, 15종 평균, ±는 3회 표준편차)

### Main table: 손상 이미지 정확도 (CIFAR-10-C)

마지막 열은 **ours − 그 방법**입니다(반올림 전 값으로 계산). ±는 라벨 subset 3개(5만 장은 학습 seed 3개)의 표준편차입니다.

| 라벨 | 방법 | Noise | Blur | Weather | Digital | 평균 | **ours 대비** |
|---|---|---|---|---|---|---|---|
| 없음 | CLIP zero-shot (TTA 없음) | 44.69 | 63.53 | 76.53 | 55.79 | 61.16 | +29.49 |
| 없음 | TENT | 24.30 | 64.91 | 85.31 | 63.46 | 61.84 | +28.81 |
| 없음 | ETA | 67.64 | 76.45 | 86.64 | 75.24 | 77.08 | +13.57 |
| 없음 | SAR | 45.19 | 70.04 | 84.51 | 71.29 | 69.26 | +21.39 |
| 없음 | TDA | 45.41 | 65.11 | 76.83 | 57.34 | 62.22 | +28.43 |
| 없음 | BATCLIP | 62.32 | 73.30 | 84.08 | 68.76 | 72.77 | +17.89 |
| 없음 | CDIM | 71.53 | 79.52 | 87.86 | 78.28 | 79.81 | +10.84 |
| 없음 | CDIM + DINO 적응 | 73.80 | 80.89 | 89.01 | 79.86 | 81.36 | +9.29 |
| 160장, TTA 없음 | CLIP 미세조정 | 47.70 | 69.93 | 82.79 | 61.75 | 66.73 ±0.98 | +23.93 |
| 160장, TTA 없음 | Tip-Adapter | 42.86 | 67.22 | 79.67 | 58.76 | 63.41 ±0.24 | +27.25 |
| 160장, TTA 없음 | DINO 캐시 (Tip 방식) | 69.72 | 78.98 | 87.92 | 78.46 | 79.37 ±0.46 | +11.28 |
| 160장 + TTA | CDIM + CLIP 미세조정 (②) | 73.29 | 81.72 | 89.61 | 80.16 | 81.72 ±0.20 | +8.93 |
| 160장 + TTA | CDIM + CLIP 미세조정 + DINO 적응 | 74.65 | 82.63 | 90.41 | 81.26 | 82.74 ±0.23 | +7.91 |
| 160장 + TTA | CDIM + Tip-Adapter | 72.35 | 80.55 | 88.65 | 79.20 | 80.71 ±0.07 | +9.95 |
| 160장 + TTA | CDIM + DINO 캐시 (Tip 방식) | 78.47 | 85.14 | 91.82 | 85.19 | 85.60 ±0.11 | +5.05 |
| **160장 + TTA** | **ours** | **87.16** | **90.30** | **94.38** | **89.91** | **90.66 ±0.26** | – |
| 50,000장 (참고) | CDIM + CLIP 미세조정 (①) | 77.29 | 85.96 | 92.49 | 84.20 | 85.50 ±0.01 | +5.16 |

- ours는 모든 baseline보다 높고, 손상 유형 4개 모두에서 5만 장 모델(①)보다 높습니다. 라벨 subset 3개 모두, 손상 15종 모두에서도 ①보다 높습니다.
- Tip 계열의 α·β는 ours의 λ와 같은 규칙(탐색용 4종, TTA 없음)으로 골랐습니다.

### 라벨을 줄이면 얼마나 떨어지고, 각 모듈로 얼마나 오르나 (CIFAR-10-C, CDIM)

| 단계 | Noise | Blur | Weather | Digital | 평균 | 바로 앞 대비 |
|---|---|---|---|---|---|---|
| ① CLIP을 5만 장으로 미세조정 | 77.29 | 85.96 | 92.49 | 84.20 | 85.50 | – |
| ② 라벨을 클래스당 16장(0.32%)으로 줄임 | 73.29 | 81.72 | 89.61 | 80.16 | 81.72 | **−3.77** |
| ② + Few-shot DINO head | 79.94 | 85.86 | 92.03 | 85.98 | 86.35 | **+4.63** |
| ② + Few-shot DINO head + DINO 적응 = **ours** | **87.16** | **90.30** | **94.38** | **89.91** | **90.66** | **+4.30** |
| ours − ① | +9.87 | +4.34 | +1.89 | +5.71 | **+5.16** | |

- CLIP 미세조정 없이(zero-shot CLIP) ours를 써도 90.56%로 거의 같습니다. 라벨의 효과는 거의 모두 DINO 쪽에서 옵니다.
- 같은 방식을 5만 장에 쓴 상한은 91.51%로, 16장 ours와 0.85%p 차이입니다.

### 두 모듈은 서로를 키운다 (2×2, ② 81.72 기준)

| | DINO 고정 | DINO 적응 |
|---|---|---|
| head 없음 | 81.72 (②) | 82.74 (+1.02) |
| Few-shot DINO head | 86.35 (+4.63) | **90.66 (+8.93)** |

- 따로 더하면 +1.02 + 4.63 = +5.65인데, 함께 쓰면 +8.93으로 **상호작용이 +3.28%p**입니다.
- 순서를 바꿔 DINO 적응부터 더하면 +1.02, 그다음 head가 +7.91입니다. head는 DINO 특징 공간에서 프로토타입과 비교하므로, 손상으로 틀어진 DINO 특징을 test 분포에 맞추면 head가 다시 잘 작동합니다. head가 없으면 같은 DINO 적응도 효과가 작습니다.

### 다른 TTA 방법에 붙이기 (CIFAR-10-C)

| TTA 방법 | ① 5만 장 | ② 16장 | ③ ② + ours | ③ − ① |
|---|---|---|---|---|
| Source (TTA 없음) | 73.07 | 66.73 | 82.05 | +8.98 |
| TENT | 71.62 | 68.12 | 86.84 | +15.22 |
| ETA | 84.02 | 80.25 | 89.43 | +5.41 |
| SAR | 75.29 | 71.75 | 82.48 | +7.19 |
| TDA | 73.28 | 67.19 | 82.10 | +8.82 |
| BATCLIP | 80.86 | 76.57 | 88.50 | +7.64 |
| **CDIM** | **85.50** | **81.72** | **90.66** | **+5.16** |

### 라벨 수 곡선 (CIFAR-10-C)

![label curve](figures/fig7_label_curve.png)

| 클래스당 라벨 | 1 | 2 | 4 | 8 | **16** | 79 | 157 | 625 | 5000 |
|---|---|---|---|---|---|---|---|---|---|
| baseline (CLIP 미세조정 + CDIM) | 79.91 | 80.45 | 80.73 | 81.49 | **81.72** | 82.63 | 83.05 | 83.58 | 85.50 |
| **ours** | 84.76 | 88.27 | 89.17 | 90.16 | **90.66** | 91.09 | 91.15 | 91.23 | 91.51 |

### CIFAR-100-C (클래스 100개, 하이퍼파라미터 재조정 없음)

![datasets](figures/fig8_table_a_datasets.png)

| 조건 | CIFAR-10-C | CIFAR-100-C |
|---|---|---|
| 라벨 없음 (zero-shot CLIP) | 79.81 | 51.80 |
| ① 5만 장 | 85.50 | 63.51 |
| ② 16장 (CIFAR-100은 1,600장) | 81.72 | 56.29 |
| **③ ② + ours** | **90.66** | **72.96** |
| **③ − ①** | **+5.16** | **+9.45** |

**CIFAR-100-C Main table** (라벨 160장 대신 1,600장, 마지막 열 = ours − 그 방법)

| 라벨 | 방법 | Noise | Blur | Weather | Digital | 평균 | **ours 대비** |
|---|---|---|---|---|---|---|---|
| 없음 | CLIP zero-shot (TTA 없음) | 22.09 | 38.39 | 49.23 | 30.04 | 35.79 | +37.17 |
| 없음 | TENT | 7.86 | 41.72 | 51.92 | 43.23 | 38.07 | +34.89 |
| 없음 | ETA | 35.93 | 46.43 | 57.59 | 44.61 | 46.82 | +26.14 |
| 없음 | SAR | 29.28 | 46.08 | 56.10 | 44.43 | 44.95 | +28.01 |
| 없음 | TDA | 23.99 | 39.46 | 49.86 | 31.11 | 36.91 | +36.05 |
| 없음 | BATCLIP | 20.94 | 42.98 | 51.70 | 35.81 | 38.99 | +33.98 |
| 없음 | CDIM | 41.36 | 52.52 | 61.32 | 49.39 | 51.80 | +21.16 |
| 없음 | CDIM + DINO 적응 | 42.48 | 53.34 | 62.08 | 50.27 | 52.68 | +20.28 |
| 1,600장, TTA 없음 | CLIP 미세조정 | 26.68 | 44.96 | 57.66 | 33.45 | 41.62 ±0.31 | +31.34 |
| 1,600장, TTA 없음 | Tip-Adapter | 24.47 | 39.22 | 50.36 | 30.79 | 36.99 ±0.24 | +35.97 |
| 1,600장, TTA 없음 | DINO 캐시 (Tip 방식) | 44.54 | 53.83 | 67.19 | 53.02 | 55.32 ±0.33 | +17.64 |
| 1,600장 + TTA | CDIM + CLIP 미세조정 (②) | 45.56 | 56.93 | 65.95 | 54.06 | 56.29 ±0.10 | +16.67 |
| 1,600장 + TTA | CDIM + CLIP 미세조정 + DINO 적응 | 46.14 | 57.49 | 66.48 | 54.63 | 56.85 ±0.07 | +16.11 |
| 1,600장 + TTA | CDIM + Tip-Adapter | 42.32 | 53.58 | 62.55 | 50.68 | 52.95 ±0.04 | +20.01 |
| 1,600장 + TTA | CDIM + DINO 캐시 (Tip 방식) | 53.58 | 62.50 | 72.75 | 61.46 | 63.17 ±0.10 | +9.79 |
| **1,600장 + TTA** | **ours** | **67.36** | **72.12** | **79.60** | **71.37** | **72.96 ±0.09** | – |
| 50,000장 (참고) | CDIM + CLIP 미세조정 (①) | 52.18 | 64.49 | 72.78 | 61.75 | 63.51 ±0.01 | +9.45 |

- 같은 1,600장을 쓰는 가장 강한 baseline(CDIM + DINO 캐시)은 5만 장 모델에 못 미치지만(63.17 vs 63.51), ours는 5만 장 모델보다 +9.45%p 높습니다.
- Tip 계열의 α·β는 CIFAR-100에서 같은 규칙으로 다시 골랐고(Tip-Adapter α 1, β 3 / DINO 캐시 α 10, β 3), ours는 CIFAR-10 값을 그대로 썼습니다.

| 단계 (CIFAR-100-C, CDIM) | Noise | Blur | Weather | Digital | 평균 | 바로 앞 대비 |
|---|---|---|---|---|---|---|
| ① CLIP을 5만 장으로 미세조정 | 52.18 | 64.49 | 72.78 | 61.75 | 63.51 | – |
| ② 라벨을 클래스당 16장(3.2%)으로 줄임 | 45.56 | 56.93 | 65.95 | 54.06 | 56.29 | **−7.21** |
| ② + Few-shot DINO head | 57.49 | 64.36 | 74.30 | 63.93 | 65.52 | **+9.23** |
| ② + Few-shot DINO head + DINO 적응 = **ours** | **67.36** | **72.12** | **79.60** | **71.37** | **72.96** | **+7.44** |
| ours − ① | +15.18 | +7.63 | +6.82 | +9.62 | **+9.45** | |

2×2 (② 기준): DINO 적응만 +0.56, head만 +9.23, 둘 다 +16.67, 상호작용 +6.88.

더 자세한 표는 [`results/`](results/)에 있습니다(`analyze.py`, `sum_c100.py`가 만든 표 그대로).

---

## 코드 구성

```
config.py          실험 조건과 stage(실행 단위) 정의. 조건은 여기서만 바꿈
main.py            실행: check / sanity / kshot --stage <이름>
core/
  data.py          CIFAR-10(-C)·CIFAR-100(-C) 로드, 라벨 subset 추출, test 스트림 순서
  models.py        CLIP 분류기(텍스트 초기화), DINOv2 로드
  source.py        CLIP 미세조정(LayerNorm + 분류기)과 체크포인트
  methods.py       TTA 방법(Source, TENT, ETA, SAR, TDA, BATCLIP, CDIM), Few-shot DINO head, DINO 적응
  runner.py        run 실행과 결과 CSV 기록(끊겨도 이어서 실행), continual 프로토콜
  runtime.py       GPU·AMP 설정
analyze.py         CIFAR-10 표와 라벨 수 곡선(fig7)
sum_c100.py        CIFAR-100 표와 fig8
sum_ext.py         추가 실험(severity 3, continual) 표
fig_model.py       모델 구조도(fig6, 실제 예측 포함)
make_figs.py       analyze.py + fig_model.py
make_data_figures.py  데이터 그림(fig1~fig5)
prepare_data.py    데이터·가중치 다운로드
diag_tip.py        Tip 계열 baseline의 α·β 선택 (diag_dino.py 함수 사용)
pick_ours.py       head 구성 선택 기록
smoke_cont.py      continual 경로 점검
```

## 실행 방법

**환경** (V100 기준. PyTorch는 CUDA 빌드에 맞춰 설치)
```bash
conda create -n cdim python=3.10 -y && conda activate cdim
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install --no-deps https://github.com/openai/CLIP/archive/refs/heads/main.zip
```

**데이터·가중치** (`~/cdim_data`에 저장, 약 4GB + CIFAR-100-C)
```bash
python prepare_data.py                      # CIFAR-10, CIFAR-10-C, CLIP, DINOv2
python prepare_data.py --dataset cifar100   # CIFAR-100, CIFAR-100-C
```

**실행** (GPU 1장만 보이게 해야 실행됩니다)
```bash
export CUDA_VISIBLE_DEVICES=0
python main.py check                         # 환경 점검 (zero-shot 약 90%면 정상)
python main.py kshot --stage Q15_dlnh        # 예: ③ ours, 15종, subset 3개
python main.py kshot --stage Q15_dlnh --plan # 남은 run과 예상 시간만 출력
python analyze.py                            # 표·그림 (GPU 불필요)
```
- 결과는 run마다 `experiments/<EXP_NAME>/results_kshot*.csv`에 한 줄씩 쌓입니다. 같은 설정은 어느 stage에서 돌았든 한 번만 돕니다.
- CIFAR-100은 `DATASET=cifar100`, 추가 실험은 `SEVERITY=3` 또는 `PROTOCOL=continual` 환경 변수로 고릅니다. 결과 폴더가 따로 생깁니다.

**주요 표의 stage** (CIFAR-10. stage 정의는 `config.py`)

| 표의 행 | stage |
|---|---|
| ① 5만 장 (10 epoch, 학습 seed 3개) | `K_full_ep10`, `K_full_rep_s1`, `K_full_rep_s2` |
| ② 16장 CLIP 미세조정 + CDIM | `L_p5` |
| ③ ours | `Q15_dlnh` |
| head만 / DINO 적응만 / CLIP 학습 없이 ours / 상한 | `N_head15_b` / `R_nohead` / `R_zs` / `R_full` |
| 라벨 수 곡선 (baseline / ours) | `L_curve_base_a`, `L_curve_base_b` / `R_curve_a`, `R_curve_b` |
| 다른 TTA 방법 (② / ③) | `M_ft` / `R_tta` |
| Tip-Adapter·DINO 캐시 | `diag_tip.py` → `G_fewshot_tuned`, `G_tta_tip` |
| CIFAR-100-C (`DATASET=cifar100`) | `X_zs`, `X_base16`, `X_basefull`, `X_basetta`, `X_ours16`, `X_oursfull`, `X_ourstta_a`, `X_ourstta_b`, `diag_tip.py` → `Y_tip_a`, `Y_tip_b` |

## 주의

- 모든 비교는 같은 구현·같은 프로토콜 안의 상대 비교입니다.
- CDIM은 연구실 원고 단계의 방법입니다(manuscript in preparation). 이 저장소의 CDIM 수치는 모두 직접 돌린 값입니다. 우리 기여는 CDIM 위에 더한 Few-shot DINO head와 DINO 적응입니다.
- ETA·SAR·TDA는 같은 프로토콜에 맞춘 재구현입니다. ETA는 Fisher 정규화가 없고, SAR은 AdamW를 쓰고, TDA는 positive cache만 씁니다. SAR은 이 설정에서 붕괴 감지로 거의 매 배치 초기화돼 사실상 TTA가 없는 값입니다.
- 클래스당 1장은 라벨 subset에 따라 크게 흔들립니다(84.76 ±2.43).
- ours는 TTA 시간이 CDIM의 약 2.3배, 메모리가 약 1.9배입니다.

## 참고문헌

- CLIP: Radford et al., ICML 2021 / DINOv2: Oquab et al., 2023
- CIFAR-10-C: Hendrycks & Dietterich, ICLR 2019
- TENT: Wang et al., ICLR 2021 / EATA·ETA: Niu et al., ICML 2022 / SAR: Niu et al., ICLR 2023 / TDA: Karmanov et al., CVPR 2024 / BATCLIP
- Tip-Adapter: Zhang et al., ECCV 2022 / CaFo: Zhang et al., CVPR 2023 / Prototypical Networks: Snell et al., NeurIPS 2017
- CDIM: manuscript in preparation
