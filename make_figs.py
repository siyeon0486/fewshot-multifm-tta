"""채택된 ours(config.CHOSEN_HEAD)로 CIFAR-10 표·그림을 다시 만든다 (2026-11-24 사용자 "채택된 결과에 따라 figure도 새로 생산").
㉛ 대기열에서 pick_ours.py 바로 다음에 GPU 0으로 실행한다 (대기열이 DATASET=cifar100이어도 여기서는 cifar10으로 고정).
  1) analyze.py: 모든 표 + fig7 라벨 수 곡선 (ours = CHOSEN_HEAD)
  2) fig_model.py --recompute: fig6 모델 그림 (CHOSEN_HEAD로 online TTA를 끝까지 돌린 실제 예측, 약 6분)
  3) 두 그림과 예측 기록을 ~/cdim_tta/figures/로 복사 (로컬 figures/와 같은 이름)"""
import os, sys, shutil, subprocess
here = os.path.dirname(os.path.abspath(__file__))
os.environ['DATASET'] = 'cifar10'
rc = 0
for cmd in (['analyze.py'], ['fig_model.py', '--recompute']):
    r = subprocess.run([sys.executable] + cmd, cwd=here, env=dict(os.environ))
    print(f"== {' '.join(cmd)} → exit {r.returncode}", flush=True)
    rc = rc or r.returncode

sys.path.insert(0, here)
from config import CFG, CHOSEN_HEAD
src = f"{CFG['EXP_ROOT']}/{CFG['EXP_NAME']}/analysis/figures"
dst = f'{here}/figures'
os.makedirs(dst, exist_ok=True)
for f in ('fig6_model_ours.png', 'fig7_label_curve.png', 'model_fig_preds.json'):
    if os.path.exists(f'{src}/{f}'):
        shutil.copy2(f'{src}/{f}', f'{dst}/{f}')
        print('복사:', f'{dst}/{f}')
print('ours head:', CHOSEN_HEAD)
sys.exit(rc)
