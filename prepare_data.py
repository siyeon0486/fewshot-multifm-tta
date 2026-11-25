"""데이터·가중치 다운로드 (GPU 불필요 → master에서 실행 가능, 홈이 공유돼 GPU 노드에서도 그대로 보임).

  python prepare_data.py                              # 전부 인터넷에서 받기
  python prepare_data.py --c10c-tar ~/CIFAR-10-C.tar  # Colab에서 쓰던 tar를 scp로 올렸다면 그걸 사용
  python prepare_data.py --dataset cifar100           # 확장 실험(2026-11-24): CIFAR-100 + CIFAR-100-C만 받기

받는 것: CIFAR-10 (~170MB), CIFAR-10-C (~2.9GB tar, 풀면 ~18GB), CLIP ViT-B/16 (~340MB), DINOv2 ViT-B/14 (~330MB)
         --dataset cifar100: CIFAR-100 (~170MB), CIFAR-100-C (~2.9GB tar, Zenodo 3555552)
"""
import os, sys, hashlib, subprocess, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CFG

C10C_URLS = ['https://zenodo.org/records/2535967/files/CIFAR-10-C.tar',
             'https://huggingface.co/datasets/torch-uncertainty/CIFAR-C/resolve/main/CIFAR-10-C.tar']
C10C_MD5 = '56bf5dcef84df0e2308c6dcbcbbd8499'
C100C_URLS = ['https://zenodo.org/records/3555552/files/CIFAR-100-C.tar']
C100C_MD5 = '11f0ed0f1191edbf9fa23466ae6021d3'       # Zenodo API의 checksum (2026-11-24 확인)


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def get_c10c(data_dir, tar):
    if glob.glob(f'{data_dir}/**/CIFAR-10-C/labels.npy', recursive=True):
        print('✓ CIFAR-10-C 이미 있음'); return
    if tar is None:
        tar = f'{data_dir}/CIFAR-10-C.tar'
        if not os.path.exists(tar):
            for url in C10C_URLS:
                print('CIFAR-10-C 다운로드:', url)
                r = subprocess.run(['wget', '-c', '-O', tar, url])
                if r.returncode == 0: break
            else:
                sys.exit('❌ 다운로드 실패. Colab에서 쓰던 tar를 scp로 올린 뒤 --c10c-tar 로 지정하세요.')
    print('md5 확인 중...')
    got = md5(tar)
    if got != C10C_MD5:
        sys.exit(f'❌ md5 불일치 ({got}). 파일이 손상됐습니다. 지우고 다시 받으세요: rm {tar}')
    print('압축 해제 중 (몇 분 걸림)...')
    subprocess.run(['tar', '-xf', tar, '-C', data_dir], check=True)
    print('✓ CIFAR-10-C 준비 완료. 공간이 필요하면 tar는 지워도 됩니다:', tar)


def get_c100c(data_dir):
    if glob.glob(f'{data_dir}/**/CIFAR-100-C/labels.npy', recursive=True):
        print('✓ CIFAR-100-C 이미 있음'); return
    tar = f'{data_dir}/CIFAR-100-C.tar'
    if not os.path.exists(tar) or md5(tar) != C100C_MD5:
        for url in C100C_URLS:
            print('CIFAR-100-C 다운로드:', url)
            r = subprocess.run(['wget', '-c', '-q', '-O', tar, url])
            if r.returncode == 0: break
        else:
            sys.exit('❌ CIFAR-100-C 다운로드 실패')
    print('md5 확인 중...')
    got = md5(tar)
    if got != C100C_MD5:
        sys.exit(f'❌ md5 불일치 ({got}). 파일이 손상됐습니다. 지우고 다시 받으세요: rm {tar}')
    print('압축 해제 중 (몇 분 걸림)...')
    subprocess.run(['tar', '-xf', tar, '-C', data_dir], check=True)
    print('✓ CIFAR-100-C 준비 완료. 공간이 필요하면 tar는 지워도 됩니다:', tar)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--c10c-tar', default=None)
    ap.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    a = ap.parse_args()
    d = CFG['DATA_DIR']; os.makedirs(d, exist_ok=True)

    import torch, torchvision
    torch.set_num_threads(4)
    if a.dataset == 'cifar100':          # CLIP·DINOv2는 CIFAR-10 준비 때 이미 받았음
        torchvision.datasets.CIFAR100(f'{d}/cifar100', train=True, download=True)
        torchvision.datasets.CIFAR100(f'{d}/cifar100', train=False, download=True)
        print('✓ CIFAR-100')
        get_c100c(d)
        return
    torchvision.datasets.CIFAR10(f'{d}/cifar10', train=True, download=True)
    torchvision.datasets.CIFAR10(f'{d}/cifar10', train=False, download=True)
    print('✓ CIFAR-10')

    get_c10c(d, os.path.expanduser(a.c10c_tar) if a.c10c_tar else None)

    from core.models import model_dirs
    clip_dir, _ = model_dirs()
    import clip
    clip.load('ViT-B/16', device='cpu', jit=False, download_root=clip_dir)
    print('✓ CLIP ViT-B/16 →', clip_dir)
    torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', trust_repo=True)
    print('✓ DINOv2 ViT-B/14 →', torch.hub.get_dir())
    print('\n모두 준비됐습니다. GPU 노드에서 `python main.py check`를 실행하세요.')


if __name__ == '__main__':
    main()
