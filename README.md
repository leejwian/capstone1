Capstone Model Compression
==========================

두 단계 파이프라인: 학습 → 압축 → 벤치마크
- `train.py`: CIFAR-10/100용 ResNet/ViT 학습 및 체크포인트 저장.
- `compress.py`: 비구조적 가지치기 + 선택적 양자화(Linear 동적)로 압축, 압축 강도 레벨 1~5 지원.
- `benchmark.py`: 원본 vs 압축 모델을 같은 설정으로 성능/정확도/파일크기 비교, 레벨 반복 지원.

폴더 구조
----------------
- `checkpoints/`: 학습된 원본 모델 (예: `resnet18_cifar10.pth`, `vit_b_16_cifar10.pth`).
- `compressed_models/`: 압축 결과 (`compressed_{model}_l{level}.pt`).
- `data/`: CIFAR 데이터 자동 다운로드 위치.
- `benchmark_levels.csv`: 압축 레벨별 벤치마크 요약(현재 결과 반영).

의존성 설치
-----------
```
pip install -r requirements.txt
```
(torch 2.2.2 / torchvision 0.17.2, numpy<2)

학습 (원본 모델 생성)
---------------------
예) ResNet18 CIFAR-10:
```
python train.py --model resnet18 --dataset cifar10 --data-root data \
  --epochs 25 --batch-size 256 --output checkpoints/resnet18_cifar10.pth
```
예) ViT-B/16 CIFAR-10:
```
python train.py --model vit_b_16 --dataset cifar10 --data-root data \
  --epochs 30 --batch-size 128 --output checkpoints/vit_b_16_cifar10.pth
```

압축 (레벨별 생성)
------------------
레벨이 높을수록 가지치기 강도↑, 일부 레벨에서 양자화 적용(ResNet 3+, ViT 4+).
예) ResNet18 레벨 1~5:
```
for L in 1 2 3 4 5; do
  python compress.py --model resnet18 --checkpoint checkpoints/resnet18_cifar10.pth \
    --num-classes 10 --compression-level $L --output compressed_models/compressed_resnet18_l${L}.pt
done
```
예) ViT-B/16 레벨 1~5:
```
for L in 1 2 3 4 5; do
  python compress.py --model vit_b_16 --checkpoint checkpoints/vit_b_16_cifar10.pth \
    --num-classes 10 --compression-level $L --output compressed_models/compressed_vit_b_16_l${L}.pt
done
```

벤치마크
--------
원본 vs 압축을 동일 설정으로 비교. 레벨 반복:
```
python benchmark.py --levels 1 --levels 2 --levels 3 --levels 4 --levels 5 \
  --max-eval-samples 500
```
출력: params/nonzero/sparsity/latency/throughput/acc/file_size 모두 원본·압축 나란히 표시. 파일 경로 패턴은 기본값(`compressed_{model}_l{level}.pt`) 사용.

현재 벤치마크 인사이트 (500 샘플 기준)
------------------------------------
- ResNet: 레벨 1~2는 정확도 유지·GPU 지연 유지/소폭 개선. 레벨 3~5는 양자화로 CPU 실행 → 지연 크게 증가, 정확도 소폭 하락.
- ViT: 레벨 1~3은 정확도 유지·지연 큰 변화 없음. 레벨 4~5는 양자화로 CPU 실행 → 지연 크게 증가, 정확도 약간 하락, 파일 크기 절반 수준.
- 자세한 수치는 `benchmark_levels.csv` 참고.

한계와 확장
-----------
- 현재 프루닝은 텐서 shape를 유지하는 희소화라 파일 크기 감소가 제한적. 채널 제거 기반 구조적 프루닝과 정적 양자화(QAT/FX)로 shape/파일 크기 축소가 필요.
- 필요 시 `torch-pruning` 기반 구조적 프루닝 또는 FX 정적 양자화 파이프라인을 추가하여 채널/비트폭을 줄이는 방향으로 확장 가능.
