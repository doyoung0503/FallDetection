# FallDetection

Wi-Fi CSI 기반 동작 분류 및 전처리/학습 실험 프로젝트입니다.

## Included

- `scripts/`: CSI 전처리, 시계열 데이터셋 생성, 학습, 시각화 스크립트
- `analysis/`: gap 분석, 시각화 결과, 실험 요약
- `artifacts/`: 학습 결과물과 요약 파일

## Excluded

- `dataset/`: 원본 및 전처리 데이터셋
- `paper/`: 논문 자료
- `자료집/`: 참고 자료
- `.venv/`, `.venv313/`: 로컬 가상환경

## Notes

- CSI는 HT-LTF 기준으로 추출해 사용했습니다.
- 시계열 데이터셋은 10 ms grid 기반 제한적 interpolation 규칙으로 생성했습니다.
- 학습 실험에는 row-level MLP와 sequence 1D CNN, SAM optimizer 비교가 포함됩니다.
