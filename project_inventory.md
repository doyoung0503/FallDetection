# Project Inventory

이 문서는 현재 `/Users/doyoung/Documents/FallDetection` 워크스페이스를 **코드/데이터셋/분석/학습 결과 관점에서 전수 정리**한 문서다.  
목표는 이 파일 하나만 읽어도:

1. 프로젝트가 어떤 문제를 다루는지  
2. 어떤 코드가 어떤 역할을 하는지  
3. 어떤 디렉터리가 원본/전처리/결과물인지  
4. 현재 로컬에서 어떤 수정이 추가되었는지  
5. 무엇을 실행하면 어떤 산출물이 생기는지  

를 한 번에 이해할 수 있게 하는 것이다.

---

## 1. 저장소 전체 목적

이 프로젝트는 **Wi-Fi CSI(Channel State Information)** 를 이용한 사람 상태/동작 분류 실험 저장소다.  
현재까지 다룬 문제는 크게 세 종류다.

1. `none / occupy / walk`
   - 사람이 없음 / 정적으로 존재 / 걷는 상태
2. `large / normal / small`
   - stride가 다른 3-class 문제
3. `big / small` (`dataset/esp32`)
   - 사람 기준 split이 핵심인 binary stride classification

이 저장소는 단순 모델 코드 모음이 아니라,
- raw CSI 해석
- LTF 분리
- 시간축 재구성
- 시계열 feature 설계
- XFall/SDP 재현
- 각 전처리/모델 ablation 결과

를 함께 쌓아온 **실험형 저장소**다.

---

## 2. 현재 Git / 로컬 변경 상태

현재 확인된 Git 이력:

- `d6391a7` `Initial project snapshot`
- `1a52d7a` `Rewrite README as experiment report`
- `37c7bee` `Add XFall SDP preprocessing pipelines`

현재 로컬 상태의 중요한 특징:

- 이미 원격 저장소에 올라간 기본 실험 코드 위에
- `esp32` 데이터셋용 스크립트/분석/아티팩트가 **대량으로 로컬 추가**되어 있음
- 현재 `git status` 기준으로:
  - 새 스크립트 다수
  - 새 `analysis/`, `artifacts/`, `dataset/` 하위 산출물 다수
  - `experience.md` 추가
  - `scripts/train_sequence_cnn_torch.py` 수정

즉, 현재 워크스페이스는 **커밋된 기본 저장소 + 아직 정리되지 않은 로컬 실험 결과물**의 상태다.

---

## 3. 루트 파일 설명

루트 파일은 현재 많지 않다.

### 3.1 `.gitignore`

- Git에 포함하지 않을 파일/폴더를 정의한다.
- 특히 대용량 `dataset/`, `paper/`, `자료집/`, 가상환경 등을 제외하는 용도로 중요하다.

### 3.2 `README.md`

- 공개 저장소용 보고서형 개요 문서다.
- 실험 배경, 주요 결과, 시각화 예시를 담고 있다.
- 외부 독자에게 “이 프로젝트가 뭘 했는지”를 보여주는 문서다.

### 3.3 `experience.md`

- 이번 대화 흐름에서 만든 **실험 로그 요약 문서**다.
- 어떤 실험을 했고 어떤 수치가 나왔는지 시간 순으로 요약되어 있다.
- 실험 결과 중심이다.

### 3.4 `project_inventory.md`

- 지금 읽고 있는 문서다.
- 파일/디렉터리/코드 역할 자체를 정리하는 인벤토리 문서다.

---

## 4. 최상위 디렉터리 역할

### 4.1 `scripts/`

프로젝트의 핵심 코드가 모여 있다.

- 전처리 스크립트
- 데이터셋 생성 스크립트
- 학습 스크립트
- 3-seed 반복 실행 스크립트
- 시각화/분석 스크립트
- SAM optimizer 구현

이 저장소의 “실행 가능한 로직” 대부분은 여기에 있다.

### 4.2 `dataset/`

원본 데이터, 전처리 데이터, 윈도우 데이터셋이 모여 있다.

구조적으로는 크게:
- 원본 raw CSV
- HT-LTF 추출 데이터
- 시계열 윈도우 데이터
- XFall/SDP 데이터셋

로 나뉜다.

### 4.3 `artifacts/`

학습 결과와 집계 요약이 모인다.

예:
- `training_summary.json`
- `aggregate_summary.json`
- best checkpoint
- 시드별 실험 결과 디렉터리

실험 성능을 다시 확인할 때 가장 먼저 보는 위치다.

### 4.4 `analysis/`

시각화 및 탐색적 분석 결과가 저장된다.

예:
- sampling gap 분포
- amplitude/phase 시각화
- class separation plot
- curve figure

### 4.5 `paper/`

- 참고 논문 PDF를 보관하는 위치다.
- 특히 XFall 논문 재현 실험의 근거가 되는 자료가 여기에 있다.

### 4.6 `자료집/`

- 별도 참고 자료 모음 폴더다.
- 코드보다 문서/자료 보관 성격이 강하다.

### 4.7 `.venv`, `.venv313`

- 로컬 Python 가상환경이다.
- `.venv313`은 특히 PyTorch/MPS 실험에서 사용됐다.

---

## 5. `scripts/` 전수 조사

아래는 현재 `scripts/` 안의 파일과 역할이다.

### 5.1 Raw CSI 전처리 계열

#### `scripts/preprocess_raw_htltf.py`

역할:
- 클래스 폴더 구조의 raw ESP32 CSI CSV에서 **유효 HT-LTF subcarrier만 추출**
- `384 ints -> 228 ints`

사용처:
- `dataset/raw`, `dataset/raw_stride`류 데이터셋 전처리

출력:
- `dataset/preprocessed_raw`
- `dataset/preprocessed_raw_stride`

---

#### `scripts/extract_esp32_raw_csi_variants.py`

역할:
- `dataset/esp32` 파일명에서 `person`, `big/small`을 파싱
- `LLTF only`, `HT-LTF only`, `LLTF+HT-LTF` 3종 raw variant 생성
- 사람 기준 `train/validation` split 생성

입력:
- `dataset/esp32`

출력:
- `dataset/esp32_raw_csi_variants`

중요:
- 현재 `esp32` 실험의 raw 입력 기준점이다.

---

### 5.2 일반 시계열 데이터셋 생성 계열

#### `scripts/build_resampled_sequence_dataset.py`

역할:
- 초기 `none / occupy / walk`, `raw_stride` 실험용
- 고정 grid(`10ms`)와 제한적 선형 보간으로 amplitude 시계열 윈도우 생성

출력 예:
- `dataset/sequence_10ms_amp_mask`
- `dataset/sequence_10ms_amp_mask_stride10`
- `dataset/sequence_10ms_amp_mask_stride20`
- `dataset/sequence_10ms_amp_mask_raw_stride_tol4000`

---

#### `scripts/build_esp32_sequence_variants.py`

역할:
- `esp32`용 기본 시계열 variant 생성
- 동일 amplitude 기반 sequence에 대해
  - `interp_only`
  - `interp_mask`
  - `interp_mask_deltat`
  를 만든다.

입력:
- 보통 `dataset/esp32_raw_csi_variants/htltf_only`

출력:
- `dataset/esp32_sequence_htltf_variants_*`

의미:
- feature extraction 이전의 **기본 sequence preprocessing baseline**

---

#### `scripts/build_esp32_sequence_variants_adaptive.py`

역할:
- adaptive gap reconstruction 실험용
- `initial median + EMA + k배수 판정 + long-gap split` 방식으로 시계열 복원

핵심 로직:
- 파일 내부 초기 positive gap median으로 base gap 시작
- accepted 1-step gap만 EMA 업데이트
- `k=1..K` 배수 여부를 상대/절대 tolerance로 판정
- 짧은 손실만 보간, 긴 gap은 segment 분리

출력:
- `dataset/esp32_sequence_htltf_variants_w64_s10_adaptive_*`

의미:
- 고정 `11ms`/`11017us` grid 대신 **로컬 적응형 시간축**을 시험한 스크립트

---

### 5.3 amplitude-derived feature 데이터셋 생성

#### `scripts/build_esp32_sequence_ma10_dataset.py`

역할:
- 현재 `esp32` 실험에서 가장 중심적인 feature builder
- 고정 grid + 보간 후 amplitude 시퀀스에서 파생 feature를 만든다.

지원 feature:
- `raw_amplitude`
- `ma_residual`
- `first_difference`
- `rolling_std`

지원 raw preprocessing option:
- `grid_us`
- `grid_tolerance_us`
- `max_interp_gap_steps`
- `interp_mode = linear / forward_fill / nearest`
- `packet_filter = none / amp_mad3 / amp_mad4 / amp_mad5`

출력 예:
- `dataset/esp32_sequence_htltf_firstdiff_*`
- `dataset/esp32_sequence_htltf_ma5diff_*`
- `dataset/esp32_sequence_htltf_rawamp_*`
- `dataset/esp32_sequence_htltf_rollstd5_*`

의미:
- 현재 `esp32` 전처리 연구의 중심 스크립트

---

#### `scripts/build_esp32_sequence_feature_adaptive.py`

역할:
- adaptive gap reconstruction 위에 amplitude-derived feature를 쌓는다.
- `build_esp32_sequence_variants_adaptive.py`와 `build_esp32_sequence_ma10_dataset.py`의 중간 다리 역할

대표 사용:
- adaptive 전처리 + `first_difference`

출력 예:
- `dataset/esp32_sequence_htltf_firstdiff_w64_s10_adaptive_*`

---

### 5.4 phase-derived feature 데이터셋 생성

#### `scripts/build_esp32_sequence_phase_dataset.py`

역할:
- phase 기반 시계열 feature 생성

지원 feature:
- `phase_sin_cos`
- `phase_temporal_diff`
- `phase_rolling_std`

출력 예:
- `dataset/esp32_sequence_phase_sincos_*`
- `dataset/esp32_sequence_phase_tdiff_*`
- `dataset/esp32_sequence_phase_rollstd5_*`

의미:
- wrapped phase가 직접 유용한지, 또는 phase 변화량이 더 중요한지 확인하기 위한 스크립트

---

### 5.5 XFall / SDP 계열

#### `scripts/build_xfall_sdp_dataset.py`

역할:
- 기존 `preprocessed_raw` 데이터셋에 대해 paper-style XFall SDP 생성

특징:
- HT-LTF complex CSI 복원
- 고정 grid + 제한 보간
- lagged correlation으로 `lag x time` SDP 생성

출력:
- `dataset/xfall_sdp_*`

---

#### `scripts/build_xfall_sdp_lagfirst_dataset.py`

역할:
- 사용자가 제안한 **lag-first SDP** 생성
- 각 시점마다 lag vector를 먼저 만든 뒤, 그 시퀀스를 window로 자름

출력:
- `dataset/xfall_sdp_lagfirst_*`

의미:
- paper-style보다 현재 데이터에 더 잘 맞았던 대안 표현

---

#### `scripts/build_esp32_xfall_sdp_dataset.py`

역할:
- `esp32` 데이터셋 전용 paper-style XFall SDP 생성

입력:
- `dataset/esp32_raw_csi_variants/htltf_only`

출력:
- `dataset/esp32_xfall_sdp_*`

---

### 5.6 학습 스크립트

#### `scripts/train_row_mlp.py`

역할:
- `none / occupy / walk`용 row-level 3-layer MLP

특징:
- NumPy 기반
- `228`차원 HT-LTF row 입력

---

#### `scripts/train_esp32_row_mlp_torch.py`

역할:
- `esp32 big/small`용 row-level 3-layer MLP

입력:
- `LLTF only`, `HT-LTF only`, `LLTF+HT-LTF` raw row vector

---

#### `scripts/train_sequence_cnn.py`

역할:
- 초기 sequence baseline용 NumPy 1D CNN

의미:
- PyTorch 이전 빠른 baseline 용도

---

#### `scripts/train_sequence_cnn_torch.py`

역할:
- `none / occupy / walk`용 PyTorch 1D CNN

특징:
- file/time split
- Adam / SAM 지원
- 현재 로컬 수정으로 W&B 로깅 기능 추가

현재 로컬 수정 포인트:
- `wandb` import
- W&B run config/summary/table logging
- run metadata 확장

즉 이 파일은 현재 저장소에서 **유일하게 기존 tracked 파일을 직접 수정한 예**다.

---

#### `scripts/train_esp32_sequence_cnn_torch.py`

역할:
- `esp32` 시계열용 핵심 학습기
- amplitude/phase/derived sequence windows를 읽어 1D CNN 학습

특징:
- 입력 차원 자동 추론
- `interp_mask`, `delta_t_ms` 추가 채널 자동 감지
- 현재 `esp32` 전처리 ablation의 표준 학습기

---

#### `scripts/train_xfall_sdp_cnn_torch.py`

역할:
- `none / occupy / walk`용 XFall-style SDP 2D CNN baseline

---

#### `scripts/train_esp32_xfall_sdp_cnn_torch.py`

역할:
- `esp32 big/small`용 paper-style SDP 2D CNN baseline
- weighted CE 실험의 기반 코드

---

### 5.7 반복 실행 / 집계 스크립트

#### `scripts/run_esp32_row_mlp_experiments.py`

역할:
- row MLP를 3시드로 자동 반복
- aggregate summary 생성

#### `scripts/run_esp32_sequence_variant_experiments.py`

역할:
- `interp_only / interp_mask / interp_mask_deltat` 3종 비교

#### `scripts/run_esp32_sequence_ma10_experiments.py`

역할:
- amplitude-derived feature 데이터셋을 3시드로 반복 학습
- 현재 `esp32` preprocessing ablation의 가장 자주 쓰는 runner

#### `scripts/run_esp32_xfall_sdp_experiments.py`

역할:
- `esp32` SDP 실험을 3시드로 반복

#### `scripts/run_sequence_cnn_wandb_search.py`

역할:
- `windows_50 / windows_64`에 대해 미리 정한 10개 조합을 W&B에 기록하며 실행

---

### 5.8 보조 분석 / 시각화

#### `scripts/analyze_esp32_timeseries_structure.py`

역할:
- `esp32` 파일 길이와 sampling gap 분포 분석

출력:
- `analysis/esp32_timeseries_stats`

#### `scripts/visualize_esp32_amp_phase.py`

역할:
- 대표 파일의 amplitude / wrapped phase heatmap과 subcarrier summary 시각화

출력:
- `analysis/esp32_amp_phase_visuals`

#### `scripts/visualize_sequence_class_separation.py`

역할:
- sequence embedding, heatmap, temporal variation plot 생성

#### `scripts/compare_row_mlp_errors.py`

역할:
- 여러 row-MLP 실험에서 같은 샘플을 틀리는지 비교

---

### 5.9 Optimizer 구현

#### `scripts/sam_optimizer.py`

역할:
- NumPy/기존 학습 코드용 SAM optimizer 구현

#### `scripts/sam_torch.py`

역할:
- PyTorch용 SAM wrapper 구현

---

## 6. `dataset/` 인벤토리

`dataset/`은 “원본 → 전처리 → 윈도우” 계보를 중심으로 이해하면 된다.

### 6.1 원본 계열

- `dataset/raw`
  - 초기 `none / occupy / walk` 원본
- `dataset/raw_stride`
  - `large / normal / small` 원본
- `dataset/esp32`
  - 사람 이름과 `big/small` 라벨이 파일명에 들어간 새 원본 데이터셋

### 6.2 HT-LTF 추출 계열

- `dataset/preprocessed_raw`
  - `raw`에서 HT-LTF only 추출
- `dataset/preprocessed_raw_stride`
  - `raw_stride`에서 HT-LTF only 추출
- `dataset/esp32_raw_csi_variants`
  - `esp32`에서 `LLTF only`, `HT-LTF only`, `LLTF+HT-LTF` 추출 + 사람 split

### 6.3 일반 시계열 데이터셋 계열

- `dataset/sequence_10ms_amp_mask*`
  - 초기 `none / occupy / walk` / `raw_stride`용 amplitude sequence 데이터셋
- `dataset/esp32_sequence_htltf_variants_*`
  - `interp_only / mask / delta_t` basic sequence 데이터셋
- `dataset/esp32_sequence_htltf_firstdiff_*`
  - `first difference` feature 데이터셋
- `dataset/esp32_sequence_htltf_ma*`
  - moving average residual 데이터셋
- `dataset/esp32_sequence_htltf_rollstd*`
  - rolling std 데이터셋
- `dataset/esp32_sequence_htltf_rawamp_*`
  - raw amplitude sequence 데이터셋
- `dataset/esp32_sequence_phase_*`
  - phase-derived feature 데이터셋

### 6.4 SDP 계열

- `dataset/xfall_sdp_*`
  - `none / occupy / walk`용 paper-style SDP
- `dataset/xfall_sdp_lagfirst_*`
  - `none / occupy / walk`용 lag-first SDP
- `dataset/esp32_xfall_sdp_*`
  - `esp32`용 paper-style SDP

---

## 7. `analysis/` 인벤토리

`analysis/`는 코드가 아니라 “해석용 산출물”을 모아둔 위치다.

### 7.1 `analysis/esp32_timeseries_stats`

내용:
- 파일별 row 수 분포
- row 간 gap 분포
- `11ms` 근처 sampling cadence 검증

주요 파일:
- `summary.json`
- `file_level_distributions.png`
- `gap_distributions.png`

### 7.2 `analysis/esp32_amp_phase_visuals`

내용:
- HT-LTF amplitude / wrapped phase heatmap
- subcarrier summary plot

의미:
- amplitude와 phase가 어떤 성격을 갖는지 EDA 관점에서 확인

### 7.3 `analysis/raw_stride_sam50_curves`

내용:
- `raw_stride` 데이터셋에서 SAM 50 epoch 수렴/과적합 양상 plot

### 7.4 `analysis/sequence_stride20_windows50_visuals`

내용:
- `none / occupy / walk` 시계열 class separation 시각화
- PCA, t-SNE, heatmap, temporal variation plot 등

---

## 8. `artifacts/` 인벤토리

`artifacts/`는 “학습 실행 결과”를 보관하는 위치다.  
보통 각 폴더 내부에는:

- `training_summary.json`
- `best_model.pt` 또는 `best_model.npz`
- sometimes `history`

가 들어 있다.

중요한 가족 단위만 정리하면 다음과 같다.

### 8.1 row MLP 계열

- `mlp_row_split*`
  - `none / occupy / walk` row MLP
- `esp32_row_mlp_*`
  - `esp32` row MLP

### 8.2 기본 sequence 1D CNN 계열

- `sequence_cnn_*`
  - 초기 `none / occupy / walk` sequence baseline
- `esp32_sequence_variant_cnn_mps`
  - `interp_only / interp_mask / interp_mask_deltat`
- `esp32_sequence_firstdiff_*`
  - `first difference` 중심 전처리 ablation
- `esp32_sequence_ma*`
  - moving-average residual 계열
- `esp32_sequence_rollstd5_*`
  - rolling std
- `esp32_sequence_phase_*`
  - phase feature 실험
- `esp32_sequence_rawamp_*`
  - raw amplitude baseline

### 8.3 XFall / SDP 계열

- `xfall_sdp_*`
  - `none / occupy / walk` SDP 실험
- `esp32_xfall_sdp_*`
  - `esp32` SDP 실험

### 8.4 요약 JSON

루트에 있는 비교 요약:
- `esp32_sequence_ma_comparison.json`
- `esp32_sequence_diff_rollstd_comparison.json`
- `esp32_sequence_phase_feature_comparison.json`

의미:
- 각 feature family의 3-seed 평균 비교 결과를 한 파일에 정리한 것

---

## 9. 현재 코드 수정사항 요약

현재 로컬 변경을 “무엇이 새로 추가되었는가” 기준으로 요약하면:

### 9.1 새로 추가된 큰 기능

1. `esp32` 데이터셋 전용 raw CSI variant 추출
2. `esp32` 사람 기준 split 파이프라인
3. `esp32` 시계열 dataset builder
4. amplitude / phase / SDP feature builder
5. adaptive gap reconstruction builder
6. 3-seed batch runner
7. amplitude/phase/gap EDA 스크립트
8. `experience.md` 및 이번 `project_inventory.md`

### 9.2 기존 파일의 실질 수정

대표적으로 수정된 기존 파일:

- `scripts/train_sequence_cnn_torch.py`
  - W&B 관련 인자 추가
  - W&B init / epoch logging / confusion table 저장
  - run summary 확장

- `scripts/build_esp32_sequence_ma10_dataset.py`
  - 원래 MA residual 중심 builder였지만, 현재는
    - `raw_amplitude`
    - `ma_residual`
    - `first_difference`
    - `rolling_std`
    - `interp_mode`
    - `packet_filter`
    를 모두 지원하는 범용 builder로 확장됨

즉 현재 `esp32` preprocessing 연구에서 사실상 가장 많이 확장된 스크립트는  
`build_esp32_sequence_ma10_dataset.py` 이다.

---

## 10. 현재 기준 추천 읽기 순서

프로젝트를 처음 이해하려면 아래 순서가 가장 좋다.

1. `README.md`
   - 전체 배경과 큰 결과 이해
2. `experience.md`
   - 어떤 실험을 어떤 순서로 했는지 이해
3. `project_inventory.md`
   - 파일/코드 구조 이해
4. 핵심 스크립트 읽기
   - `scripts/extract_esp32_raw_csi_variants.py`
   - `scripts/build_esp32_sequence_ma10_dataset.py`
   - `scripts/train_esp32_sequence_cnn_torch.py`
5. 결과 확인
   - `artifacts/esp32_sequence_firstdiff_median11017_tol5000_nearest_cnn_mps/aggregate_summary.json`

---

## 11. 현재 가장 중요한 기준점

현재 `esp32 big/small` 실험에서 사실상 기준점이 되는 설정은 다음이다.

raw preprocessing:
- `HT-LTF only`
- `grid_us = 11017`
- `grid_tolerance_us = 5000`
- `interp_mode = nearest`
- `allow up to 2 missing packets`
- `packet_filter = none`

feature:
- `first difference`

model:
- `1D CNN`

결과:
- validation accuracy `0.8600 ± 0.0110`
- macro F1 `0.8480 ± 0.0092`

관련 결과:
- `artifacts/esp32_sequence_firstdiff_median11017_tol5000_nearest_cnn_mps/aggregate_summary.json`

이 설정이 현재 이후 실험의 baseline으로 보는 것이 가장 자연스럽다.

