# Experiment Log

이 문서는 현재 프로젝트에서 수행한 주요 전처리/모델링 실험과 결과를 요약한다. 수치는 가능한 한 3-seed 평균 또는 최종 집계 JSON 기준으로 정리했다.

## 1. Project Scope

현재까지 실험은 크게 두 축으로 나뉜다.

1. `none / occupy / walk` 3-class 분류
2. `esp32` 데이터셋 기반 `big / small` 2-class 보폭 분류

핵심 목적은 다음 두 가지였다.

1. CSI에서 어떤 표현이 실제로 분류 신호를 담고 있는지 확인
2. raw CSI 자체의 전처리(`LTF 선택`, `시간축 재구성`, `보간`, `packet filtering`)가 성능에 어떤 영향을 주는지 확인

## 2. `none / occupy / walk` 3-Class Experiments

### 2.1 Raw CSI 이해와 기본 전처리

- 원본 ESP32 CSI `data`는 `384 ints = 192 complex`
- HT 40MHz non-STBC 패킷으로 해석했고, 구조는 `LLTF 64 complex + HT-LTF 128 complex`
- 단순 baseline에서는 `HT-LTF only`만 사용
- 유효 carrier만 남기면 `228 ints = 114 complex`

관련 스크립트:
- `scripts/preprocess_raw_htltf.py`

관련 데이터셋:
- `dataset/preprocessed_raw`
- `dataset/preprocessed_raw_stride`

### 2.2 Row-Level MLP

행 하나를 하나의 샘플로 보고 3-layer MLP를 학습했다.

대표 결과:
- 20 epoch Adam: validation accuracy 약 `0.9989`
- 50 epoch Adam: validation accuracy 약 `0.9990`
- 25 epoch SAM-SGD: validation accuracy 약 `0.9985`

관련 결과:
- `artifacts/mlp_row_split`
- `artifacts/mlp_row_split_50ep`
- `artifacts/mlp_row_split_sam_sgd_25ep`

해석:
- row 단위 CSI만으로도 세 클래스가 강하게 분리됨
- 다만 row random split은 낙관적일 수 있어서 이후 file/time split 시퀀스 실험으로 확장

### 2.3 Sequence 1D CNN

재샘플링:
- `10ms` 기준 grid
- 짧은 gap만 보간
- `window=50`, `64`

대표 결과:
- `windows_64`, file/time split 기준 validation accuracy `1.0`
- `windows_50`, file/time split 기준 validation accuracy `1.0`

관련 결과:
- `artifacts/sequence_cnn_torch_windows64_time_cpu_nw0`
- `artifacts/sequence_cnn_torch_windows50_time_cpu_nw0`

추가 확인:
- stride를 `10`, `20`으로 줄여도 여전히 완전 분리
- block split까지 넣어도 분리 강도가 매우 높음

해석:
- 이 데이터셋은 현재 표현 수준에서 난도가 낮거나, 클래스별 환경 차이가 매우 강함

### 2.4 XFall / SDP 계열

논문식 paper-style SDP와 lag-first SDP를 모두 구현했다.

paper-style SDP:
- `window=50, lag=20`: accuracy `0.6884`
- `window=64, lag=20`: accuracy `0.7242`
- `window=64, lag=49`: accuracy `0.7376`

lag-first SDP:
- `window=50, lag=50`: accuracy `0.7549`
- `window=64, lag=30`: accuracy `0.7844`
- `window=64, lag=50`: accuracy `0.8231`

관련 결과:
- `artifacts/xfall_sdp_cnn_windows50_time_mps`
- `artifacts/xfall_sdp_cnn_windows64_time_mps`
- `artifacts/xfall_sdp_cnn_windows64_lag49_time_mps`
- `artifacts/xfall_sdp_cnn_lagfirst_windows50_lag50_time_mps`
- `artifacts/xfall_sdp_cnn_lagfirst_windows64_lag30_time_mps`
- `artifacts/xfall_sdp_cnn_lagfirst_windows64_lag50_time_mps`

해석:
- paper-style SDP보다 lag-first SDP가 현재 데이터에 더 잘 맞음
- 긴 lag 문맥과 충분한 time width를 동시에 확보하는 것이 중요했음

## 3. `raw_stride` (`large / normal / small`) 3-Class Experiments

### 3.1 전처리

- `HT-LTF only` 추출
- `10ms` grid + tolerance 확장(`4000us`)
- amplitude 기반 시퀀스 구성

관련 데이터:
- `dataset/preprocessed_raw_stride`
- `dataset/sequence_10ms_amp_mask_raw_stride_tol4000`

### 3.2 1D CNN Results

Adam 20 epoch:
- `windows_64`: accuracy `0.4078`, macro F1 `0.343`
- `windows_50`: accuracy `0.4422`, macro F1 `0.414`

SAM 50 epoch:
- `windows_64`: accuracy `0.4665`, macro F1 `0.383`
- `windows_50`: accuracy `0.5336`, macro F1 `0.456`

관련 결과:
- `artifacts/sequence_cnn_torch_raw_stride_tol4000_windows64_timefile_mps_seed42`
- `artifacts/sequence_cnn_torch_raw_stride_tol4000_windows50_timefile_mps_seed42`
- `artifacts/sequence_cnn_torch_raw_stride_tol4000_windows64_timefile_sam50_mps_seed42`
- `artifacts/sequence_cnn_torch_raw_stride_tol4000_windows50_timefile_sam50_mps_seed42`

해석:
- `raw_stride`는 이전 `none/occupy/walk`보다 훨씬 어렵다
- SAM이 Adam보다 최고 성능은 더 좋았지만, 여전히 과적합이 강함

## 4. `esp32` Big/Small Dataset Setup

### 4.1 파일 구조와 split

파일명 규칙:
- `csi_YYMMDD_HHMMSS_{person}_{big|small}.csv`

라벨 정리:
- `cheawon -> chaewon`
- `smal -> small`
- 파싱 불가능 파일은 제외

사람 기준 split:
- train: `chaewon, heewon, junho, junsoo, minhyeok`
- validation: `doyun, jinyoung`

관련 요약:
- `dataset/esp32_raw_csi_variants/summary.json`
- `dataset/esp32_raw_csi_variants/split_manifest.csv`

### 4.2 Raw LTF Variants

추출한 raw CSI variant:
- `LLTF only`: `104 ints = 52 complex`
- `HT-LTF only`: `228 ints = 114 complex`
- `LLTF + HT-LTF`: `332 ints = 166 complex`

관련 데이터:
- `dataset/esp32_raw_csi_variants/lltf_only`
- `dataset/esp32_raw_csi_variants/htltf_only`
- `dataset/esp32_raw_csi_variants/lltf_htltf`

## 5. `esp32` Row-Level Baseline

행 하나를 하나의 샘플로 보고 3-layer MLP를 학습했다.

3-seed 평균:
- `LLTF only`: acc `0.6068 ± 0.0093`, macro F1 `0.4245 ± 0.0072`
- `HT-LTF only`: acc `0.6196 ± 0.0045`, macro F1 `0.4884 ± 0.0180`
- `LLTF+HT-LTF`: acc `0.6130 ± 0.0059`, macro F1 `0.4858 ± 0.0150`

관련 결과:
- `artifacts/esp32_row_mlp_mps/aggregate_summary.json`

해석:
- row 단독 분류는 약함
- `HT-LTF only`가 가장 낫고, 단순 concat(`LLTF+HT-LTF`)은 오히려 손해

## 6. `esp32` Sequence Baseline: 초기 비교

초기 재샘플링 baseline:
- `11ms` grid
- `tolerance=4000us`
- `<=3 step` gap만 보간
- `window=64`, `stride=10`

입력 변형:
- `interp_only`: amplitude only
- `interp_mask`: amplitude + 보간 여부
- `interp_mask_deltat`: amplitude + 보간 여부 + delta_t

3-seed 평균:
- `interp_only`: acc `0.6756 ± 0.0228`, macro F1 `0.6753 ± 0.0233`
- `interp_mask`: acc `0.6786 ± 0.0102`, macro F1 `0.6774 ± 0.0122`
- `interp_mask_deltat`: acc `0.6793 ± 0.0066`, macro F1 `0.6783 ± 0.0079`

관련 결과:
- `artifacts/esp32_sequence_variant_cnn_mps/aggregate_summary.json`

해석:
- mask와 delta_t는 큰 폭의 향상은 아니지만 안정성 개선에 도움

## 7. `esp32` Feature Extraction Results

이 절은 raw preprocessing 위에서 amplitude/phase 파생 feature를 비교한 결과다.

### 7.1 Amplitude-derived Features

3-seed 평균:
- `raw amplitude only`: acc `0.6756 ± 0.0228`, macro F1 `0.6753 ± 0.0233`
- `MA5 residual`: acc `0.7323 ± 0.0249`, macro F1 `0.7262 ± 0.0236`
- `MA10 residual`: acc `0.6686 ± 0.0130`, macro F1 `0.6450 ± 0.0236`
- `MA20 residual`: acc `0.6328 ± 0.0021`, macro F1 `0.5877 ± 0.0094`
- `MA30 residual`: acc `0.5930 ± 0.0047`, macro F1 `0.5518 ± 0.0060`
- `first difference`: acc `0.7468 ± 0.0150`, macro F1 `0.7447 ± 0.0133`
- `rolling std(5)`: acc `0.7278 ± 0.0073`, macro F1 `0.7070 ± 0.0129`

관련 결과:
- `artifacts/esp32_sequence_ma_comparison.json`
- `artifacts/esp32_sequence_diff_rollstd_comparison.json`

핵심 해석:
- amplitude 계열에서는 `first difference`가 가장 강력함
- 너무 긴 moving average는 유효한 저주파 변화까지 제거해 손해

### 7.2 Phase-derived Features

3-seed 평균:
- `phase_sin_cos`: acc `0.5793 ± 0.0000`, macro F1 `0.3668 ± 0.0000`
- `phase_temporal_diff`: acc `0.6639 ± 0.0078`, macro F1 `0.6521 ± 0.0102`
- `phase_rolling_std5`: acc `0.6002 ± 0.0074`, macro F1 `0.5817 ± 0.0106`

관련 결과:
- `artifacts/esp32_sequence_phase_feature_comparison.json`

핵심 해석:
- raw wrapped phase 자체는 약함
- phase에서도 “절대값”보다 “시간 변화량”인 `phase_temporal_diff`가 가장 의미 있었음

## 8. `esp32` XFall / SDP Experiments

사용 설정:
- `HT-LTF only`
- `window=100`
- `lag=20`
- `stride=10`

paper-style SDP:
- `real`: unweighted / weighted 모두 실패
- `magnitude` + weighted CE: acc `0.6892 ± 0.0000`, macro F1 `0.6883 ± 0.0011`

관련 결과:
- `artifacts/esp32_xfall_sdp_cnn_w100_mps`
- `artifacts/esp32_xfall_sdp_cnn_w100_mps_mag`
- `artifacts/esp32_xfall_sdp_cnn_w100_mps_mag_weighted`
- `artifacts/esp32_xfall_sdp_cnn_w100_mps_real_weighted`

핵심 해석:
- 현재 `big/small` 문제에서는 paper-style SDP보다 amplitude 변화량 계열이 더 강함
- class imbalance를 고려하지 않으면 trivial solution으로 붕괴

## 9. `esp32` Raw Preprocessing Ablation

이 절이 현재 전처리 연구의 핵심이다. feature는 `first difference`로 고정하고, raw preprocessing만 바꿨다.

### 9.1 Global `11ms` vs Median-fixed vs Adaptive EMA

3-seed 평균:
- fixed `11ms`: acc `0.7468 ± 0.0150`, macro F1 `0.7447 ± 0.0133`
- median-fixed `11017us`: acc `0.7626 ± 0.0099`, macro F1 `0.7541 ± 0.0159`
- adaptive EMA gap model: acc `0.7274 ± 0.0096`, macro F1 `0.7213 ± 0.0122`

관련 결과:
- `artifacts/esp32_sequence_firstdiff_cnn_mps/aggregate_summary.json`
- `artifacts/esp32_sequence_firstdiff_median11017_cnn_mps/aggregate_summary.json`
- `artifacts/esp32_sequence_firstdiff_adaptive_cnn_mps/aggregate_summary.json`

해석:
- global median fixed gap이 adaptive EMA보다 더 안정적이고 더 좋았음

### 9.2 Grid Tolerance Ablation (`linear`)

조건:
- `grid_us=11017`
- `interp_mode=linear`
- `allow up to 2 missing packets`

3-seed 평균:
- `2000us`: acc `0.5833`, macro F1 `0.3684`
- `3000us`: acc `0.5672`, macro F1 `0.3882`
- `4000us`: acc `0.7626`, macro F1 `0.7541`
- `5000us`: acc `0.8582`, macro F1 `0.8411`

관련 결과:
- `artifacts/esp32_sequence_firstdiff_median11017_tol2000_cnn_mps/aggregate_summary.json`
- `artifacts/esp32_sequence_firstdiff_median11017_tol3000_cnn_mps/aggregate_summary.json`
- `artifacts/esp32_sequence_firstdiff_median11017_cnn_mps/aggregate_summary.json`
- `artifacts/esp32_sequence_firstdiff_median11017_tol5000_cnn_mps/aggregate_summary.json`

해석:
- tolerance가 너무 작으면 usable window가 급감하며 붕괴
- `5000us`가 가장 좋았음

### 9.3 Missing-packet Handling

`grid_us=11017`, `tolerance=5000`, `linear` 기준으로 비교:

- `missing 0`: validation window가 없어 비교 불가
- `missing 1`: acc `0.8393 ± 0.0242`, macro F1 `0.8269 ± 0.0317`
- `missing 2`: acc `0.8582 ± 0.0049`, macro F1 `0.8411 ± 0.0068`
- `missing 3`: acc `0.8605 ± 0.0120`, macro F1 `0.8370 ± 0.0144`

관련 결과:
- `artifacts/esp32_sequence_firstdiff_median11017_tol5000_miss1_cnn_mps/aggregate_summary.json`
- `artifacts/esp32_sequence_firstdiff_median11017_tol5000_cnn_mps/aggregate_summary.json`
- `artifacts/esp32_sequence_firstdiff_median11017_tol5000_miss3_cnn_mps/aggregate_summary.json`

해석:
- accuracy만 보면 `missing 3`이 가장 높음
- macro F1과 안정성까지 보면 `missing 2`가 더 균형적

### 9.4 Interpolation Method Comparison

조건:
- `grid_us=11017`
- `tolerance=5000`
- `allow up to 2 missing packets`

3-seed 평균:
- `linear`: acc `0.8582 ± 0.0049`, macro F1 `0.8411 ± 0.0068`
- `forward_fill`: acc `0.8435 ± 0.0110`, macro F1 `0.8256 ± 0.0114`
- `nearest`: acc `0.8600 ± 0.0110`, macro F1 `0.8480 ± 0.0092`

관련 결과:
- `artifacts/esp32_sequence_firstdiff_median11017_tol5000_cnn_mps/aggregate_summary.json`
- `artifacts/esp32_sequence_firstdiff_median11017_tol5000_ffill_cnn_mps/aggregate_summary.json`
- `artifacts/esp32_sequence_firstdiff_median11017_tol5000_nearest_cnn_mps/aggregate_summary.json`

해석:
- `nearest`가 가장 좋았음
- `linear`도 거의 비슷하게 강함
- `forward_fill`은 분명히 더 약함

### 9.5 Grid Tolerance Ablation (`nearest`)

조건:
- `grid_us=11017`
- `interp_mode=nearest`
- `allow up to 2 missing packets`

3-seed 평균:
- `2000us`: acc `0.5833 ± 0.0000`, macro F1 `0.3684 ± 0.0000`
- `3000us`: acc `0.5638 ± 0.0029`, macro F1 `0.3640 ± 0.0072`
- `4000us`: acc `0.7570 ± 0.0055`, macro F1 `0.7554 ± 0.0067`
- `5000us`: acc `0.8600 ± 0.0110`, macro F1 `0.8480 ± 0.0092`

관련 결과:
- `artifacts/esp32_sequence_firstdiff_median11017_tol2000_nearest_cnn_mps/aggregate_summary.json`
- `artifacts/esp32_sequence_firstdiff_median11017_tol3000_nearest_cnn_mps/aggregate_summary.json`
- `artifacts/esp32_sequence_firstdiff_median11017_tol4000_nearest_cnn_mps`
- `artifacts/esp32_sequence_firstdiff_median11017_tol5000_nearest_cnn_mps/aggregate_summary.json`

핵심 해석:
- `nearest`에서도 결론은 동일하게 `5000us`가 가장 좋음

### 9.6 Packet Filtering

조건:
- `grid_us=11017`
- `tolerance=5000`
- `interp_mode=nearest`
- `allow up to 2 missing packets`

필터 정의:
- `amp_madK`: 파일 내부 packet mean amplitude의 robust z-score가 `K`를 넘으면 제거

3-seed 평균:
- `none`: acc `0.8600 ± 0.0110`, macro F1 `0.8480 ± 0.0092`
- `amp_mad3`: acc `0.8405 ± 0.0082`, macro F1 `0.8259 ± 0.0083`
- `amp_mad4`: acc `0.8597 ± 0.0085`, macro F1 `0.8454 ± 0.0069`
- `amp_mad5`: acc `0.8486 ± 0.0078`, macro F1 `0.8383 ± 0.0064`

관련 결과:
- `artifacts/esp32_sequence_firstdiff_median11017_tol5000_nearest_ampmad3_cnn_mps/aggregate_summary.json`
- `artifacts/esp32_sequence_firstdiff_median11017_tol5000_nearest_ampmad4_cnn_mps/aggregate_summary.json`
- `artifacts/esp32_sequence_firstdiff_median11017_tol5000_nearest_ampmad5_cnn_mps/aggregate_summary.json`

해석:
- packet filtering은 현재 기준으로 이득이 없었음
- 가장 좋은 설정은 `packet_filter = none`

## 10. Raw Amplitude vs First Difference Under Best Preprocessing

최종적으로 같은 raw preprocessing에서 `raw amplitude`와 `first difference`를 직접 비교했다.

고정 전처리:
- `HT-LTF only`
- `grid_us = 11017`
- `grid_tolerance_us = 5000`
- `interp_mode = nearest`
- `allow up to 2 missing packets`
- `packet_filter = none`

3-seed 평균:
- `raw amplitude`: acc `0.7151 ± 0.0137`, macro F1 `0.6940 ± 0.0097`
- `first difference`: acc `0.8600 ± 0.0110`, macro F1 `0.8480 ± 0.0092`

관련 결과:
- `artifacts/esp32_sequence_rawamp_median11017_tol5000_nearest_cnn_mps/aggregate_summary.json`
- `artifacts/esp32_sequence_firstdiff_median11017_tol5000_nearest_cnn_mps/aggregate_summary.json`

해석:
- 현재 문제에서는 raw amplitude보다 변화량 강조(`first difference`)가 훨씬 강력함

## 11. LLTF-Based Amplitude Normalization

다음으로 `LLTF`와 `HT-LTF`의 공통 52개 carrier를 이용해, 각 row마다 robust median ratio scale factor를 계산하고 전체 `HT-LTF amplitude`를 정규화하는 전처리를 추가했다.

정규화 함수:
- `scripts/csi_amplitude_normalization.py`

핵심 아이디어:
- `H_L`, `H_HT`에서 amplitude만 사용
- 공통 carrier 구간에서 `(HT amplitude) / (LLTF amplitude)` ratio 계산
- ratio의 median을 scale factor로 사용
- 전체 `HT-LTF amplitude`를 이 scale factor로 나눔

같은 raw preprocessing + `first difference` 기준 3-seed 평균:
- 기존 `HT-LTF first difference`: acc `0.8600 ± 0.0110`, macro F1 `0.8480 ± 0.0092`
- `LLTF-normalized HT-LTF first difference`: acc `0.8663 ± 0.0030`, macro F1 `0.8486 ± 0.0022`

관련 결과:
- `artifacts/esp32_sequence_firstdiff_median11017_tol5000_nearest_cnn_mps/aggregate_summary.json`
- `artifacts/esp32_sequence_lltfnorm_firstdiff_median11017_tol5000_nearest_cnn_mps/aggregate_summary.json`

해석:
- 사람 기준 split에서는 LLTF normalization이 작은 개선을 보였음
- accuracy는 약간 상승했고, std가 줄어들어 결과가 더 안정적이었음
- macro F1은 거의 같아서 “큰 성능 도약”보다는 “약한 보정 이득”에 가까움

## 12. Date-Based Split Experiments

사람 기준 split 대신, **같은 날짜에 수집된 파일이 train/validation에 섞이지 않도록** 날짜 기준 split을 새로 만들었다.

가능한 날짜:
- `260331`
- `260401`
- `260402`

정확한 `8:2`는 날짜가 3개뿐이라 불가능했고, 가장 가까운 조합은 아래였다.

split:
- train: `260331`, `260402`
- validation: `260401`

행 수 비율:
- train: `303126` rows (`75.67%`)
- validation: `97458` rows (`24.33%`)

관련 요약:
- `dataset/esp32_raw_csi_variants_by_date/summary.json`
- `dataset/esp32_raw_csi_variants_by_date/split_manifest.csv`

### 12.1 Sequence Baseline Under Date Split

고정한 전처리:
- `grid_us = 11017`
- `grid_tolerance_us = 5000`
- `interp_mode = nearest`
- `allow up to 2 missing packets`
- `window = 64`
- `stride = 10`

3-seed 평균:
- `raw amplitude`: acc `0.4656 ± 0.0246`, macro F1 `0.4533 ± 0.0205`
- `first difference`: acc `0.6814 ± 0.0175`, macro F1 `0.5482 ± 0.0783`
- `LLTF-normalized first difference`: acc `0.7364 ± 0.0196`, macro F1 `0.6511 ± 0.0348`

관련 결과:
- `artifacts/esp32_date_sequence_rawamp_median11017_tol5000_nearest_cnn_mps/aggregate_summary.json`
- `artifacts/esp32_date_sequence_firstdiff_median11017_tol5000_nearest_cnn_mps/aggregate_summary.json`
- `artifacts/esp32_date_sequence_lltfnorm_firstdiff_median11017_tol5000_nearest_cnn_mps/aggregate_summary.json`

해석:
- date split은 사람 split보다 훨씬 어려웠음
- `raw amplitude`는 날짜 도메인 변화에 매우 약했음
- `first difference`는 date shift에서도 유효했음
- `LLTF-normalized first difference`가 date split에서 가장 좋았고, `first difference` 대비 accuracy 약 `+5.5%p`, macro F1 약 `+10.3%p` 개선

## 13. Date-Based SDP Comparison

같은 date split에서 XFall paper-style SDP도 비교했다.

설정:
- `window = 100`
- `lag = 20`
- `rho_mode = magnitude`
- `column_normalization = shift`
- `grid_us = 11017`
- `grid_tolerance_us = 5000`
- `max_interp_gap_steps = 3`
- `weighted cross entropy`

비교 대상:
- `HT-LTF only` SDP
- `LLTF-normalized HT-LTF` SDP

3-seed 평균:
- `HT-LTF SDP`: acc `0.4308 ± 0.0034`, macro F1 `0.4268 ± 0.0044`
- `LLTF-normalized SDP`: acc `0.4318 ± 0.0017`, macro F1 `0.4273 ± 0.0032`

관련 결과:
- `artifacts/esp32_date_xfall_sdp_htltf_w100_l20_grid11017_tol5000_mag_weighted/aggregate_summary.json`
- `artifacts/esp32_date_xfall_sdp_lltfnorm_w100_l20_grid11017_tol5000_mag_weighted/aggregate_summary.json`

해석:
- date split에서는 SDP가 sequence `first difference` baseline보다 훨씬 약했음
- LLTF normalization을 넣어도 SDP 성능은 거의 달라지지 않았음
- 현재 `big/small` 문제에서 date-domain generalization에는 paper-style SDP가 잘 맞지 않음

## 14. Current Best Setting

현재까지 `esp32 big/small` 문제에서 가장 강한 실험적 baseline은 **사람 기준 split**과 **날짜 기준 split**에서 모두 `LLTF-normalized first difference` 계열이다.

전처리:
- `LLTF+HT-LTF raw`에서 LLTF 기반 scale factor 계산
- 정규화된 `HT-LTF amplitude` 사용
- `grid_us = 11017`
- `grid_tolerance_us = 5000`
- `interp_mode = nearest`
- `allow up to 2 missing packets`
- `packet_filter = none`

입력 feature:
- `first difference`

모델:
- 1D CNN

사람 기준 split 성능:
- validation accuracy `0.8663 ± 0.0030`
- macro F1 `0.8486 ± 0.0022`

관련 결과:
- `artifacts/esp32_sequence_lltfnorm_firstdiff_median11017_tol5000_nearest_cnn_mps/aggregate_summary.json`

날짜 기준 split 성능:
- validation accuracy `0.7364 ± 0.0196`
- macro F1 `0.6511 ± 0.0348`

관련 결과:
- `artifacts/esp32_date_sequence_lltfnorm_firstdiff_median11017_tol5000_nearest_cnn_mps/aggregate_summary.json`

## 15. Practical Conclusions

지금까지의 실험에서 얻은 결론은 다음과 같다.

1. `HT-LTF`가 주된 분류 신호였고, `LLTF`는 직접 입력보다 **정규화 기준**으로 사용할 때 더 유용했다.
2. 사람 기준 split을 쓰면 row 단독 모델은 약하고, 시계열 모델이 필요하다.
3. `global 11ms`보다 실제 median인 `11017us`를 grid로 두는 것이 더 좋았다.
4. tolerance는 작게 잡으면 데이터가 무너지고, 현재는 `5000us`가 가장 좋았다.
5. interpolation은 `nearest`가 가장 좋고, `linear`도 거의 비슷했다.
6. packet filtering은 현재 기준으로 도움이 되지 않았다.
7. raw amplitude보다 변화량 기반 feature가 훨씬 중요했다.
8. amplitude 계열에서는 `first difference`가 가장 좋은 baseline이었다.
9. LLTF 기반 amplitude normalization은 사람 split에서는 작은 이득, date split에서는 더 큰 이득을 보였다.
10. phase는 단독으로는 약했지만, `phase_temporal_diff`는 보조 feature 후보로 남아 있다.
11. date split에서는 raw amplitude가 크게 무너졌고, 변화량 feature가 훨씬 더 robust했다.
12. paper-style SDP는 현재 `big/small` 문제의 date-domain generalization 기준으로는 경쟁력이 낮았다.

## 16. Recommended Next Steps

다음 실험 우선순위는 아래와 같다.

1. date split 기준으로 `raw amplitude + first difference` 멀티채널 입력
2. date split 기준으로 `first difference + phase_temporal_diff` 조합
3. `LLTF + HT-LTF`를 두 branch로 나눠 입력하는 모델
4. date split 기준으로 `LLTF normalization`이 다른 feature에도 이득이 있는지 확인
5. preprocessing은 현재 best setting을 고정한 채 feature 조합만 비교
