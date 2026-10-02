# Battery Modeling Report

## 1. Dataset

| 구분             | Cell 수 | 역할                                 |
|------------------|--------:|--------------------------------------|
| Batch 1 Train    |      36 | Feature 비교 및 5-fold CV, 최종 학습 |
| Batch 1 Hold-out |      10 | Feature 확정 후 내부 검증            |
| Batch 2 Test     |      39 | 확정된 Ridge의 최종 batch 평가       |

- Batch 1: `data/2017-05-12_batchdata_updated_struct_errorcorrect.mat`
- Batch 2: `data/2018-02-20_batchdata_updated_struct_errorcorrect.mat`
- 데이터 사용 범위는 Batch 1과 Batch 2다.
- Target: **`log10(cycle_life)`**. 실제값과 예측값을 `10 ** value`로 복원한 뒤 cycle 단위 MAPE (%)를 계산했다.
- 양의 유효 label과 ΔQ entry를 가진 cell을 사용한다. 분할은 80/20, `random_state=42`다. Train 내부 shuffled 5-fold CV는 **각 fold 평가 부분의
  MAPE를 단순 산술평균**한다.
- 기존 feature API와 초기 100 cycle 이내 feature 식을 재사용한다. Capacity nominal·median imputer·StandardScaler는 각 fold의 학습 cell에서만
  추정·fit하고 평가 fold에 적용한다.
- 최종 nominal·전처리·Ridge는 **Batch 1 Train만으로 학습**한다. Hold-out/Test에는 학습된 값을 그대로 적용한다.
- cell은 **(`batch_id`, `cell_id`) 쌍**으로 식별한다. pandas `MultiIndex`로 관리하고 split의 ID 순서대로 feature와 target을 정렬한다.

### 학습과 예측 단계

`fit_model`은 다음 순서로 학습하고 `(imputer, scaler, model)`을 반환한다.

1. `SimpleImputer(strategy="median").fit_transform(학습 입력)`으로 학습 중앙값을 구하고 결측값을 채운다.
2. `StandardScaler().fit_transform(결측값 처리된 학습 입력)`으로 학습 평균·표준편차를 구하고 표준화한다.
3. `clone(MODELS[model_name]).fit(표준화된 학습 입력, 로그 수명)`으로 새 회귀 모델을 학습한다.

`predict_model`은 **imputer.transform → scaler.transform → model.predict**를 실행한다. 학습 입력에서 통계를 구하고 평가 입력에는 적용만 한다. 트리 모델에도
같은 표준화를 적용한다.

CV는 Batch 1 Train 내부에서 수행한다. 모든 비교 조합에 같은 fold별 feature 표를 사용하고, 조합·fold마다 전처리와 모델을 새로 학습한다. 초기 리팩터링은 설정과 계산 순서를 유지하며
`Pipeline`을 이 두 함수의 명시적 단계로 바꿨다.

## 2. Feature Sets

| Feature Set | 개수 | Features                                                                                 |
|-------------|-----:|------------------------------------------------------------------------------------------|
| F1          |    1 | `log_dQ_var`                                                                             |
| F2          |    2 | `log_dQ_var`, `dQ_kurt`                                                                  |
| F3          |    3 | `log_dQ_var`, `dQ_kurt`, `chargetime_med`                                                |
| F4          |    4 | `log_dQ_var`, `dQ_kurt`, `chargetime_med`, `QD_max_minus_init`                           |
| F5          |    5 | `log_dQ_var`, `dQ_kurt`, `chargetime_med`, `QD_max_minus_init`, `Tmin_med`               |
| F6          |    6 | `log_dQ_var`, `dQ_kurt`, `chargetime_med`, `QD_max_minus_init`, `Tmin_med`, `early_fade` |

F1~F6 전체를 평가한다. Predictor는 각 feature set에 나열한 항목으로 한정한다.

## 3. Model Comparison

모든 수치는 **Batch 1 Train 내부의 CV MAPE (%)**다. 모델 비교는 이 CV 결과로 수행한다.

| Feature Set |  Ridge | Random Forest | Gradient Boosting |
|-------------|-------:|--------------:|------------------:|
| F1          | 10.063 |         9.627 |            10.584 |
| F2          | 10.656 |        10.484 |            11.340 |
| F3          | 10.416 |        10.215 |            10.952 |
| F4          | 10.770 |        10.461 |            11.007 |
| F5          | 10.729 |        10.186 |            11.487 |
| F6          | 11.778 |        10.323 |            11.424 |

원래 설계의 Linear Regression baseline은 `log_dQ_var` 하나인 F1만 평가했다.

| Model             | Feature Set | Features     | Batch 1 CV MAPE (%) |
|-------------------|-------------|--------------|--------------------:|
| Linear Regression | F1          | `log_dQ_var` |              10.059 |

고정 설정: Ridge `alpha=1.0`; Random Forest `n_estimators=100`, `n_jobs=1`, `random_state=42`; Gradient Boosting
`n_estimators=100`, `learning_rate=0.1`, `max_depth=3`, `random_state=42`.

## 4. Selected Model

- 모델과 feature set은 설계에 따라 **Ridge + F3**로 명시적으로 지정한다.
- 최종 predictor는 `log_dQ_var`, `dQ_kurt`, `chargetime_med`다.
- CV 표는 비교 자료로 사용하고, 지정된 설정으로 Train 학습과 Hold-out 평가를 수행한다.
- **Test 기반 모델·feature·하이퍼파라미터 선택과 튜닝은 금지한다.**

Ridge F1의 CV MAPE는 10.063%, F3는 10.416%다. 비교표 최저값은 Random Forest F1의 9.627%다. 지정된 Ridge F3의 Hold-out MAPE는 8.344%다.

## 5. Performance Reporting

| 구분                     | MAPE (%) | 비고             |
|--------------------------|---------:|------------------|
| Train (Batch 1 CV)       |   10.416 | Batch 1 CV 평균  |
| Valid (Batch 1 Hold-out) |    8.344 | Batch 1 Hold-out |
| Test (Batch 2)           |   37.345 | Batch 2 Test     |
| Gap (Train-Valid)        |   -2.072 | pp; Valid − CV   |
| Gap (Valid-Test)         |  +29.000 | pp; Test − Valid |
| Gap (Target-Test)        |  +28.245 | pp; Test − 9.1%  |

MAPE의 단위는 %, Gap의 단위는 **percentage points (pp)**이다.

- `Gap (Train-Valid) = valid_mape - train_cv_mape`
- `Gap (Valid-Test) = test_mape - valid_mape`
- `Gap (Target-Test) = test_mape - 9.1`

9.1%는 작업 명세와 `PAPER_TARGET_MAPE = 9.1`의 **프로젝트 비교 기준**이다. 원논문 출처와 데이터·분할·feature·평가 조건의 동등성은 미검증이다. Gap은 프로젝트 기준과의 차이로
해석한다.

반올림 전 MAPE는 CV `10.416115867916506`, Hold-out `8.344286316379039`, Test `37.34452704441607`이다. Gap도 원값에서 계산해 소수 셋째 자리까지
표시한다. Test를 소수 둘째 자리까지 직접 반올림한 값은 **37.34%**다.

Hold-out MAPE는 CV보다 2.072pp 낮고, Test는 Hold-out보다 29.000pp 높다. CV는 fold 평가오차이며 학습 데이터 재대입 MAPE는 미계산이다.

## 6. 평가 이력과 검증

Batch 2는 앞선 구현과 추가 진단에서 열람·평가한 데이터다. 최종 Test 셀은 모델·feature 확정과 Hold-out 평가 후 실행하고, `final_test_started`로 한 커널의 실행 시작을
1회로 제한한다. 모델 선택은 지정된 설정을 유지하고 Test는 평가·진단 용도로 한정한다.

| 검증 기준                                                        | 결과                                                                                                                                                                   |
|------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `a87c8ba4c94b4db2ec7915614af7fd43741d99c8`                       | Python 3.12.14의 새 IPython 프로세스에서 코드 셀 25개 중 비어 있지 않은 24개를 파일 순서대로 read-only 재실행. 저장된 rich output·stdout·stderr 및 비교 조합 19개 일치 |
| 같은 기준 revision의 Notebook 검사                               | Ruff lint exit 1: B905 2개·C416 1개. Format exit 1: 재포맷 필요                                                                                                        |
| 같은 기준 revision의 Notebook + `src/skala_ds_mini_project` lint | 진단 9개: Notebook 3개와 기존 모듈 docstring RUF002 6개                                                                                                                |
| 포맷 커밋 `11e2a045177e92801797c5824d446aadba47539f`             | Notebook-only lint exit 1: B905 2개·C416 1개. Format 통과: `1 file already formatted`                                                                                  |

Notebook 검사 명령:

```sh
uv run --offline --frozen --no-sync ruff check --output-format=json notebooks/model.ipynb
uv run --offline --frozen --no-sync ruff format --check notebooks/model.ipynb
```

24개 셀 재현성은 `a87c8ba` 기준의 코드·저장 출력 일치 결과다. `11e2a04`에서는 Notebook lint·format만 다시 확인했다. 초기 리팩터링의 11개 셀 전후 비교와 당시 검사 통과는 과거
기록으로 구분한다.

## 7. Cell별 오차와 예측 방향

| Batch 2 Test APE      |    결과 |
|-----------------------|--------:|
| Mean APE              |  37.34% |
| Median APE            |  37.48% |
| 75% quantile          |  47.43% |
| 20% 초과 (`APE > 20`) | 31 / 39 |
| 50% 초과 (`APE > 50`) | 10 / 39 |

| 예측 방향               | Batch 1 Hold-out | Batch 2 Test |
|-------------------------|-----------------:|-------------:|
| 과대예측 cell           |                4 |           34 |
| 과소예측 cell           |                6 |            5 |
| Mean signed error       |           -2.53% |      +33.89% |
| 예측 / 실제 비율 중앙값 |                — |        1.375 |

Test에서는 높은 오차가 널리 분포하고 과대예측이 우세하다.

## 8. Train 기준 feature 분포

Batch 1 Train에 fit한 frozen median imputer와 StandardScaler를 적용한 Test z-score다. 범위 이탈도 Train 기준으로 집계한다.

| Feature          | Test 평균 z | Test 중앙값 z | Train 범위 밖 |
|------------------|------------:|--------------:|--------------:|
| `log_dQ_var`     |     +0.828σ |       +1.118σ |         28.2% |
| `chargetime_med` |     -0.738σ |       -0.714σ |           0개 |

Test의 `log_dQ_var`는 높은 영역으로, `chargetime_med`는 낮은 영역으로 이동했다.

## 9. Frozen Ridge 계수와 평균 예측 기여도

계수는 Notebook 셀 24/In[24]와 29/In[31]의 이름 기준 매핑을 따른다. Train 기준 표준화 값에 계수를 곱해 **Test − Hold-out 평균 예측 기여도 차이**를 구했다. 기여도
단위는 **log10 수명**이다.

| Feature          | Frozen Ridge 계수 | 평균 기여도 차이 |
|------------------|------------------:|-----------------:|
| `log_dQ_var`     |            -0.066 |           -0.067 |
| `dQ_kurt`        |            -0.003 |           -0.001 |
| `chargetime_med` |            +0.012 |           -0.013 |
| 합계             |                 — |          -0.0804 |

Frozen Ridge의 평균 log 예측은 Hold-out보다 Test에서 낮아졌다. 합계는 반올림 전 기여도로 계산했다.

## 10. 산술평균과 기하평균 수명 비교

비교 split은 **Batch 2 Test / Batch 1 Hold-out**이다. 아래 cycle 평균은 산술평균이며, 비율은 반올림 전 값에서 계산했다.

| Cycle 산술평균 | Hold-out |    Test | Test / Hold-out |
|----------------|---------:|--------:|----------------:|
| 실제           |  895.300 | 565.744 |   0.632 (63.2%) |
| 예측           |  856.629 | 711.160 |   0.830 (83.0%) |

예측 평균의 감소폭이 실제 평균보다 작았다.

`10 ** (Test log10 평균 − Hold-out log10 평균)`은 별도의 **기하평균 비율**이다.

| 기하평균 비율 | Test / Hold-out |
|---------------|----------------:|
| 실제          |           0.613 |
| 예측          |           0.831 |

## 11. Log 잔차

잔차는 **실제 log10 수명 − 예측 log10 수명**이다. 음수는 과대예측을 뜻한다.

| Batch 2 Test 잔차 | log10 수명 |
|-------------------|-----------:|
| 평균              |     -0.119 |
| 중앙값            |     -0.138 |
| 표준편차          |      0.083 |

| Feature          | Test log 잔차와의 상관 |
|------------------|-----------------------:|
| `log_dQ_var`     |                 -0.739 |
| `dQ_kurt`        |                 +0.713 |
| `chargetime_med` |                 -0.812 |

Feature 값에 따른 잔차 패턴이 관측됐다. Log intercept를 `δ`만큼 바꾸면 cycle 예측은 `10 ** δ`배가 된다. 상수 log 보정은 잔차의 위치를 이동시키며 잔차-feature 상관은
유지한다. 보정 후 성능은 평가 범위 밖이다.

## 12. Feature와 log target의 상관

각 split에서 feature와 실제 **`log10(cycle_life)`**의 단순 상관을 비교했다.

| Feature          | Train B1 | Valid B1 | Test B2 |
|------------------|---------:|---------:|--------:|
| `log_dQ_var`     |   -0.846 |   -0.951 |  -0.918 |
| `dQ_kurt`        |   +0.303 |   +0.373 |  +0.653 |
| `chargetime_med` |   +0.719 |   +0.824 |  -0.935 |

`log_dQ_var`의 음의 상관은 세 split에서 유지됐다. `chargetime_med`는 Batch 1의 양의 상관에서 Batch 2의 음의 상관으로 반전됐다.

## 13. Feature 간 상관과 VIF

| Feature 쌍                      | Train B1 | Valid B1 | Test B2 |
|---------------------------------|---------:|---------:|--------:|
| `log_dQ_var` ↔ `chargetime_med` |   -0.812 |   -0.809 |  +0.946 |
| `dQ_kurt` ↔ `chargetime_med`    |   +0.360 |        — |  -0.557 |
| `log_dQ_var` ↔ `dQ_kurt`        |   -0.393 |        — |  -0.496 |

`chargetime_med`와 다른 feature의 상관 방향이 Batch 2에서 바뀌었다.

| Feature VIF      | Train B1 | Valid B1 | Test B2 |
|------------------|---------:|---------:|--------:|
| `chargetime_med` |     2.96 |     2.98 |   10.61 |
| `log_dQ_var`     |     3.04 |     3.24 |    9.70 |
| `dQ_kurt`        |     1.19 |     1.15 |    1.47 |

VIF는 다른 feature들로 해당 feature를 설명하는 회귀에 기반한다. Test의 높은 VIF를 고려해 진단 계수는 해당 split의 연관성으로 해석한다.

## 14. 진단 OLS와 ablation

진단 OLS는 **Train·Valid·Test 각 split의 실제 log target에 각각 적합**한다. Test OLS의 `chargetime_med` 계수는 **-1.075**다. 이 값은 frozen
Ridge의 +0.012와 구분해 진단 자료로 사용한다.

Ablation은 각 feature를 제거한 뒤 **Batch 1 Train만으로 전처리와 Ridge를 재학습**한 비교다. Hold-out/Test 결과는 사후 진단에 사용하고 최종 Ridge + F3는
frozen 상태로 유지한다.

| Feature Set              | Valid MAPE | Test MAPE |
|--------------------------|-----------:|----------:|
| Full                     |      8.34% |    37.34% |
| Without `chargetime_med` |      7.75% |    36.38% |
| Without `dQ_kurt`        |      8.42% |    37.56% |
| Without `log_dQ_var`     |     14.65% |    53.47% |

`chargetime_med` 제거·재학습 후에도 Test 오차는 36.38%다. `log_dQ_var` 제거 시 53.47%로 증가했다. 이 비교는 함께 재학습된 계수와 feature 조합의 예측 성능으로
해석한다.
