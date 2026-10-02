# Battery Modeling Report

기준 설계: `docs/Day1.md`. 결과 원본: `notebooks/model.ipynb`의 실제 실행 출력.  

**F1~F6을 모두 비교**하되, 최종 모델과 feature set은 성능 순위와 관계없이 **Ridge + F3로 고정**한다. 이번 리팩터링은 이 설정과 계산 순서를 유지하면서 `Pipeline`을 명시적인 학습·예측 단계로 바꾼 것이다. 노트북에는 짧은 결과 중심 제목과 한국어 코드 설명을 두며, 상세 방법·선택 근거·해석은 이 문서에 분리한다.

## 1. Dataset

| 구분 | Cell 수 | 역할 |
|---|---:|---|
| Batch 1 Train | 36 | Feature 비교 및 5-fold CV, 최종 학습 |
| Batch 1 Hold-out | 10 | Feature 확정 후 내부 검증 |
| Batch 2 Test | 39 | 확정된 Ridge의 최종 batch 평가 |

- Batch 1: `data/2017-05-12_batchdata_updated_struct_errorcorrect.mat`
- Batch 2: `data/2018-02-20_batchdata_updated_struct_errorcorrect.mat`
- Batch 3는 읽거나 사용하지 않았다.
- Target: **`log10(cycle_life)`**. 실제값과 예측값을 `10 ** value`로 복원한 뒤 cycle 단위 MAPE(%)를 계산했다.
- 기존 양의 유효 label과 ΔQ entry를 가진 cell을 사용했다. 기존 분할인 80/20, `random_state=42`와 Train 내부 shuffled 5-fold를 유지했다. CV 수치는 fold별 MAPE의 단순 산술평균이다.
- 기존 feature API와 초기 100 cycle 이내 feature 식을 재사용한다. Capacity nominal은 각 fold의 학습 cell에서만 추정하여 해당 평가 fold에 재사용한다. Median imputer와 StandardScaler 역시 학습 데이터에서만 fit한다. 최종 Train의 nominal과 학습된 결측값 처리기·스케일러·Ridge를 Hold-out/Test에 그대로 적용하며, Hold-out을 합쳐 재학습하지 않는다.
- cell의 식별자는 `cell_id` 하나가 아니라 **(`batch_id`, `cell_id`) 쌍**이다. pandas `MultiIndex`로 두 ID를 함께 관리하여 batch가 다르면 같은 cell 번호도 다른 cell로 구분한다. Feature 표는 split 당시 ID 순서로 정렬하여 입력과 정답의 행 순서를 맞춘다.

### 학습과 예측 단계

`fit_model`은 다음 세 단계를 직접 실행하고 `(imputer, scaler, model)` 튜플을 반환한다.

1. `SimpleImputer(strategy="median").fit_transform(학습 입력)`으로 학습 중앙값을 구하고 결측값을 채운다.
2. `StandardScaler().fit_transform(결측값 처리된 학습 입력)`으로 학습 평균·표준편차를 구하고 표준화한다.
3. `clone(MODELS[model_name]).fit(표준화된 학습 입력, 로그 수명)`으로 새 회귀 모델을 학습한다.

`predict_model`은 위 튜플을 받아 **imputer.transform → scaler.transform → model.predict**만 실행한다. `fit`은 통계를 학습하는 단계이고, `transform`은 이미 학습한 통계를 적용하는 단계다. 평가 입력에서 다시 fit하면 평가 데이터의 분포가 학습에 섞이므로, 이 경계는 반드시 유지해야 한다. 트리 모델에도 기존과 동일한 표준화를 적용하여 계산 결과를 보존한다.

CV에서는 Batch 1 Train 내부를 5개 fold로 나눈다. 매 fold의 학습 부분으로 nominal·중앙값·평균·표준편차·회귀 계수를 구하고, 해당 평가 부분에는 적용만 한다. Hold-out은 어느 fold에도 들어가지 않는다. 동일한 fold별 feature 표를 모든 비교 조합에 재사용하되, 결측값 처리기·스케일러·모델은 각 조합과 fold마다 새로 학습한다.

`Pipeline`은 전처리와 모델을 묶어 단계 누락을 줄이고 `cross_val_score`나 `GridSearchCV` 같은 자동 평가·탐색에 전달하기 유용하다. 이 노트북은 두 API를 사용하지 않고 nominal 추정부터 CV를 직접 반복하므로, 작은 두 함수로 단계를 드러내는 것으로 충분하다. **Pipeline 제거가 학습 데이터만으로 통계를 추정해야 한다는 원칙을 없애는 것은 아니다.**

## 2. Feature Sets

| Feature Set | 개수 | Features |
|---|---:|---|
| F1 | 1 | `log_dQ_var` |
| F2 | 2 | `log_dQ_var`, `dQ_kurt` |
| F3 | 3 | `log_dQ_var`, `dQ_kurt`, `chargetime_med` |
| F4 | 4 | `log_dQ_var`, `dQ_kurt`, `chargetime_med`, `QD_max_minus_init` |
| F5 | 5 | `log_dQ_var`, `dQ_kurt`, `chargetime_med`, `QD_max_minus_init`, `Tmin_med` |
| F6 | 6 | `log_dQ_var`, `dQ_kurt`, `chargetime_med`, `QD_max_minus_init`, `Tmin_med`, `early_fade` |

F1→F6 전부를 같은 CV fold로 평가했다. 중간 단계의 오차가 증가해도 비교를 중단하지 않았다. ID, target 및 `knee_cycle` 등 제외 feature는 predictor로 사용하지 않았다.

## 3. Model Comparison

모든 수치는 **Batch 1 Train의 CV MAPE (%)**이며 낮을수록 좋다. Hold-out/Test 결과는 이 비교나 feature 선택에 사용하지 않았다.

| Feature Set | Ridge | Random Forest | Gradient Boosting |
|---|---:|---:|---:|
| F1 | 10.063 | 9.627 | 10.584 |
| F2 | 10.656 | 10.484 | 11.340 |
| F3 | 10.416 | 10.215 | 10.952 |
| F4 | 10.770 | 10.461 | 11.007 |
| F5 | 10.729 | 10.186 | 11.487 |
| F6 | 11.778 | 10.323 | 11.424 |

원래 설계의 Linear Regression baseline은 `log_dQ_var` 하나인 F1만 평가했다.

| Model | Feature Set | Features | Batch 1 CV MAPE (%) |
|---|---|---|---:|
| Linear Regression | F1 | `log_dQ_var` | 10.059 |

설정은 변경하거나 탐색하지 않았다: Ridge `alpha=1.0`; Random Forest `n_estimators=100`, `n_jobs=1`, `random_state=42`; Gradient Boosting `n_estimators=100`, `learning_rate=0.1`, `max_depth=3`, `random_state=42`.

## 4. Selected Model

- **Selected Model: Ridge** — 최초 지정한 주 모델을 사용하는 설계 결정이다. 다른 모델의 CV 성능이 더 좋더라도 교체하지 않는다.
- **Selected Feature Set: F3**
- **Selected Features: `log_dQ_var`, `dQ_kurt`, `chargetime_med`**
- Feature 선택 기준: 현재 노트북에서 F3를 명시적으로 지정한다. CV 최솟값(`argmin`)이나 동률 처리 규칙으로 선택하지 않는다.

Ridge의 CV MAPE는 F1에서 **10.063%**, 지정된 F3에서 **10.416%**다. F1이 더 낮지만 F1~F6 표는 비교용이며, 최종 predictor는 지정된 F3의 3개 feature를 유지한다. F3가 CV 최적 조합이라는 주장은 하지 않는다.

Random Forest F1의 CV MAPE는 9.627%로 지정된 Ridge F3보다 낮지만, 이는 비교 결과일 뿐 **최종 모델 선택 기준이 아니다**. Ridge F3를 Train으로 학습한 Hold-out MAPE는 **8.344%**다. Hold-out 결과에 따른 재선택은 하지 않는다.

## 5. Performance Reporting

| 구분 | MAPE (%) | 비고 |
|---|---:|---|
| Train (Batch 1 CV) | 10.416 | Batch 1 CV 평균 |
| Valid (Batch 1 Hold-out) | 8.344 | Batch 1 Hold-out |
| Test (Batch 2) | 37.345 | Batch 2 Test |
| Gap (Train-Valid) | -2.072 | pp; Valid − CV |
| Gap (Valid-Test) | +29.000 | pp; Test − Valid |
| Gap (Target-Test) | +28.245 | pp; Test − 9.1% |

MAPE의 단위는 %, Gap의 단위는 **percentage points (pp)**이다.

- `Gap (Train-Valid) = valid_mape - train_cv_mape`
- `Gap (Valid-Test) = test_mape - valid_mape`
- `Gap (Target-Test) = test_mape - 9.1`

9.1%는 작업 명세에서 지정한 원논문 비교 기준으로 사용했다.

## 6. Minimal Interpretation

1. **Batch 1 CV와 Hold-out:** Hold-out MAPE가 CV보다 **2.072pp 낮다**. 이 차이에서는 Hold-out 오차 증가가 나타나지 않았다.
2. **Batch 1 Hold-out과 Batch 2 Test:** Test MAPE가 **29.000pp 높다**. 현재 결과에서 batch 간 일반화 성능 저하가 관측되었다. 그 원인을 규명하는 추가 분석은 수행하지 않았다.
3. **원논문 기준 9.1%와 Batch 2 Test:** Test MAPE는 **37.345%**로 기준보다 **28.245pp 높다**.

### 평가 이력 및 검증 범위

Batch 2는 앞선 구현에서도 평가된 데이터다. 따라서 이 결과를 **최초의 미관측 blind Test**로 주장하지 않는다. Ridge + F3는 비교 표의 최솟값이나 Hold-out/Test 결과로 재선택하지 않는다. 최종 Test 셀은 모델·feature 확정과 Hold-out 평가 이후에만 실행하며, `final_test_started` 가드로 한 커널에서 두 번 시작할 수 없게 한다. Test에는 Train에서 학습한 통계와 모델을 그대로 적용한다.

수정 전후 노트북의 코드 셀 11개를 각각 실제 데이터로 실행하여 결과를 대조했다. 반올림 전 CV MAPE는 `10.416115867916506`, Hold-out은 `8.344286316379039`, Test는 `37.34452704441607`로 정확히 일치했다. 비교 조합 19개의 모든 수치, Hold-out 예측값 10개와 Test 예측값 39개, cell ID와 순서, nominal 추정값, 성능 표도 허용 오차 없이 동일했다. Gap은 반올림 전 수치로 계산한 뒤 소수점 셋째 자리까지 표시한다. 저장된 노트북 출력은 그대로 보존했으며 Ruff lint/format 검사도 통과했다. 이 검증은 실제 실행 비교이며 별도 회귀 테스트 스위트를 추가한 것은 아니다. 동등성 확인을 위해 Batch 2를 수정 전후 각각 재평가했지만 그 결과로 설정을 바꾸지 않았다. 별도의 EDA·feature 생성·추가 모델·원인 분석은 수행하지 않았다.

여기까지의 결과를 한 문장으로 먼저 정리하면,

> **Ridge 모델이 Batch 1을 과적합해서 실패했다기보다, Batch 1에서 학습한 데이터 관계가 Batch 2에서 상당히 달라졌고, 그 결과 Batch 2의 Cycle Life 감소폭을 충분히 따라가지 못하면서 전반적인 과대예측이 발생한 것으로 해석하는 것이 가장 자연스럽습니다.**

아래 순서로 이해하면 전체 흐름이 잘 연결됩니다.

## 1. 출발점: Batch 1에서는 잘 되는데 Batch 2에서 무너졌다

현재 노트북에서 다시 확인된 성능은 대략 다음과 같습니다.

| 평가 | MAPE |
|---|---:|
| Batch 1 CV | 약 10% |
| Batch 1 Hold-out | 약 8% |
| Batch 2 Test | **37.35%** |

핵심은 `Train/CV → Valid`가 아닙니다.

```text
Batch 1 CV         약 10%
Batch 1 Hold-out    약 8%
```

둘 사이가 크게 벌어지지 않았습니다.

일반적인 과적합이라면 보통

```text
Train 매우 좋음
Valid 급격히 나빠짐
```

같은 형태를 예상합니다.

그런데 현재 모델은 Batch 1 내부의 새로운 Cell에서도 잘 작동합니다.

진짜 변화는:

```text
Batch 1 Hold-out    8.34%
        ↓
Batch 2            37.35%
```

입니다.

그래서 첫 번째 결론은:

> **문제의 중심은 Batch 1 내부 일반화가 아니라 Batch 1 → Batch 2 일반화다.**

입니다.

---

# 2. 몇 개의 이상치 때문에 MAPE가 높아진 것도 아니었다

Batch 2 Cell별 APE를 확인했을 때:

| 항목 | 결과 |
|---|---:|
| Mean APE | 37.35% |
| Median APE | 37.48% |
| 75% quantile | 47.43% |
| 20% 이상 오차 Cell | 31 / 39 |
| 50% 이상 오차 Cell | 10 / 39 |

특히 Mean과 Median이 거의 같습니다.

```text
Mean   37.35%
Median 37.48%
```

만약 몇 개의 이상치만 문제였다면 예를 들어

```text
Mean   37%
Median 10%
```

같은 형태가 나왔을 가능성이 큽니다.

하지만 그렇지 않았습니다.

따라서:

> **Batch 2의 일부 이상 Cell 때문이 아니라 Batch 2 전반의 문제다.**

라고 볼 수 있습니다.

---

# 3. 그리고 오차의 방향도 무작위가 아니었다

Batch 2의 39개 Cell 중:

```text
과대예측 : 34
과소예측 :  5
```

였습니다.

평균 Signed Error도:

```text
+33.89%
```

입니다.

예측 / 실제 비율의 중앙값은:

```text
1.375
```

즉 전형적인 Batch 2 Cell에서는 대략

\[
Prediction \approx Actual \times 1.375
\]

형태였습니다.

반면 Batch 1 Hold-out에서는:

```text
Mean Signed Error = -2.53%

과대예측 4
과소예측 6
```

으로 한쪽 방향의 bias가 거의 없었습니다.

그래서 중요한 두 번째 결론은:

> **Ridge 모델 자체가 항상 과대예측하는 것은 아니다. Batch 2에서만 체계적인 과대예측이 발생한다.**

입니다.

---

# 4. Batch 2에서는 입력 Feature 분포도 바뀌었다

모델이 실제로 사용하는 feature는:

```text
log_dQ_var
dQ_kurt
chargetime_med
```

세 개였습니다.

Batch 1 Train 기준으로 Standard Scaling한 뒤 비교했을 때 특히:

### `log_dQ_var`

```text
Test Mean   = +0.828σ
Test Median = +1.118σ

Batch 2 중 Train 범위 밖 = 28.2%
```

즉 Batch 2의 `log_dQ_var` 분포는 Batch 1보다 상당히 이동했습니다.

### `chargetime_med`

```text
Test Mean   = -0.738σ
Test Median = -0.714σ
```

Train 범위 밖 샘플은 없지만 Batch 1보다 낮은 영역으로 전체적으로 이동했습니다.

따라서 **Covariate Shift**, 즉

\[
P(X)_{\text{Batch1}}
\neq
P(X)_{\text{Batch2}}
\]

의 징후는 분명히 있습니다.

---

# 5. 그런데 재미있는 점: 이 Feature Shift가 직접 과대예측을 만든 것은 아니었다

Ridge 계수는:

| Feature | Ridge coef |
|---|---:|
| `log_dQ_var` | -0.066 |
| `dQ_kurt` | -0.003 |
| `chargetime_med` | +0.012 |

입니다.

각 feature의 Batch 이동이 Ridge 예측값에 얼마나 영향을 줬는지 분해했더니:

```text
log_dQ_var        -0.067
dQ_kurt           -0.001
chargetime_med    -0.013
--------------------------------
총 변화            -0.0804
```

가 나왔습니다.

즉 Batch 2 feature를 보고 Ridge도 실제로:

> "Batch 2의 수명은 Batch 1보다 낮을 것이다."

라고 판단하고 있었습니다.

그래서 Batch 1 Valid 대비 Batch 2 prediction도 감소했습니다.

이게 굉장히 중요한 포인트입니다.

문제는 모델이 **방향을 틀린 것이 아닙니다.**

---

# 6. 모델이 감소를 감지하기는 했는데 실제 감소폭이 훨씬 컸다

실제 Cycle Life를 비교했더니:

```text
Batch 1 Valid 실제 평균
895 cycle

Batch 2 실제 평균
566 cycle
```

정도였습니다.

비율로 보면:

\[
\frac{Batch2}{Batch1}
\approx 0.613
\]

즉 실제 Cycle Life가 약 **61% 수준까지 떨어졌습니다.**

그런데 Ridge prediction은:

```text
Batch 1 Valid 예측 평균
857 cycle

Batch 2 예측 평균
711 cycle
```

이고 비율은:

\[
0.831
\]

입니다.

쉽게 표현하면:

```text
실제
100 → 61

Ridge 예상
100 → 83
```

입니다.

따라서 현재 문제를 가장 직관적으로 표현하면:

> **Ridge도 Batch 2에서 수명이 줄어드는 것은 감지했지만, 실제 감소폭을 충분히 설명하지 못했다.**

입니다.

이 때문에 Batch 2에서 전반적인 과대예측이 발생했습니다.

---

# 7. 단순히 intercept만 틀린 것도 아니었다

Batch 2 residual을 봤을 때:

```text
Mean   = -0.119 log10
Median = -0.138 log10
Std    =  0.083
```

이었습니다.

음수 residual은

\[
Actual - Prediction < 0
\]

이므로 역시 과대예측입니다.

그런데 residual과 feature의 관계를 봤더니:

```text
log_dQ_var       -0.739
dQ_kurt          +0.713
chargetime_med   -0.812
```

처럼 강한 관계가 있었습니다.

즉 단순히

```text
모든 예측에 -100 cycle 하면 해결
```

하는 형태의 문제만은 아닙니다.

**feature 값에 따라 틀리는 정도도 달라지고 있습니다.**

---

# 8. 가장 특이했던 것은 `chargetime_med`

실제 target과 feature의 단순 correlation을 비교하면:

| Feature | Train B1 | Valid B1 | Test B2 |
|---|---:|---:|---:|
| `log_dQ_var` | -0.846 | -0.951 | -0.918 |
| `dQ_kurt` | +0.303 | +0.373 | +0.653 |
| `chargetime_med` | **+0.719** | **+0.824** | **-0.935** |

`log_dQ_var`는 상당히 안정적입니다.

```text
Batch 1 : 음의 관계
Batch 2 : 음의 관계
```

그런데 `chargetime_med`는:

```text
Batch 1 : 강한 양의 관계
Batch 2 : 강한 음의 관계
```

로 완전히 반전되었습니다.

처음에는 이것이 매우 강한 `Concept Shift` 신호처럼 보였습니다.

그런데 여기서 한 단계 더 확인한 것이 중요합니다.

---

# 9. Batch 2에서는 Feature끼리의 관계도 크게 바뀌었다

특히:

### `log_dQ_var ↔ chargetime_med`

```text
Train : -0.812
Valid : -0.809
Test  : +0.946
```

였습니다.

이것은 굉장히 큰 변화입니다.

Batch 1에서는:

```text
log_dQ_var ↑
chargetime_med ↓
```

방향으로 움직였는데,

Batch 2에서는:

```text
log_dQ_var ↑
chargetime_med ↑
```

으로 거의 같이 움직입니다.

`dQ_kurt ↔ chargetime_med`도:

```text
Train : +0.360
Test  : -0.557
```

으로 반전되었습니다.

반면:

```text
log_dQ_var ↔ dQ_kurt

Train : -0.393
Test  : -0.496
```

은 비교적 유지됩니다.

즉 **Batch 2에서 특히 `chargetime_med`를 중심으로 feature 구조가 달라졌습니다.**

---

# 10. 그래서 Batch 2에서 다중공선성도 크게 증가했다

VIF 결과가 이것을 다시 확인해줬습니다.

| Feature | Train B1 | Valid B1 | Test B2 |
|---|---:|---:|---:|
| `chargetime_med` | 2.96 | 2.98 | **10.61** |
| `log_dQ_var` | 3.04 | 3.24 | **9.70** |
| `dQ_kurt` | 1.19 | 1.15 | 1.47 |

Batch 1에서는:

```text
VIF ≈ 3
```

정도였는데 Batch 2에서는:

```text
chargetime_med ≈ 10.6
log_dQ_var     ≈ 9.7
```

입니다.

그 이유가 앞서 본:

```text
corr(log_dQ_var, chargetime_med) = 0.946
```

입니다.

그래서 Batch 2 데이터에 별도의 OLS를 진단 목적으로 적합했을 때:

```text
chargetime_med coefficient = -1.075
```

같이 매우 큰 계수가 나온 것입니다.

이 숫자를 그대로 물리적인 관계라고 믿으면 안 됩니다.

강한 다중공선성 때문에 계수가 불안정할 수 있기 때문입니다.

---

# 11. `chargetime_med`를 제거하면 문제가 해결되는가?

그래서 이 feature 하나가 문제인지 ablation도 해봤습니다.

| Feature Set | Valid MAPE | Test MAPE |
|---|---:|---:|
| Full | 8.34% | 37.35% |
| Without `chargetime_med` | **7.75%** | **36.38%** |
| Without `dQ_kurt` | 8.42% | 37.56% |
| Without `log_dQ_var` | 14.65% | **53.47%** |

결과가 중요합니다.

`chargetime_med`를 제거하면:

```text
37.35 → 36.38
```

밖에 개선되지 않습니다.

즉:

> **`chargetime_med`가 Batch Shift의 강한 신호이기는 하지만, 이 feature 하나 때문에 모델이 실패한 것은 아니다.**

입니다.

반대로 `log_dQ_var`를 제거하면:

```text
37.35 → 53.47
```

로 훨씬 악화됩니다.

따라서 `log_dQ_var`는 Batch 2에서도 여전히 의미 있는 정보입니다.

---

# 12. 지금까지 가장 합리적인 전체 해석

현재 상황을 데이터 흐름으로 표현하면 이렇습니다.

```text
Batch 1
──────────────────────────────

log_dQ_var
dQ_kurt
chargetime_med
      │
      │ 비교적 일정한 관계
      ▼
 Cycle Life

Ridge가 이 관계를 학습
      │
      ├─ CV 약 10%
      └─ Hold-out 약 8%
          → Batch 1 내부 일반화 OK


            ↓ Batch 변경


Batch 2
──────────────────────────────

① Feature 분포가 이동함

② 특히 chargetime_med와
   다른 feature 간 관계가 크게 변함

③ log_dQ_var ↔ chargetime_med
   -0.812 → +0.946

④ 실제 Cycle Life도 크게 감소
   Batch 1 대비 약 61%

⑤ Ridge도 감소 방향은 감지
   하지만 83% 정도로만 예상

                ↓

실제보다 Cycle Life를 높게 예측

                ↓

34 / 39 Cell 과대예측

                ↓

Test MAPE ≈ 37%
```

---

# 13. 그래서 이것을 단순한 "Ridge 성능 부족"이라고 부르면 정확하지 않습니다

현재 증거를 보면:

```text
Ridge가 나쁜 모델이다
```

보다는

> **Batch 1에서 학습된 Ridge의 관계식이 Batch 2의 데이터 생성 구조를 충분히 대표하지 못한다.**

가 더 정확합니다.

즉 모델링 관점에서는 크게 두 종류의 변화가 같이 관찰됩니다.

### Covariate shift

입력 feature의 분포:

\[
P(X)
\]

가 달라졌습니다.

`log_dQ_var`의 위치 이동과 Train 범위 이탈이 대표적인 증거입니다.

### Feature 관계 구조 변화

\[
P(X_1,X_2,X_3)
\]

도 달라졌습니다.

특히:

\[
corr(log\_dQ\_var, chargetime\_med)
\]

가

\[
-0.812 \rightarrow +0.946
\]

으로 변했습니다.

그리고 이 때문에 **feature와 target의 관계까지 동일하지 않을 가능성**도 강하게 의심됩니다.

다만 여기서 엄밀하게

> "Concept drift가 확정되었다."

라고까지 표현하는 것은 조금 과합니다.

Batch가 두 개뿐이고, Batch 2 내부 feature 간 다중공선성도 매우 높기 때문입니다.

그래서 보고서에서는

> **Batch 간 feature distribution 및 correlation structure의 뚜렷한 변화가 확인되었으며, 이로 인해 Batch 1에서 학습된 feature-target mapping의 Batch 2 일반화가 제한된 것으로 판단된다.**

정도가 가장 안전합니다.

---

# 14. 현재 Ridge에 대한 판단

지금까지 결과로 Ridge 모델 자체에 대해서는 이렇게 평가할 수 있습니다.

| 관점 | 판단 |
|---|---|
| Batch 1 학습 | 정상 |
| Batch 1 CV | 약 10%, 합리적 |
| Batch 1 Hold-out | 약 8%, 양호 |
| 과적합 징후 | 강하지 않음 |
| Batch 2 일반화 | **실패** |
| Batch 2 error 형태 | 전반적인 과대예측 |
| Outlier 문제 | 아님 |
| 특정 feature 하나의 문제 | 아님 |
| Batch shift 증거 | **강함** |
| `log_dQ_var` | 가장 안정적이고 중요한 feature |
| `chargetime_med` | Batch 간 구조 변화가 가장 큰 feature |

---

## 마지막으로 가장 중요한 해석

이번 분석에서 "`MAPE 37%라서 Ridge를 버려야 한다`"가 핵심 결론은 아닙니다.

더 중요한 발견은:

> **왜 37%가 되었는지 설명할 수 있게 되었다는 것**입니다.

Batch 1에서는 모델이 정상적으로 작동했습니다. Batch 2에서도 수명이 줄어드는 방향 자체는 감지했습니다. 하지만 Batch 2에서는 실제 수명이 훨씬 크게 감소했고, 동시에 feature들의 분포와 상호관계, 특히 `chargetime_med`를 둘러싼 구조가 Batch 1과 크게 달라졌습니다.

그래서 **Batch 1에서 학습한 하나의 고정된 선형 관계가 Batch 2의 변화를 충분히 표현하지 못했다**고 이해하면 지금까지의 모든 결과가 가장 자연스럽게 연결됩니다.

그리고 우리가 중간에 Test를 이용한 OLS·ablation 등을 수행한 것은 **새 모델을 고르기 위한 것이 아니라 원인을 설명하기 위한 diagnostic analysis**였습니다. 따라서 최종 모델 선택이나 성능 개선 단계에서는 다시 **Batch 1만을 이용해서 의사결정을 하고 Batch 2를 튜닝 대상으로 사용하지 않는 원칙**을 유지하는 것이 중요합니다.