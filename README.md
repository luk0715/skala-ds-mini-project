# ESS 배터리 수명 예측

초기 100 cycle 이내의 배터리 특성으로 수명 (cycle life)을 예측한다. Batch 1에서 학습한 모델을 Batch 2에 적용해
다른 배치에서도 예측 성능을 유지하는지 평가하고, 성능이 떨어진 원인과 ESS 운영에 활용할 때의 한계를 살펴봤다.
상세 결과와 검증 이력은 [모델 보고서](model_report.md)에 정리했다.

## 프로젝트 개요

- 데이터셋 : MIT-Stanford Battery Dataset (Severson et al., Nature Energy 2019)
- 학습 데이터 : Batch 1 (2017-05-12)
- 평가 데이터 : Batch 2 (2018-02-20)
- 태스크 : 회귀 (Regression), 배터리 수명 예측
- 예측 대상 : `log10(cycle_life)`로 학습한다. 실제값과 예측값을 `10 ** value`로 cycle 단위로 복원한 뒤 MAPE (%)를 계산한다.

| 구분             | Cell 수 | 역할                            |
|------------------|--------:|---------------------------------|
| Batch 1 Train    |      36 | 피처 비교, 5-fold CV, 최종 학습 |
| Batch 1 Hold-out |      10 | 피처 확정 후 내부 검증          |
| Batch 2 Test     |      39 | 확정된 Ridge의 배치 간 평가     |

수명 레이블이 유효한 양수이고 ΔQ 데이터가 있는 셀을 사용한다. Batch 1은 `random_state=42`로 고정해 80/20으로 나눈다.

## 파일 구조

```text
.
├── assets           # 첨부 이미지
├── data             # 모델 데이터
├── docs             # 기타 문서
├── notebooks        
│   ├── eda.ipynb    # eda 수행
│   └── model.ipynb  # 모델 분석 및 평가
├── src              # helper functions
├── pyproject.toml
├── README.md
└── uv.lock
```

- `notebooks/model.ipynb` : 피처·모델 비교, 최종 평가 및 오류 진단
- `model_report.md` : 모델링 결과와 검증 이력
- `data/` : Batch 1·2 MAT 데이터 파일

## 환경 설정

Python 3.12 이상과 `uv`를 사용한다.

```bash
git clone https://github.com/luk0715/skala-ds-mini-project
cd skala-ds-mini-project
uv sync
```

[링크](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)에서 아래 파일을 `data/`에 준비한 뒤
`notebooks/model.ipynb`를 실행한다.

- `2017-05-12_batchdata_updated_struct_errorcorrect.mat`
- `2018-02-20_batchdata_updated_struct_errorcorrect.mat`

## EDA

- Cycle Life 분포
    - 수명의 평균 Batch 1 Hold-out 895.300 cycle, Batch 2 Test 565.744 cycle이다.
    - Test 평균 수명은 Hold-out의 63.2%다.

- 열화 곡선 분석
    - 초기 용량 변화를 나타내는 `QD_max_minus_init`, `early_fade`를 포함해 피처 조합을 비교했다.
    - 특정 시점에 급격한 열화가 진행되지만, 이는 모두 100cycle 이후에 등장하므로 feature 대상에서는 제외했다.

- ΔQ (V) 곡선 분석
    - ΔQ 관련 피처 `log_dQ_var`, `dQ_kurt`를 사용했다.
    - `log_dQ_var`와 로그 수명의 상관은 Train −0.846, Hold-out −0.951, Test −0.918이다.
    - 세 데이터 분할에서 모두 음의 상관을 보였다.

- 충전 속도 (C-rate)와 수명의 관계
    - 충전 시간 피처 `chargetime_med`와 로그 수명의 상관은 Train +0.719, Hold-out +0.824, Test −0.935다.
    - Batch 1에서는 양의 상관, Batch 2에서는 음의 상관을 보였다. 이 상관만으로 C-rate가 수명에 미치는 영향을 설명할 수는 없다.

- 배치 간 피처 분포 변화
    - Train에서 학습한 전처리를 Test에 적용했을 때 평균 z-score는 `log_dQ_var` +0.828σ, `chargetime_med` −0.738σ다.
    - Test의 `log_dQ_var` 중 28.2%가 Train 범위를 벗어났다.

## Modeling

### 피처 엔지니어링 전략

상관관계를 분석하며 피처를 선택했다. 최종적으로는 상관계수가 높은 피처들과 도메인적으로 의미가 있을 것 같은 피처들을 후보로 선택했다.

| 구분 | 피처                                                    | 근거                                                                                  |
|------|---------------------------------------------------------|---------------------------------------------------------------------------------------|
| 핵심 | log_dQ_var                                              | 상관계수 -0.89로 매우 높음                                                            |
| 보조 | dQ_kurt                                                 | 상관 0.32로 약하지만 batch별 부호 일관 / log_dQ_var와 상관계수 낮음                   |
| 후보 | chargetime_med, QD_max_minus_init, Tmin_med, early_fade | batch마다 상관 부호 다름 / 도메인 특성에 맞게 의미 있어보이며 상관계수 높은 피처 선별 |

모델 학습시에는 피처 개수를 하나씩 늘려가며 테스트 해보았다.

| Feature Set | 추가 피처           | 누적 개수 |
|-------------|---------------------|----------:|
| F1          | `log_dQ_var`        |         1 |
| F2          | `dQ_kurt`           |         2 |
| F3          | `chargetime_med`    |         3 |
| F4          | `QD_max_minus_init` |         4 |
| F5          | `Tmin_med`          |         5 |
| F6          | `early_fade`        |         6 |

최종 피처 조합인 F3는 ΔQ 분산·첨도와 충전 시간을 사용한다.

너무 적은 개수의 feature 만으로는 모든 batch를 설명하기 어려울 것이라 생각했고,

### 모델 선택 및 근거

- 최종 모델 : Ridge (`alpha=1.0`) + F3
- 후보 모델 : Ridge,
- 선택 근거 : 피처와 데이터 수가 적어, 과적합과 다중공선성을 줄이려는 목적으로 Ridge를 사용했다.
- 비교 모델 : Linear Regression (baseline), Random Forest/Gradient Boosting (선형 관계 확인)

F1 기준으로 모델을 비교하였다.

| Model             | MAPE (%) |
|-------------------|----------|
| Ridge             | 10.063   |
| Linear Regression | 10.059   |
| Random Forest     | 9.627    |
| Gradient Boosting | 10.584   |

- 비교 결과 : 완벽한 선형 관계는 아니지만 선형성이 강한 경향이 나타난다고 알 수 있었다. 혹은 일부 비선형 데이터로 인한 오차일 것이라 생각한다.

## 성능 결과

- 최종 모델 : Ridge (`alpha=1.0`) + F3

| 구분                     | MAPE (%) | 비고                    |
|--------------------------|---------:|-------------------------|
| Train (Batch 1 CV)       |   10.416 | 5-fold 평가 MAPE의 평균 |
| Valid (Batch 1 Hold-out) |    8.344 | 내부 검증               |
| Test (Batch 2)           |   37.345 | 배치 간 평가            |
| Gap (Train-Valid)        |   −2.072 | pp; Valid − CV          |
| Gap (Valid-Test)         |  +29.000 | pp; Test − Valid        |
| Gap (Target-Test)        |  +28.245 | pp; Test − 9.1%         |

Batch 2를 이용한 최종 Test MAPE 는 37.34%이다.
원논문의 9.1%와는 매우 큰 차이를 보여준다.

## 오류 분석

- Test의 실제 평균 수명은 Hold-out의 63.2%였지만, 예측 평균은 83.0%였다.
- 모델 자체의 과적합 문제는 아니라고 판단했다.
- 몇 개의 이상치 때문에 MAPE가 높아진 것도 아니었다. Batch 2 Cell별 APE를 확인했을 때 Mean과 Median이 거의 같아 Batch 2 전반의 문제라고 생각했다.
- 따라서 Batch 2를 이용한 추가적인 분석을 진행 하였다.
- Feature Shift도 직접 과대예측을 만든 것은 아니었다. 각 feature의 Batch 이동이 Ridge 예측값에 얼마나 영향을 줬는지 분해했더니:

```text
log_dQ_var        -0.067
dQ_kurt           -0.001
chargetime_med    -0.013
--------------------------------
총 변화            -0.0804
```

방향을 틀린 것은 아니라고 생각했다. 하지만 모델의 예측보다 실제 감소폭이 훨씬 컸다.

- 또 발견한 원인으로는 피처 분포와 피처–수명 관계의 변화를 의심할 수 있다. Batch 2에서 `chargetime_med`는 수명과의 상관 방향이 바뀌었다.
  Test에서는 VIF가 10.61로 높아졌다. 다만 이 결과만으로 성능 저하의 원인을 확정할 수는 없었다.
- 그 후 피처를 하나씩 뺀 뒤 Batch 1 Train으로 다시 학습했다 (ablation).
  `chargetime_med`를 빼면 Test MAPE는 36.38%, `log_dQ_var`를 빼면 53.47%였다.
  충전 시간 피처를 제거하는 것만으로는 Test 오차가 크게 줄지 않았다. 이 비교는 진단에만 사용했고, 최종 Ridge + F3는 바꾸지 않았다.

최종적으로는 모델을 학습하기 위한 데이터 자체가 문제라는 생각이 든다. Batch 1과 Batch 2 데이터의 분포는 매우 다르다. Batch 2에 수명이 낮은 배터리의 데이터들이 몰려있었기 때문이다.

원논문에서 제시한 방법처럼 Batch 1, Batch 2의 데이터의 일부를 각각 합쳐서 Train을 진행했다면 더 좋은 결과가 나왔을 것 같다.

전체 분석과정은 `notebooks/model.ipynb`에 남아있다.

## ESS 도메인 해석

- 초기 cycle 특성으로 셀의 예상 수명을 비교하고, 점검 순서나 교체 계획을 정할 때 보조 지표로 활용하는 방안이다.
- 다만 Batch 2에서는 수명을 실제보다 길게 예측한 셀이 많았다. 예측값만으로 교체 시점을 정하거나 안전을 판단해서는 안 된다.
- 검증 범위는 Batch 1·2의 셀 단위 수명 예측이다. ESS 운영 데이터나 팩 단위 성능은 검증하지 않았다.
  초기 100 cycle 이내의 피처로 전체 수명을 예측하므로, 임의 시점의 잔여수명 (RUL)을 추정하는 모델과도 다르다.
- 배포 전에는 실제 ESS 운전 조건에서 외부 검증을 거쳐야 한다. 예측 불확실성을 평가하고 데이터 분포 변화를 감시하는 방법도 필요하다.
  수명을 과대예측할 때의 위험을 고려해 보수적인 판단 기준을 마련해야 한다.

## 참고문헌

- Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4,
  383–391.

## 팀 구성

luk0715 (박태준)
