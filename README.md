# 전력 소비량 예측 모델 개발 보고서

## 1. 프로젝트 개요

본 프로젝트는 건물별 전력 소비량 데이터를 기반으로 미래 전력 소비량을 예측하는 모델을 구축하는 것을 목표로 한다.

주어진 데이터는 건물 정보와 기상 데이터로 구성되어 있으며, 이를 활용하여 **머신러닝 기반 회귀 모델(XGBoost)** 을 학습하였다.

실험 과정에서는 다음을 중심으로 성능 개선을 진행하였다.

* Feature Engineering
* 건물 정보 활용
* 시간 특성 반영
* 데이터 정제
* 건물 패턴 기반 클러스터링

모델 평가는 **SMAPE (Symmetric Mean Absolute Percentage Error)** 를 사용하였다.

---

# 2. 데이터 구성

## 2.1 데이터 종류

### 기상 데이터

* 기온(C)
* 강수량(mm)
* 풍속(m/s)
* 습도(%)

### 시간 정보

* month
* day
* hour
* dayofweek
* 공휴일 여부

### 건물 정보

추가 데이터 `building_info.csv` 활용

* 건물유형
* 연면적(m2)
* 냉방면적(m2)
* 태양광용량(kW)
* ESS저장용량(kWh)
* PCS용량(kW)

---

# 3. Validation 전략

시계열 데이터 특성을 고려하여 **TimeSeriesSplit 교차검증**을 사용하였다.

* Fold 수: 5
* Metric: SMAPE

```text
Train → 과거 데이터
Validation → 미래 데이터
```

이를 통해 **데이터 누수(Data Leakage)** 를 방지하였다.

---

# 4. Baseline 모델

초기 파이프라인 기반 모델

### 사용 Feature

```
건물번호
기온
강수량
풍속
습도
DI
month
day
hour
dayofweek
is_holiday
```

### 결과

| Metric  | Score |
| ------- | ----- |
| Public  | 9.04  |
| Private | 10.80 |

---

# 5. Baseline 개선

## team_base

결측치 처리 및 TimeSeriesSplit 도입

### 결측치 처리

* 강수량 / 일사 / 일조 → 0
* 풍속 / 습도 / 기온 → 선형 보간

### Validation 추가

* TimeSeriesSplit
* Fold별 SMAPE 측정

### 결과

| Metric     | Score |
| ---------- | ----- |
| Public     | 9.55  |
| Private    | 10.46 |
| Validation | 13.35 |

---

## team_base_n

공휴일 Feature 개선

기존

```
06-06 현충일
```

수정

```
06-06
08-15
주말 포함
```

```python
train['is_holiday'] = train['일시'].dt.strftime('%m-%d').isin(['06-06', '08-15']).astype(int)
train['is_holiday'] = ((train['is_holiday'] == 1) | (train['dayofweek'] >= 5)).astype(int)
```

### 결과

| Metric     | Score |
| ---------- | ----- |
| Public     | 9.32  |
| Private    | 10.37 |
| Validation | 12.57 |

---

# 6. Feature Engineering 실험

## team_ex1

체감온도 (Apparent Temperature) 추가

```python
AT = 1.04*temp + 0.2*e - 0.65*wind - 2.7
```

### 결과

Validation SMAPE
13.11

성능 개선 효과는 제한적이었다.

---

## team_ex2

데이터 오류 수정

* test 강수량 0 처리
* test 일사/일조 컬럼 생성
* test 기온/풍속/습도 선형 보간
* 6월 1일 선거일 공휴일 추가

### 결과

| Validation |
| ---------- |
| 12.41      |

---

## team_ex3

체감온도만 단독 적용

Validation
12.22

---

# 7. Building Information 활용

## team_ex4

건물 정보 데이터 추가

추가 Feature

```
건물유형
연면적
냉방면적
태양광용량
ESS저장용량
PCS용량
```

### 결과

| Metric     | Score |
| ---------- | ----- |
| Public     | 8.58  |
| Private    | 9.75  |
| Validation | 10.68 |

이 시점에서 **Baseline 성능을 크게 개선하였다.**

---

## team_ex5

체감온도 + building_info

| Metric     | Score |
| ---------- | ----- |
| Public     | 8.49  |
| Private    | 9.67  |
| Validation | 10.58 |

---

# 8. 추가 Feature 실험

## team_ex6

건물별 날짜 단위 온도 통계

```
평균기온
최대기온
```

Validation
10.61

정보 중복 가능성이 높다고 판단.

---

## team_ex8

주말 Feature 추가

```
is_weekend
```

Validation
10.61

성능 변화 거의 없음.

---

## team_ex9

시간 주기성 Feature

```
hour_sin
hour_cos
```

### 결과

| Metric     | Score |
| ---------- | ----- |
| Public     | 8.30  |
| Private    | 9.28  |
| Validation | 10.39 |

시간 주기성을 반영하여 성능이 개선되었다.

---

# 9. 데이터 정제

## team_ex10

임시 휴무일 데이터 제거

건물별 하루 전력 소비량이 **평소 대비 30% 이하인 날**을 휴무일로 판단하여 제거

### 결과

| Validation |
| ---------- |
| 10.04      |

노이즈 제거 효과 확인.

---

# 10. 건물 패턴 클러스터링

## team_ex11

건물 사용 패턴 기반 KMeans 클러스터링

사용 Feature

```
평균 전력 사용량
기온-전력 상관계수
주말/주중 전력 사용 비율
```

Cluster 수 = 4

### 결과

| Validation |
| ---------- |
| 9.99       |

---

## team_ex13

Cluster 수 조정

```
n_clusters = 5
```

### 결과

| Metric     | Score |
| ---------- | ----- |
| Public     | 8.20  |
| Private    | 8.91  |
| Validation | 9.88  |

---

# 11. Feature Selection

## team_ex15

day feature 제거

가설
day는 전력 패턴과 직접적인 관계가 낮고 노이즈 가능성이 존재

### 결과

| Validation |
| ---------- |
| **9.17**   |

성능 개선 확인.

---

## team_ex16

hour feature 제거 실험

### 결과

| Validation |
| ---------- |
| 9.23       |

성능 소폭 하락.

따라서 **hour feature는 중요한 변수로 판단된다.**

---

# 12. 결론

본 실험에서 다음과 같은 결과를 확인하였다.

### 성능 향상에 가장 기여한 요소

1. building_info 데이터 활용
2. 시간 주기성 Feature
3. 건물 패턴 클러스터링
4. 노이즈 데이터 제거

### 주요 개선 흐름

```
Baseline
→ building_info 추가
→ 시간 주기성 feature
→ 휴무 데이터 제거
→ 건물 클러스터링
→ feature selection
```

---

# 13. 향후 개선 방향

* Feature importance 기반 선택
* LightGBM / CatBoost 비교
* 모델 앙상블
* 시계열 전용 모델 적용

---

# 14. 최종 성능

Best 실험

team_ex15

| Metric     | Score    |
| ---------- | -------- |
| Public     | ?        |
| Private    | ?        |
| Validation | **9.17** |
