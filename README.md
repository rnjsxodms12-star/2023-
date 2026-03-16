# 전력 소비량 예측 모델 개발 실험 로그

## 프로젝트 개요

본 프로젝트는 건물별 전력 소비량 데이터를 기반으로 미래 전력 소비량을 예측하는 모델을 구축하는 것을 목표로 한다.

주어진 데이터는 다음과 같이 구성된다.

* 건물별 전력 소비 데이터
* 기상 데이터 (기온, 풍속, 습도 등)
* 건물 정보 데이터 (연면적, 건물 유형 등)

모델은 **XGBoost 기반 회귀 모델**을 사용하였으며,
실험 과정에서 다양한 Feature Engineering 및 데이터 정제 기법을 단계적으로 적용하였다.

평가지표는 **SMAPE**를 사용하였다.

---

# 실험 로그

## Baseline

### base.ipynb

가장 처음 구축한 파이프라인.

**Score**

| pub  | pri  | val |
| ---- | ---- | --- |
| 9.04 | 10.8 | -   |

**Features**

```
건물번호
기온(C)
강수량(mm)
풍속(m/s)
습도(%)
DI
month
day
hour
dayofweek
is_holiday
```

---

# 1. 파이프라인 안정화

## team_base

**변경사항**

* 결측치 처리

  * 강수량, 일사, 일조량 → `0`
  * 풍속, 습도, 기온 → **선형 보간**
* Validation 방식 개선

  * **TimeSplit 적용**
* 평가 방식 통일

  * **SMAPE metric**

**Score**

| pub  | pri   | val  |
| ---- | ----- | ---- |
| 9.55 | 10.46 | 13.3 |

---

## team_base_n

**변경사항**

* 공휴일 정의 수정
* 주말 포함한 휴일 변수 생성

```python
train['is_holiday'] = train['일시'].dt.strftime('%m-%d').isin(['06-06', '08-15']).astype(int)
train['is_holiday'] = ((train['is_holiday'] == 1) | (train['dayofweek'] >= 5)).astype(int)
```

**Score**

| pub  | pri   | val  |
| ---- | ----- | ---- |
| 9.32 | 10.37 | 12.5 |

---

# 2. 파생 피처 실험

## team_ex1

### 체감온도(Apparent Temperature) 추가

```python
def calculate_at(temp, humid, wind):
    e = (humid / 100) * 6.105 * np.exp(17.27 * temp / (237.7 + temp))
    return 1.04 * temp + 0.2 * e - 0.65 * wind - 2.7
```

**Score**

| pub  | pri   | val  |
| ---- | ----- | ---- |
| 9.33 | 10.30 | 13.1 |

*Error 발견으로 이후 수정 진행*

---

## team_ex2

### 데이터 정합성 수정

수정 사항

* test 강수량 0 처리
* test에 없는 컬럼 강제 생성

  * 일조
  * 일사
* test 기상 데이터 선형보간
* 6월1일 선거일 공휴일 추가

기준 파이프라인을 **team_base_n으로 회귀 후 수정 적용**

**Score**

| pub  | pri   | val   |
| ---- | ----- | ----- |
| 9.21 | 10.12 | 12.41 |

---

## team_ex3

체감온도만 추가한 실험

**Score**

| pub  | pri   | val   |
| ---- | ----- | ----- |
| 9.25 | 10.20 | 12.22 |

---

# 3. 건물 정보 Feature 추가

## team_ex4 (첫 성능 돌파 파이프라인)

건물 메타데이터 병합

```python
train = pd.merge(train, building_info, on='건물번호', how='left')
test = pd.merge(test, building_info, on='건물번호', how='left')
```

추가 Feature

* 건물유형
* 연면적
* 냉방면적
* 태양광용량
* ESS저장용량
* PCS용량

XGBoost 설정

```
enable_categorical=True
```

**Score**

| pub  | pri  | val   |
| ---- | ---- | ----- |
| 8.58 | 9.75 | 10.68 |

**첫 baseline 돌파 파이프라인**

---

## team_ex5

team_ex4 + 체감온도(AT)

**Score**

| pub  | pri  | val   |
| ---- | ---- | ----- |
| 8.49 | 9.67 | 10.58 |

---

## team_ex6

일별 기온 파생 변수 추가

```
평균기온
최대기온
```

**Score**

| pub  | pri  | val   |
| ---- | ---- | ----- |
| 8.71 | 9.73 | 10.61 |

---

## team_ex7

team_ex3 기반 + 평균기온 / 최대기온

**Score**

| pub  | pri   | val   |
| ---- | ----- | ----- |
| 9.65 | 10.73 | 12.18 |

평균기온 / 최대기온은 **정보 중복 가능성 존재**

---

# 4. 시간 Feature Engineering

## team_ex8

주말 변수 추가

```python
is_weekend
```

Score 변화 없음

---

## team_ex9

시간 주기성 추가

```
hour_sin
hour_cos
```

**Score**

| pub  | pri  | val   |
| ---- | ---- | ----- |
| 8.30 | 9.28 | 10.61 |

---

# 5. 데이터 정제

## team_ex10

### 임시 휴무일 데이터 제거

건물별 **일별 총 전력량 / 건물 평소 전력량** 비율 계산

기준

```
전력비율 < 0.3
```

→ 임시 휴무로 판단하여 데이터 제거

**Score**

| pub  | pri  | val   |
| ---- | ---- | ----- |
| 8.35 | 9.20 | 10.04 |

---

# 6. 건물 성향 분석

## team_ex11

### 건물 사용 패턴 클러스터링

사용 Feature

* 평균 전력 사용량
* 기온-전력 상관관계
* 주말/주중 사용 패턴

클러스터링

```
KMeans (n_clusters=4)
```

**Score**

| pub  | pri  | val  |
| ---- | ---- | ---- |
| 8.45 | 9.38 | 9.99 |

---

## team_ex13

클러스터 수 변경

```
n_clusters = 5
```

**Score**

| pub  | pri  | val  |
| ---- | ---- | ---- |
| 8.20 | 8.91 | 9.88 |

---

## team_ex14

클러스터 수 변경

```
n_clusters = 6
```

**Score**

| pub  | pri  | val  |
| ---- | ---- | ---- |
| 8.36 | 9.11 | 9.98 |

**최적 클러스터 수 → 5**

---

# 7. Feature 정리

## team_ex15

`day` feature 제거

이유

* 트리 모델에서 과적합 발생
* 날짜 자체는 의미 없는 노이즈 가능성

**Score**

| pub  | pri  | val  |
| ---- | ---- | ---- |
| 7.99 | 8.46 | 9.17 |

현재 **Best Score**

---

## team_ex16

`hour` feature 제거

**Score**

| pub  | pri  | val  |
| ---- | ---- | ---- |
| 8.25 | 8.73 | 9.23 |

---

## team_ex17

카테고리 처리 강화

```python
train['건물번호'] = train['건물번호'].astype('category')
train['건물유형'] = train['건물유형'].astype('category')
```

XGBoost

```
enable_categorical=True
```

---

# 최종 파이프라인 특징

주요 Feature

```
건물번호
건물유형
연면적
냉방면적
태양광용량
ESS저장용량
PCS용량

기온
풍속
습도
강수량
DI
AT

month
hour_sin
hour_cos
dayofweek
is_holiday
is_weekend

building_cluster
```

---

# Best Result

| pub      | pri      | val      |
| -------- | -------- | -------- |
| **7.99** | **8.46** | **9.17** |

---

# 향후 개선 방향

* 건물별 모델 분리 (Building-wise modeling)
* 클러스터 기반 모델 학습
* Lag Feature 추가
* Temperature Interaction Feature
* Hyperparameter tuning

---

# 참고

Feature importance 확인

```python
model.feature_importances_
xgb.plot_importance(model)
```

