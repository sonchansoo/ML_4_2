## 💻 Fundamentals of Computer Vision - 기말고사 예상 코드 문제 (MD 형식)

교수님의 강의 자료와 프로그래밍 과제 스타일을 분석하여, 기말고사에 출제될 가능성이 높은 코드 문제들을 GitHub에 올리기 좋은 Markdown 형식으로 정리했습니다. 라이브러리 활용보다는 **알고리즘의 근본 원리를 파이썬과 NumPy로 직접 구현**하는 능력을 평가하는 데 초점을 맞춘 문제들입니다.

---

### 🥇 출제 확률 매우 높음 (95% 이상)

#### 문제 1: 계산 그래프(Computational Graph)에서의 역전파(Backpropagation) 구현

**출제 근거:** 강의 자료(Lecture 7)에서 매우 상세하게 다루어졌으며, 프로그래밍 과제 7번의 핵심 주제입니다. 신경망 학습의 가장 근본적인 원리이므로 출제 1순위로 예상됩니다.

> **예상 문제**
>
> 아래는 간단한 연산을 수행하는 신경망의 순전파(forward pass) 코드입니다. 최종 출력값은 `L`입니다.
>
> 이 코드의 `backward()` 함수를 완성하여, 최종 출력 `L`에 대한 각 입력(`x`, `w`, `b`)의 그래디언트(gradient) `dL/dx`, `dL/dw`, `dL/db`를 계산하고 출력하는 코드를 작성하시오. (NumPy 외 다른 라이브러리 사용 불가)

```python
import numpy as np

# 입력값 및 초기 파라미터
x = -2
w = -3
b = 5

# Forward pass
mul_xw = x * w      # 곱셈 노드
add_b = mul_xw + b  # 덧셈 노드
L = max(0, add_b)   # ReLU 활성화 함수 노드

print(f"Forward pass 결과: L = {L}") # L = 11

# Backward pass
def backward():
    # 최종 출력 L에 대한 그래디언트는 1로 시작
    dL_dL = 1.0

    # --- 이 아래부터 코드를 작성하시오 ---

    # ReLU 노드의 역전파: dL/d(add_b) 계산
    # ReLU의 미분은 입력이 0보다 크면 1, 아니면 0
    dL_dadd_b = dL_dL * (1 if add_b > 0 else 0)

    # 덧셈 노드의 역전파: dL/d(mul_xw)와 dL/db 계산
    # 덧셈 게이트는 업스트림 그래디언트를 그대로 흘려보냄 (local gradient = 1)
    dL_dmul_xw = dL_dadd_b * 1.0
    dL_db = dL_dadd_b * 1.0

    # 곱셈 노드의 역전파: dL/dx와 dL/dw 계산
    # 곱셈 게이트는 입력을 서로 바꿔서 곱해줌
    dL_dx = dL_dmul_xw * w
    dL_dw = dL_dmul_xw * x

    return dL_dx, dL_dw, dL_db

# 결과 출력
dx, dw, db = backward()
print(f"dL/dx = {dx}") # 예상 결과: -3.0
print(f"dL/dw = {dw}") # 예상 결과: -2.0
print(f"dL/db = {db}") # 예상 결과: 1.0
```

**핵심 평가 요소:**
*   연쇄 법칙(Chain Rule)에 대한 정확한 이해
*   덧셈, 곱셈, ReLU 등 각 연산의 지역 그래디언트(local gradient) 계산 능력
*   업스트림(Upstream)과 지역(Local) 그래디언트를 곱하여 다운스트림(Downstream) 그래디언트를 계산하는 과정의 구현 능력

---

### 🥈 출제 확률 높음 (70% 이상)

#### 문제 2: 2D 컨볼루션(Convolution) 연산의 순전파 구현

**출제 근거:** CNN의 핵심 연산이며, 강의 자료(Lecture 8)와 과제 8번에서 NumPy를 이용한 직접 구현을 강조했습니다. FC 레이어와의 차이점을 이해하는지 평가하기 좋은 문제입니다.

> **예상 문제**
>
> 입력 이미지 행렬 `X` (5x5)와 컨볼루션 필터(커널) `W` (3x3)가 주어졌을 때, 스트라이드(stride) 1, 패딩(padding) 0으로 2D 컨볼루션 연산을 수행하는 `convolve_2d` 함수를 작성하시오. (NumPy 외 다른 라이브러리 사용 불가)

```python
import numpy as np

def convolve_2d(X, W):
    # 입력과 필터의 크기
    H, W_in = X.shape
    KH, KW = W.shape

    # 출력의 크기 계산 (stride=1, padding=0)
    OH = H - KH + 1
    OW = W_in - KW + 1
    output = np.zeros((OH, OW))

    # 컨볼루션 연산 수행 (슬라이딩 윈도우)
    for y in range(OH):
        for x in range(OW):
            # 입력 이미지에서 필터와 겹치는 영역(local region) 추출
            region = X[y:y+KH, x:x+KW]
            # Element-wise 곱셈 후 모든 값을 합산하여 출력의 한 픽셀 값을 계산
            output[y, x] = np.sum(region * W)
            
    return output

# 테스트
X = np.array([[1, 2, 3, 0, 1],
              [0, 1, 2, 3, 0],
              [1, 0, 1, 2, 3],
              [2, 3, 0, 1, 2],
              [3, 2, 1, 0, 1]])
# 세로 엣지를 검출하는 Sobel 필터와 유사
W = np.array([[1, 0, -1],
              [1, 0, -1],
              [1, 0, -1]])

feature_map = convolve_2d(X, W)
print(feature_map)
# 예상 결과:
# [[-2. -2. -2.]
#  [-2. -2.  0.]
#  [ 0.  0. -2.]]
```

**핵심 평가 요소:**
*   컨볼루션 연산의 '슬라이딩 윈도우' 방식에 대한 이해
*   출력 피처맵의 크기를 정확히 계산하는 능력
*   `for` 루프를 사용하여 필터를 이동시키고, 각 위치에서 요소별 곱셈 및 합산을 구현하는 능력

---

### 🥉 출제 확률 중간 (50% 이상)

#### 문제 3: k-최근접 이웃(k-NN) 분류기 구현

**출제 근거:** 가장 직관적인 분류 알고리즘으로, 강의 자료(Lecture 5)와 과제 5번에서 NumPy만을 사용한 Raw Pixel 기반 구현을 다루었습니다. 거리 계산, 정렬, 투표 등 프로그래밍 기본 요소를 평가하기에 적합합니다.

> **예상 문제**
>
> NumPy만을 사용하여 k-최근접 이웃(k-NN) 분류기의 `predict` 함수를 구현하시오. 이 함수는 다수의 학습 데이터(`X_train`, `y_train`)와 하나의 테스트 데이터(`x_test`)를 입력받아, `x_test`의 클래스를 예측하여 반환합니다.
>
> - 거리 척도는 유클리드 거리(Euclidean distance)를 사용합니다.
> - `k`는 가장 가까운 이웃의 수를 의미합니다.

```python
import numpy as np
from collections import Counter

def predict(X_train, y_train, x_test, k=3):
    # 1. 모든 학습 데이터와 테스트 데이터 사이의 유클리드 거리 계산
    distances = [np.sqrt(np.sum((x_train_sample - x_test)**2)) for x_train_sample in X_train]
    
    # 2. 계산된 거리를 기준으로 가장 가까운 k개의 인덱스 찾기
    k_nearest_indices = np.argsort(distances)[:k]
    
    # 3. k개의 가장 가까운 이웃의 레이블(클래스) 가져오기
    k_nearest_labels = [y_train[i] for i in k_nearest_indices]
    
    # 4. 다수결 투표(majority voting)를 통해 가장 빈번한 레이블 반환
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]

# 테스트
X_train = np.array([[1, 1], [1, 2], [2, 2], [6, 6], [6, 7], [7, 7]])
y_train = np.array([0, 0, 0, 1, 1, 1]) # Class 0과 Class 1
x_test = np.array([2, 3])

prediction = predict(X_train, y_train, x_test, k=3)
print(f"Test data [2, 3]의 예측 클래스: {prediction}") # 예상 결과: 0
```

**핵심 평가 요소:**
*   벡터 간 유클리드 거리를 NumPy로 계산하는 능력
*   `np.argsort`를 활용하여 가장 가까운 이웃을 찾는 능력
*   다수결 투표 로직 구현 능력

---

### 🏅 출제 가능성 있음 (30% 미만)

#### 문제 4: L2 정규화(Regularization)가 포함된 경사 하강법(Gradient Descent) 업데이트 구현

**출제 근거:** 과적합 방지를 위한 핵심 기법으로 강의 자료(Lecture 7)에서 비중 있게 다루어졌습니다. 경사 하강법 업데이트 식에 간단한 항을 추가하는 방식으로 구현되므로, 기존 로직을 수정하는 형태로 출제될 수 있습니다.

> **예상 문제**
>
> 신경망 학습 시 파라미터 `W`를 업데이트하는 경사 하강법 코드가 아래와 같이 주어졌습니다. 여기에 L2 정규화를 추가하는 코드를 완성하시오.
>
> - `L2_lambda`는 정규화 강도를 조절하는 하이퍼파라미터입니다.
> - L2 정규화가 적용된 손실 함수(Loss)는 `L_total = L_data + (L2_lambda / 2) * sum(W**2)` 입니다.
> - 이 손실 함수를 `W`에 대해 미분하면, 기존 그래디언트에 `L2_lambda * W` 항이 더해집니다.

```python
import numpy as np

# 초기 파라미터, 그래디언트, 하이퍼파라미터
W = np.array([[0.5, -0.2], [0.1, 0.8]])
grad_W_data = np.array([[0.1, 0.3], [-0.2, 0.15]]) # 데이터 손실로부터 계산된 그래디언트
learning_rate = 0.1
L2_lambda = 0.01

# --- 이 아래에 L2 정규화를 포함한 파라미터 업데이트 코드를 작성하시오 ---

# 1. L2 정규화 항의 그래디언트(lambda * W)를 기존 그래디언트에 더함
total_grad_W = grad_W_data + L2_lambda * W

# 2. 최종 그래디언트를 사용하여 파라미터 업데이트
W_updated = W - learning_rate * total_grad_W

print("업데이트된 파라미터 W:\n", W_updated)
```

**핵심 평가 요소:**
*   정규화가 손실 함수와 그래디언트에 미치는 영향에 대한 이해
*   기존 경사 하강법 업데이트 규칙을 수정하여 정규화 항을 적용하는 능력
