## 💻 Fundamentals of Computer Vision - 기말고사 예상 코드 문제

교수님의 강의 자료와 제공해주신 GitHub 레포지토리(`cv_lab5` ~ `cv_lab8`)를 기반으로, 기말고사에 출제될 가능성이 높은 코드 문제와 정답을 Markdown 형식으로 정리했습니다. 모든 문제는 라이브러리에 의존하지 않고 **알고리즘의 핵심 원리를 Python과 NumPy로 직접 구현**하는 데 초점을 맞추었습니다.

---

### 🥇 출제 확률 매우 높음 (95% 이상)

#### 문제 1: 계산 그래프(Computational Graph)에서의 역전파(Backpropagation) 구현

**출제 근거:** `cv_lab7`의 `gate_graph.py`, `two_layer_scalar.py`에서 신경망의 학습 원리를 이해하기 위해 역전파를 직접 구현하는 것이 과제의 핵심이었습니다. 이는 교수님께서 가장 중요하게 생각하는 부분으로 보입니다.

> **문제**
>
> 아래는 `MulGate` (곱셈 게이트)의 순전파(`forward`) 코드입니다. 역전파(`backward`) 메소드를 완성하여, 업스트림 그래디언트(`upstream_grad`)를 입력 노드(`a`, `b`)로 올바르게 흘려보내는 코드를 작성하시오.

**Solution:**

```python
class Gate:
    # Gate의 기본 구조 (문제에 주어질 것으로 예상)
    def forward(self, *args): raise NotImplementedError
    def backward(self, out_node, upstream_grad): raise NotImplementedError

class Node:
    # Node의 기본 구조 (문제에 주어질 것으로 예상)
    def __init__(self, value, op=None, parents=()):
        self.value = value
        self.grad = 0.0  # 그래디언트는 0으로 초기화
        self.op = op
        self.parents = parents

# --- 아래 클래스의 backward 메소드를 완성하시오 ---

class MulGate(Gate):
    def forward(self, a, b):
        # 순전파 시 역전파에 필요한 입력 노드들을 인스턴스 변수로 저장
        self.a = a
        self.b = b
        # 새로운 출력 노드를 생성하여 반환
        return Node(a.value * b.value, op=self, parents=(a, b))

    def backward(self, out_node, upstream_grad):
        # out_node: 순전파 시 반환했던 출력 노드
        # upstream_grad: 출력 노드로부터 흘러들어온 그래디언트 (dL/d_out_node)
        
        # 연쇄 법칙(Chain Rule) 적용: (dL/da) = (dL/d_out_node) * (d_out_node/da)
        # 곱셈의 지역 그래디언트(local gradient)는 서로의 입력값입니다.
        # d(a*b)/da = b, d(a*b)/db = a
        
        # 입력 a에 대한 그래디언트 계산 및 누적
        self.a.grad += upstream_grad * self.b.value
        
        # 입력 b에 대한 그래디언트 계산 및 누적
        self.b.grad += upstream_grad * self.a.value

```

**핵심 평가 요소:**
*   곱셈 노드의 역전파 시, **입력 값을 서로 바꾸어** 업스트림 그래디언트와 곱한다는 핵심 규칙을 아는지 평가합니다.
*   `self.grad += ...` 와 같이 그래디언트를 **누적(accumulate)** 해야 하는 이유(하나의 노드가 여러 연산에 사용될 경우)를 이해하는지 평가합니다.

---

### 🥈 출제 확률 높음 (70% 이상)

#### 문제 2: 2D 컨볼루션(Convolution) 연산의 순전파 구현

**출제 근거:** `cv_lab8`에서 NumPy만을 사용하여 CNN의 핵심 레이어인 `Conv2D`를 직접 구현했습니다. 이는 CNN의 동작 원리를 이해하는 데 가장 기본적이고 중요한 부분입니다.

> **문제**
>
> 입력 이미지 `X` (높이 H, 너비 W)와 컨볼루션 필터 `W` (커널 높이 KH, 커널 너비 KW)가 주어졌을 때, 스트라이드(stride) 1, 패딩(padding) 0으로 2D 컨볼루션 연산을 수행하는 `convolve_2d` 함수를 작성하시오. (NumPy 외 다른 라이브러리 사용 불가)

**Solution:**

```python
import numpy as np

def convolve_2d(X, W):
    # 입력과 필터의 크기
    H, W_in = X.shape
    KH, KW = W.shape

    # 출력 피처맵의 크기 계산 (stride=1, padding=0)
    OH = H - KH + 1
    OW = W_in - KW + 1
    output = np.zeros((OH, OW))

    # 슬라이딩 윈도우 방식으로 필터를 이미지 위에서 이동
    for y in range(OH):
        for x in range(OW):
            # 현재 필터 위치에 해당하는 입력 이미지의 부분(region) 추출
            region = X[y:y+KH, x:x+KW]
            
            # 요소별 곱셈(element-wise multiplication) 후 모든 값을 합산
            convolution_sum = np.sum(region * W)
            
            # 결과를 출력 피처맵의 해당 위치에 저장
            output[y, x] = convolution_sum
            
    return output

# 테스트 코드
X = np.array([[1, 2, 3, 0, 1], [0, 1, 2, 3, 0], [1, 0, 1, 2, 3], [2, 3, 0, 1, 2], [3, 2, 1, 0, 1]])
W = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
feature_map = convolve_2d(X, W)
print(feature_map)```

**핵심 평가 요소:**
*   중첩 `for` 루프를 사용하여 필터를 이미지 전체에 걸쳐 이동시키는 슬라이딩 윈도우 개념을 코드로 표현할 수 있는지 평가합니다.
*   NumPy 슬라이싱(`X[y:y+KH, x:x+KW]`)을 사용하여 이미지의 특정 영역을 정확히 추출하는 능력.

---

### 🥉 출제 확률 중간 (50% 이상)

#### 문제 3: k-최근접 이웃(k-NN) 분류기 구현

**출제 근거:** `cv_lab5`에서 Raw Pixel 기반의 k-NN 분류기를 NumPy로 구현했습니다. 알고리즘이 직관적이고, 거리 계산, 정렬, 투표라는 명확한 단계로 구성되어 있어 학생의 기본적인 프로그래밍 및 알고리즘 구현 능력을 평가하기에 좋습니다.

> **문제**
>
> NumPy만을 사용하여 k-최근접 이웃(k-NN) 분류기의 `predict` 함수를 구현하시오. 이 함수는 다수의 학습 데이터(`X_train`, `y_train`)와 하나의 테스트 데이터(`x_test`)를 입력받아, `x_test`의 클래스를 예측하여 반환합니다. 거리 척도는 유클리드 거리를 사용합니다.

**Solution:**

```python
import numpy as np
from collections import Counter

def predict(X_train, y_train, x_test, k=3):
    # 1. 모든 학습 데이터와 테스트 데이터 사이의 유클리드 거리 계산
    # 브로드캐스팅을 활용하여 각 학습 데이터 샘플과 테스트 데이터의 차이를 구함
    distances = np.sqrt(np.sum((X_train - x_test)**2, axis=1))
    
    # 2. 거리가 가장 가까운 k개의 학습 데이터 인덱스 찾기
    k_nearest_indices = np.argsort(distances)[:k]
    
    # 3. 해당 인덱스의 레이블(클래스)들을 가져오기
    k_nearest_labels = y_train[k_nearest_indices]
    
    # 4. 다수결 투표로 최종 클래스 예측
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]

# 테스트 코드
X_train = np.array([[1, 1], [1, 2], [2, 2], [6, 6], [6, 7], [7, 7]])
y_train = np.array([0, 0, 0, 1, 1, 1])
x_test = np.array([2, 3])
prediction = predict(X_train, y_train, x_test, k=3)
print(f"Test data [2, 3]의 예측 클래스: {prediction}")
```

**핵심 평가 요소:**
*   NumPy 브로드캐스팅을 활용하여 모든 학습 데이터와의 거리를 효율적으로 계산하는 능력.
*   `np.argsort()`를 사용하여 정렬 후 상위 k개를 선택하는 방법.

---

### 🏅 출제 가능성 있음 (30% 미만)

#### 문제 4: 옵티마이저(Optimizer) 업데이트 규칙 구현 (Momentum)

**출제 근거:** `cv_lab6`에서 다양한 옵티마이저(Momentum, RMSProp, Adam)를 구현하고 비교했습니다. Momentum은 SGD를 개선하는 가장 기본적인 아이디어이므로, 그 업데이트 규칙을 코드로 구현하는 문제가 나올 수 있습니다.

> **문제**
>
> 아래는 SGD 파라미터 업데이트 코드입니다. 이를 **Momentum SGD** 방식으로 수정하시오.
>
> - `v_W`는 이전 그래디언트 정보를 누적하는 **속도(velocity)** 변수입니다.
> - `beta`는 관성(momentum)의 크기를 조절하는 하이퍼파라미터입니다.

**Solution:**

```python
import numpy as np

# 초기 파라미터, 그래디언트, 하이퍼파라미터 (주어진 값)
W = np.array([[0.5, -0.2], [0.1, 0.8]])
grad_W = np.array([[0.1, 0.3], [-0.2, 0.15]])
learning_rate = 0.1
beta = 0.9
v_W = np.zeros_like(W) # 속도 변수, 0으로 초기화

# --- Momentum SGD 업데이트 코드 ---

# 1. 속도(v) 업데이트: 이전 속도에 관성을 적용하고, 현재 그래디언트를 더함
# (1-beta)는 보통 생략되기도 하지만, 여기서는 강의자료의 일반적인 형태를 따름
v_W = beta * v_W + (1 - beta) * grad_W

# 2. 업데이트된 속도를 사용하여 파라미터 업데이트
W_updated = W - learning_rate * v_W

print("업데이트된 파라미터 W:\n", W_updated)
```

**핵심 평가 요소:**
*   Momentum이 단순히 현재 그래디언트만 사용하는 것이 아니라, **이전 업데이트 방향을 일정 부분 유지**하려는 개념(속도)을 코드에 반영할 수 있는지 평가합니다.
