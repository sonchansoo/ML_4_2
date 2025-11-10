## 🔍 강의 요약
이번 강의에서는 신경망을 학습시키는 핵심 알고리즘인 **Backpropagation(역전파)**과 모델의 과적합(Overfitting)을 방지하고 일반화 성능을 높이는 기법인 **Regularization(정규화)**에 대해 학습합니다. Backpropagation이 어떻게 연쇄 법칙(Chain Rule)을 사용하여 복잡한 신경망의 모든 파라미터에 대한 Gradient를 효율적으로 계산하는지 계산 그래프(Computational Graph) 예시를 통해 단계별로 알아봅니다. 또한, 모델이 훈련 데이터에만 과도하게 최적화되는 Overfitting 문제점을 이해하고, 이를 해결하기 위해 Loss 함수에 패널티를 부과하는 L1, L2 Regularization과 Early Stopping 같은 기법들을 배웁니다.

## 💡 핵심 포인트
- **Backpropagation**은 신경망 학습의 핵심으로, 연쇄 법칙(Chain Rule)을 이용해 Loss로부터 모든 가중치(Weight)에 대한 Gradient를 효율적으로 계산하는 알고리즘입니다.
- 신경망 학습은 **Forward Pass** (출력 및 Loss 계산)와 **Backward Pass** (Gradient 계산)의 반복적인 과정으로 이루어집니다.
- 계산 그래프(Computational Graph)에서 Gradient는 **Upstream Gradient**, **Local Gradient**, **Downstream Gradient**의 곱으로 역방향으로 전파됩니다.
- 모델 학습의 진짜 목표는 훈련 데이터에 대한 완벽한 정확도가 아닌, 보지 못한 데이터에 대해서도 좋은 성능을 내는 **일반화(Generalization)**입니다.
- **Overfitting(과적합)**은 모델이 훈련 데이터를 너무 완벽하게 학습하여 노이즈까지 암기해 일반화 성능이 떨어지는 현상입니다.
- **Regularization(정규화)**은 Loss 함수에 모델의 복잡도에 대한 패널티 항을 추가하여 Overfitting을 방지하는 기법으로, L1과 L2 방식이 대표적입니다.

## 세부 내용 📚
### 1️⃣ 신경망 학습과 Gradient 계산의 필요성
- **목표**: 전체 훈련 데이터셋에 대한 **Total Loss(총손실)**를 최소화하는 최적의 파라미터(가중치 `W`)를 찾는 것입니다.
- **방법**: **Gradient Descent(경사 하강법)**를 사용합니다. 이 방법은 Loss를 가중치 `W`로 미분한 값, 즉 **Gradient(`∇L`)**를 계산하여 Gradient의 반대 방향으로 `W`를 조금씩 업데이트합니다.
- **문제점**: 현대 신경망은 수백만 개 이상의 파라미터를 가지며, 이 모든 파라미터에 대한 Gradient를 손으로 계산하는 것은 불가능합니다. 따라서 이를 자동화하고 효율적으로 계산할 방법이 필요합니다.

### 2️⃣ Backpropagation: Gradient 계산의 자동화
- **정의**: 출력층의 Loss에서부터 입력층까지 연쇄 법칙(Chain Rule)을 재귀적으로 적용하여 모든 파라미터에 대한 Gradient를 효율적으로 계산하는 알고리즘입니다.
- **Computational Graph**: 신경망의 연산을 노드(Node)로, 데이터의 흐름을 엣지(Edge)로 표현한 그래프입니다.
    - **Forward Pass**: 입력값이 그래프를 따라 순방향으로 흐르며 각 노드의 연산을 거쳐 최종 출력과 Loss를 계산하는 과정입니다.
    - **Backward Pass**: 최종 Loss에서 시작하여 그래프를 역방향으로 거슬러 올라가며 각 노드의 Gradient를 계산하고 전달하는 과정입니다.
- **Gradient의 흐름**:
    - **Upstream Gradient**: 출력 쪽(다음 계층)에서 현재 노드로 흘러 들어오는 Gradient입니다. (예: `∂L/∂z`)
    - **Local Gradient**: 현재 노드가 수행하는 연산 자체의 미분값입니다. (예: `∂z/∂x`)
    - **Downstream Gradient**: 현재 노드에서 입력 쪽(이전 계층)으로 전달될 Gradient로, `Upstream Gradient × Local Gradient`로 계산됩니다. (예: `∂L/∂x = ∂L/∂z * ∂z/∂x`)

### 3️⃣ 신경망에서의 Backpropagation 예시
간단한 선형 계층(Linear Layer)과 Sigmoid 활성화 함수로 구성된 네트워크의 학습 과정을 통해 Backpropagation을 이해할 수 있습니다.

1.  **Forward Pass**: 초기 가중치와 입력값으로 모든 중간값과 최종 출력(f)을 계산합니다.
2.  **Backward Pass**:
    - 최종 출력 노드에서 Gradient를 1로 초기화합니다 (`∂f/∂f = 1`).
    - 계산 그래프의 마지막 노드부터 역순으로 각 노드의 Local Gradient를 계산하고, 들어온 Upstream Gradient와 곱하여 Downstream Gradient를 계산해 다음 노드로 전달합니다.
    - 이 과정을 모든 가중치와 입력값에 도달할 때까지 반복합니다.
3.  **가중치 업데이트 (Gradient Descent)**: 계산된 Gradient를 사용하여 `W_new = W_old - learning_rate * ∇L` 공식으로 모든 가중치를 업데이트합니다.
4.  **반복**: 가중치가 업데이트되면 모든 중간값이 무효화되므로, 다시 1단계(Forward Pass)부터 시작하여 이 과정을 모델이 수렴할 때까지 반복합니다.

### 4️⃣ Overfitting과 일반화 (Generalization)
- **Overfitting (과적합)**: 모델이 훈련 데이터에 너무 완벽하게 적응하여 데이터의 노이즈까지 학습해버린 상태입니다. 이 경우, 훈련 데이터에 대한 Loss는 계속 감소하지만, 검증(Validation) 데이터에 대한 Loss는 오히려 증가하여 일반화 성능이 떨어집니다.
- **Generalization (일반화)**: 모델 학습의 궁극적인 목표는 훈련 데이터에 대한 성능이 아니라, 한 번도 보지 못한 새로운 데이터에 대해서도 좋은 성능을 보이는 것입니다.

### 5️⃣ Regularization: Overfitting 방지 기법
- **목표**: 모델이 너무 복잡해져 훈련 데이터에 과적합되는 것을 방지합니다.
- **Regularized Loss Function**: 기존의 Loss 함수(Data Loss)에 모델의 복잡도를 제어하는 **Regularization Term(정규화 항)**을 추가합니다.
    - `Total Loss = Data Loss + λ * R(W)`
    - **λ (Lambda)**: 정규화의 강도를 조절하는 하이퍼파라미터. 값이 클수록 더 강한 규제를 의미합니다.
- **종류**:
    - **L2 Regularization (Ridge)**: `R(W) = Σw²`. 모든 가중치의 제곱 합을 패널티로 부여합니다. 가중치 값을 전반적으로 작고 고르게 만들어 모델을 부드럽게 만듭니다.
    - **L1 Regularization (Lasso)**: `R(W) = Σ|w|`. 모든 가중치의 절댓값 합을 패널티로 부여합니다. 중요하지 않은 가중치를 0으로 만들어 모델을 더 희소(sparse)하게 만드는 효과가 있습니다 (Feature Selection).

### 6️⃣ 추가적인 일반화 기법: Early Stopping
- **Early Stopping (조기 종료)**: 훈련 과정에서 훈련 Loss와 함께 검증(Validation) Loss를 지속적으로 모니터링합니다.
- 훈련 Loss는 계속 감소하더라도, 검증 Loss가 더 이상 감소하지 않고 증가하기 시작하는 지점에서 훈련을 멈추는 기법입니다.
- 이는 모델이 과적합 단계에 들어서기 직전, 일반화 성능이 가장 좋은 지점에서 학습을 중단시키는 매우 효과적이고 간단한 방법입니다.

## 오늘의 연습문제 📝
- **문제 1**: 신경망 학습에서 Forward Pass와 Backward Pass의 역할은 각각 무엇이며, Backpropagation의 핵심이 되는 수학적 원리는 무엇인지 설명해 보세요.
- **문제 2**: L1 Regularization과 L2 Regularization의 차이점을 설명하고, 어떤 상황에서 L1 Regularization이 L2 Regularization보다 더 유리할 수 있는지 설명해 보세요.
- **문제 3**: 계산 그래프의 한 노드에서 "Upstream Gradient"는 무엇을 의미하며, 이 값이 해당 노드의 "Local Gradient"와 어떻게 결합되어 "Downstream Gradient"를 계산하는 데 사용되는지 설명해 보세요.
