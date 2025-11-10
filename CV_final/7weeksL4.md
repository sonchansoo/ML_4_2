## 🔍 강의 요약
이번 강의에서는 컴퓨터가 이미지를 단순한 픽셀의 집합이 아닌, 의미 있는 정보로 이해하는 과정의 첫 단계인 **Feature Detection & Matching**에 대해 학습합니다. Raw pixel 값이나 단순한 Edge만으로는 객체를 인식하기 어렵기 때문에, 이미지 내에서 독특하고 구별되는 지점인 **Feature(특징점)**, 특히 **Corner(코너)**를 찾는 방법을 배웁니다. Moravec, Harris, Shi-Tomasi와 같은 코너 검출 알고리즘의 원리를 이해하고, 검출된 특징점을 SIFT(Scale-Invariant Feature Transform)와 같은 기술을 사용하여 고유한 숫자 벡터(Descriptor)로 변환하는 과정을 살펴봅니다. 마지막으로, 이렇게 생성된 Descriptor들을 비교하여 서로 다른 이미지 간에 동일한 지점을 연결하는 **Feature Matching** 기술과 그 응용 분야에 대해 알아봅니다.

## 💡 핵심 포인트
- **Feature의 중요성**: Raw pixel이나 단순 Edge 정보만으로는 객체 인식이 불가능하며, 형태, 질감 등 고유한 패턴을 가진 **Feature**를 추출해야 합니다.
- **Corner Detection**: 모든 방향으로 픽셀 강도(intensity) 변화가 큰 지점인 **Corner**는 이미지의 중요한 특징점입니다. Moravec, Harris, Shi-Tomasi 등의 알고리즘이 이를 검출하는 데 사용됩니다.
- **Feature Description (SIFT)**: 검출된 특징점 주변의 지역적 패턴을 **HOG(Histogram of Oriented Gradients)**를 이용해 128차원의 숫자 벡터(Descriptor)로 변환합니다. 이는 이미지의 회전, 크기 변화에도 강인한 특징을 가집니다.
- **Feature Matching**: 두 이미지에서 추출된 Descriptor들을 **Euclidean distance**로 비교하고, **NNDR(Nearest Neighbor Distance Ratio) test**를 통해 신뢰도 높은 쌍을 찾아 연결합니다.
- **응용 분야**: Feature Matching 기술은 파노라마 이미지 제작(Image Stitching), 객체 인식(Object Recognition), 3D 복원 등 다양한 컴퓨터 비전 분야의 기반이 됩니다.

## 세부 내용 📚
### 1️⃣ From Pixels to Features 픽셀에서 특징점으로
- 컴퓨터는 이미지를 숫자 배열로 인식하기 때문에, 단순히 픽셀 값만으로는 고양이와 개를 구분할 수 없습니다.
- **Edge Detection**은 객체의 윤곽선을 찾아주지만, 그 자체만으로는 뾰족한 귀나 둥근 눈과 같은 구체적인 형태(Shape)를 알려주지 못합니다.
- **Feature**란 이미지 내에서 객체를 식별하는 데 도움을 주는 고유한 패턴이나 특성(모양, 질감 등)을 의미합니다. Edge가 '변화가 일어나는 위치'를 알려준다면, Feature는 '그 변화가 무엇을 의미하는지'를 알려줍니다.

### 2️⃣ Corner Detection 코너 검출
- **Corner**는 Edge의 방향이 급격하게 변하는 지점으로, 단순한 직선 Edge보다 훨씬 더 많은 정보를 담고 있어 중요한 특징점으로 간주됩니다.
- **핵심 원리**: 이미지 위에 작은 윈도우(window)를 놓고 모든 방향으로 조금씩 움직였을 때, 픽셀 값의 변화를 측정합니다.
    - **Flat region**: 모든 방향으로 변화가 거의 없음.
    - **Edge**: Edge 방향으로는 변화가 없지만, 수직 방향으로는 큰 변화가 있음.
    - **Corner**: 모든 방향으로 큰 변화가 발생함.
- **알고리즘**:
    - **Moravec Detector**: 4개의 주요 방향(상, 하, 좌, 우, 대각선)으로의 변화량(SSD) 중 최솟값을 코너 응답(corner response)으로 사용합니다.
    - **Harris & Shi-Tomasi Detector**: Moravec을 개선하여 모든 방향의 변화를 더 부드럽게 고려합니다. 수학적으로는 **Second-moment matrix (M)**의 고유값(eigenvalues, λ₁ , λ₂)을 분석하여 코너 여부를 판단합니다.
        - **Shi-Tomasi**: `R = min(λ₁, λ₂)`
        - **Harris**: `R = det(M) - k(trace(M))²`
- **한계**: Harris, Shi-Tomasi와 같은 초기 코너 검출기는 고정된 윈도우 크기를 사용하기 때문에 이미지의 크기가 변하면(줌인/줌아웃) 코너를 놓치는 **Scale-Invariant 하지 않다**는 단점이 있습니다.

### 3️⃣ Feature Description: SIFT & HOG 특징 기술
- 코너의 좌표만으로는 어떤 모양인지 알 수 없으므로, 코너 주변의 지역적 패턴을 설명하는 숫자 표현, 즉 **Descriptor**가 필요합니다.
- **SIFT (Scale-Invariant Feature Transform)**는 대표적인 특징 기술 방법입니다.
- **HOG (Histogram of Oriented Gradients)**: SIFT의 핵심 아이디어로, 특징점 주변 지역(local patch)의 그래디언트(gradient) 분포를 요약합니다.
    1.  **Gradient 계산**: 특징점 주변 16x16 픽셀 영역에서 각 픽셀의 그래디언트 크기(magnitude)와 방향(orientation)을 계산합니다.
    2.  **공간 분할**: 16x16 영역을 4x4 크기의 16개 하위 영역(subregion)으로 나눕니다.
    3.  **Histogram 생성**: 각 하위 영역마다 8개의 방향(bin)을 가진 그래디언트 방향 히스토그램을 만듭니다. 각 픽셀의 그래디언트 크기를 해당 방향 bin에 누적합니다.
    4.  **Descriptor 생성**: 16개 하위 영역의 히스토그램(16개 * 8 bin)을 모두 연결하여 **128차원의 SIFT Descriptor 벡터**를 완성합니다.
- **Rotation Invariance (회전 불변성)**: 이미지 회전에 대응하기 위해, 특징점 주변의 주된 그래디언트 방향(dominant orientation)을 찾고, 이 방향이 0도를 향하도록 Descriptor 전체를 회전시켜 정규화합니다.

### 4️⃣ Feature Matching 특징 매칭
- 두 개 이상의 이미지에서 **서로 대응하는 점(corresponding points)**을 찾는 과정입니다.
- **매칭 과정**:
    1.  각 이미지에서 Keypoint(e.g., corner)를 검출하고, 각 Keypoint에 대한 Descriptor(e.g., SIFT)를 계산합니다.
    2.  이미지 A의 한 Descriptor를 이미지 B의 모든 Descriptor와 비교하여 가장 유사한 것을 찾습니다. 유사도는 주로 **Euclidean distance**로 측정합니다.
- **NNDR (Nearest Neighbor Distance Ratio) Test**: 매칭의 신뢰도를 높이기 위한 기법입니다.
    - 이미지 A의 한 특징점(A₁)에 대해, 이미지 B에서 가장 가까운 특징점(B₁)과 두 번째로 가까운 특징점(B₂)을 찾습니다.
    - 두 거리의 비율 `(A₁과 B₁의 거리) / (A₁과 B₂의 거리)`을 계산합니다.
    - 이 비율이 특정 임계값(e.g., 0.75)보다 작을 때만 유효한 매칭으로 인정합니다. 이는 다른 후보들과 확연히 구별되는 유일한 매칭만을 선택하는 효과가 있습니다.

## 오늘의 연습문제 📝
- **문제 1**: 평평한(flat) 영역, 엣지(edge), 코너(corner)를 구별하기 위해 컴퓨터 비전 알고리즘이 사용하는 핵심 원리는 무엇인지 설명해 보세요.
- **문제 2**: SIFT Descriptor가 128차원의 벡터로 구성되는 과정을 HOG(Histogram of Oriented Gradients) 개념을 사용하여 단계별로 설명해 보세요.
- **문제 3**: 특징 매칭(Feature Matching)에서 모호한 매칭을 걸러내고 신뢰도를 높이기 위해 사용하는 **NNDR(Lowe's Ratio Test)**의 작동 원리에 대해 설명해 보세요.
