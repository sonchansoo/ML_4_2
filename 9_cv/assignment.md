## 🔍 강의 요약
이번 강의는 이미지 필터링(Image Filtering)에 대한 실습 과제를 안내합니다. 과제의 핵심 목표는 Python과 NumPy만을 사용하여 2D 컨볼루션(Convolution) 함수를 직접 구현하고, 이를 기반으로 다양한 엣지 검출(Edge Detection) 및 스무딩(Smoothing) 필터를 처음부터 만들어보는 것입니다. 또한, 이미지에 직접 노이즈를 추가하고 각 필터의 노이즈 제거 성능을 비교하며, 검출된 엣지를 원본 이미지 위에 시각화하는 방법을 다룹니다. 마지막으로, 직접 구현한 결과와 OpenCV의 내장 함수 결과를 비교하여 구현의 정확성을 검증하고 두 방식의 차이점을 분석합니다.

## 💡 핵심 포인트
- **2D 컨볼루션 함수 직접 구현**: NumPy를 사용하여 이미지와 커널(필터)을 입력받아 컨볼루션 연산을 수행하는 함수를 `from scratch`로 구현하는 것이 과제의 핵심입니다.
- **주요 필터 구현**: 구현한 컨볼루션 함수를 활용하여 Prewitt, Sobel, Laplacian 같은 엣지 검출 필터와 Average, Gaussian 같은 스무딩 필터를 직접 구현합니다.
- **비선형 필터 구현**: Median 필터는 컨볼루션으로 구현할 수 없으므로, 주변 픽셀의 중앙값(median)으로 대체하는 로직을 별도로 구현해야 합니다.
- **노이즈 처리 및 시각화**: 이미지에 Gaussian 노이즈와 Salt-and-pepper 노이즈를 추가한 후, 스무딩 필터들의 노이즈 제거 성능을 시각적으로 비교 분석합니다.
- **결과 검증 및 비교**: 수동으로 구현한 필터의 결과와 OpenCV에 내장된 동일한 기능의 함수(`cv2.Sobel`, `cv2.GaussianBlur` 등) 결과를 비교하여 구현을 검증하고 차이점을 고찰합니다.

## 세부 내용 📚
### 1️⃣ 과제 목표 및 데이터셋 (Goal and Dataset)
- **목표**:
    - 컨볼루션을 이용해 엣지 검출 및 스무딩 필터를 직접 구현합니다.
    - 수동 구현 결과와 OpenCV 내장 함수 결과를 비교합니다.
    - 노이즈가 있는 이미지에 필터를 적용하고, 컨투어(Contour)를 오버레이하여 결과를 시각화합니다.
- **데이터셋**:
    - 제공된 `Lab 02 image set.zip` 파일의 모든 이미지를 사용합니다.
    - 직접 촬영한 2장 이상의 자연스러운 사진(unfiltered, natural images)을 추가로 사용합니다.

### 2️⃣ 수동 구현 과제 (Manual Implementation Tasks)
- **2.1 2D Convolution 함수 구현**:
    - `convolution(image, kernel, padding='zero')` 형태의 함수를 구현합니다.
    - **입력**: 2D Grayscale 이미지(np.ndarray), 홀수 크기의 2D 필터 커널(np.ndarray).
    - **패딩**: 'zero' 패딩을 기본으로 지원하며, 'replicate'나 'reflect' 방식도 구현을 권장합니다. 출력 이미지 크기는 입력과 동일해야 합니다.
    - **연산**: 내부적으로는 부동 소수점(floating-point) 연산을 사용하고, 최종 결과는 `uint8` 타입으로 스케일링합니다.
- **2.2 필터 적용**:
    - **엣지 검출**: Prewitt, Sobel(x, y 방향 및 magnitude map), Laplacian(4 또는 8-neighborhood) 필터를 적용합니다.
    - **스무딩**: Average(3x3, 5x5, 7x7), Gaussian(3x3, 5x5) 필터를 적용합니다.
    - **Median 필터**: 컨볼루션이 아니므로, 각 픽셀을 주변 픽셀 값들의 **중앙값(median)**으로 대체하는 방식으로 별도 구현합니다.
- **2.3 노이즈 추가 및 제거**:
    - 직접 선택한 이미지에 Gaussian 노이즈와 Salt-and-pepper 노이즈를 추가합니다.
    - Average, Gaussian, Median 필터를 적용하여 노이즈를 제거하고, 각 노이즈 유형에 어떤 필터가 가장 효과적인지 시각적으로 비교합니다.
- **2.4 엣지 이진화 및 컨투어 오버레이**:
    - 엣지 검출 후, 임계값(threshold)을 적용하여 엣지 맵을 흑백(Binarization)으로 만듭니다.
    - `cv2.findContours`와 `cv2.drawContours` 함수를 사용하여 검출된 엣지를 원본 이미지 위에 겹쳐 그려(Overlay) 결과를 직관적으로 확인합니다.

### 3️⃣ OpenCV 비교 및 검증 (OpenCV Comparison and Verification)
- **OpenCV 함수 사용**: `cv2.filter2D`, `cv2.Sobel`, `cv2.blur`, `cv2.GaussianBlur`, `cv2.medianBlur` 등 OpenCV의 내장 함수를 사용하여 위와 동일한 필터링 작업을 반복합니다.
- **결과 비교**: 직접 구현한 결과와 OpenCV의 결과가 동일한 커널 크기에서 어떻게 다른지 비교하고, 그 차이점에 대해 분석하고 코멘트를 작성합니다.

### 4️⃣ 제출물 및 보고서 작성 (Deliverables and Report)
- **제출 항목**:
    1.  **소스 코드**: `.ipynb` 또는 `.py` 파일.
    2.  **결과 이미지**: 모든 처리 결과를 이미지 파일로 저장.
    3.  **보고서**: PDF 파일.
- **보고서 내용**:
    - **Methods**: 컨볼루션, 노이즈 추가, Median 필터 구현 방법에 대한 설명.
    - **Results**: 수동 구현과 OpenCV 구현 결과 비교, 스무딩 커널 크기별 결과, 노이즈 제거 결과, 컨투어 오버레이 결과 등을 포함.
    - **Discussion**: 필터 간 차이점, 스무딩의 효과, 노이즈 종류별 최적 필터, 수동 구현과 OpenCV 함수 간의 차이점 등에 대한 고찰.

## 오늘의 연습문제 📝
- **문제 1**: 이미지 필터링에서 **패딩(Padding)**이 왜 필요한지 설명하고, 'zero padding'의 크기를 결정하는 일반적인 규칙(k x k 커널에 대해)을 서술하시오.
- **문제 2**: **Median 필터**가 Average 필터나 Gaussian 필터와 달리 **비선형(Non-linear) 연산**으로 분류되는 이유는 무엇이며, 이로 인해 어떤 종류의 노이즈 제거에 특히 강점을 보이는지 설명하시오.
- **문제 3**: **Sobel 필터**를 사용하여 엣지를 검출하는 과정에서, 수평 엣지(Horizontal Edge)와 수직 엣지(Vertical Edge)를 각각 검출하기 위한 커널(kernel)의 형태는 어떻게 다른지 설명하고, 두 방향의 엣지 맵을 결합하여 최종 그래디언트 크기(Gradient Magnitude)를 구하는 두 가지 일반적인 방법을 서술하시오.
