## 🔍 강의 요약
이번 강의에서는 컴퓨터 비전 및 이미지 처리를 위한 핵심 오픈소스 라이브러리인 **OpenCV**에 대해 학습합니다. OpenCV의 주요 기능과 특징을 소개하고, Python 환경에서 OpenCV를 설치하고 가상 환경(Anaconda, venv)을 사용하는 이유와 방법을 알아봅니다. 또한, 이미지 파일을 읽고 화면에 표시하는 기본 I/O 작업부터 시작하여, 이미지가 NumPy 배열로 어떻게 표현되는지, 그리고 픽셀 값에 직접 접근하고 수정하는 방법을 실습합니다. Grayscale 변환, 이진화(Thresholding), 이미지 크기 조절, 뒤집기, 회전, 히스토그램 계산 등 필수적인 이미지 처리 기법들을 OpenCV 함수와 순수 Python/NumPy 코드로 각각 구현하며 그 원리를 깊이 있게 이해합니다.

## 💡 핵심 포인트
- **OpenCV란?**: 컴퓨터 비전과 이미지 처리를 위해 설계된 강력한 오픈소스 라이브러리로, C++로 최적화되어 높은 성능을 자랑하며 Python, Java 등 다양한 언어를 지원합니다.
- **가상 환경의 중요성**: 프로젝트별로 독립된 패키지 환경을 제공하여 라이브러리 버전 충돌을 방지하고, 연구 및 과제의 재현성(Reproducibility)을 보장하는 필수적인 도구입니다.
- **이미지는 NumPy 배열**: OpenCV에서 이미지는 **(세로, 가로, 채널)** 형태의 3차원 NumPy 배열로 다루어집니다. 채널 순서는 RGB가 아닌 **BGR(Blue, Green, Red)** 순서를 기본으로 사용합니다.
- **핵심 이미지 처리 함수**:
    - **`cv2.imread()` / `cv2.imshow()`**: 이미지 읽기/쓰기 및 화면 표시.
    - **`cv2.cvtColor()`**: BGR, GRAY 등 색 공간 변환.
    - **`cv2.threshold()`**: 이미지를 흑백으로 이진화.
    - **`cv2.resize()`**: 이미지 크기 조절 (확대/축소).
    - **`cv2.flip()` / `cv2.rotate()`**: 이미지 뒤집기 및 회전.
    - **`cv2.calcHist()`**: 이미지의 픽셀 강도 분포를 계산.
- **OpenCV vs. NumPy**: OpenCV의 편리한 내장 함수를 사용하는 것과 별개로, 순수 NumPy를 이용해 동일한 기능을 직접 구현해보는 것은 이미지 처리의 기본 원리를 이해하는 데 매우 중요합니다.

## 세부 내용 📚
### 1️⃣ OpenCV 소개 및 환경 설정 (Introduction to OpenCV & Environment Setup)
- **OpenCV (Open Source Computer Vision Library)**: 누구나 무료로 사용할 수 있으며, Windows, macOS, Linux 등 다양한 플랫폼을 지원합니다. 의료 영상, 자율주행차, CCTV 등 광범위한 분야에서 활용됩니다.
- **설치 및 확인**:
    - **설치**: `pip install opencv-python` 명령어로 간단히 설치할 수 있습니다.
    - **확인**: Python 스크립트에서 `import cv2` 후 `print(cv2.__version__)`으로 설치된 버전을 확인할 수 있습니다.
- **가상 환경 (Virtual Environments)**:
    - **필요성**: Python 라이브러리들은 버전 간 충돌이 잦기 때문에(예: OpenCV 4.9는 최신 TensorFlow와 잘 맞지만, 특정 PyTorch 버전과는 충돌 가능), 프로젝트마다 패키지를 격리하여 안정성을 높입니다.
    - **종류**:
        - **Anaconda**: 데이터 과학 패키지가 미리 설치되어 있고 GUI(Anaconda Navigator)를 제공하여 초보자에게 편리합니다.
        - **venv**: Python 3.3 이상 버전에 내장되어 있어 가볍고, 서버 환경이나 최소한의 설정에 적합합니다.

### 2️⃣ 이미지 입출력 및 시각화 (Image I/O and Visualization)
- **이미지 읽기**: `img = cv2.imread("example.jpg")` 함수를 사용하여 이미지 파일을 NumPy 배열로 불러옵니다.
- **이미지 표시**:
    - **로컬 환경**: `cv2.imshow("Window Title", img)`를 사용하면 별도의 GUI 창이 나타납니다. `cv2.waitKey(0)`로 키 입력을 대기하고, `cv2.destroyAllWindows()`로 창을 닫습니다.
    - **Colab/서버 환경**: GUI 지원이 없으므로 `matplotlib.pyplot` 라이브러리를 사용해야 합니다.
- **⚠️ BGR vs. RGB**: OpenCV는 이미지를 **BGR** 순서로 읽지만, `matplotlib`은 **RGB** 순서로 이미지를 표시합니다. 따라서 Colab 등에서 올바른 색상으로 이미지를 보려면 `img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`를 통해 색상 채널 순서를 변환해주어야 합니다.

### 3️⃣ 이미지 데이터 접근 및 조작 (Image Data Access and Manipulation)
- **이미지 구조**: 이미지는 `(height, width, channels)` 형태의 NumPy 배열입니다. 예를 들어 `(480, 640, 3)`은 세로 480픽셀, 가로 640픽셀, 3개의 컬러 채널(BGR)을 의미합니다.
- **픽셀 접근**: `pixel = img[100, 200]`와 같이 `[행, 열]` 인덱싱으로 특정 위치의 픽셀 값([B, G, R])에 접근할 수 있습니다.
- **픽셀 값 수정**:
    - 단일 픽셀: `img[100, 200] = [255, 255, 255]` (흰색으로 변경)
    - 특정 영역(ROI): `img[50:100, 50:100] = [0, 0, 255]` (50x50 영역을 빨간색으로 변경)
- **이미지 저장**: `cv2.imwrite("output.jpg", img)` 함수를 사용하여 처리된 이미지를 파일로 저장할 수 있습니다.

### 4️⃣ 기본 이미지 처리 기법 (Basic Image Processing Techniques in OpenCV)
- **Grayscale 변환**: `gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`를 사용합니다. 결과 이미지는 채널 차원이 없는 `(height, width)` 형태가 됩니다.
- **이진화 (Thresholding)**: `cv2.threshold(src, thresh, maxval, type)` 함수를 사용합니다. `thresh` 값보다 크면 `maxval`로, 작으면 0으로 픽셀 값을 변환합니다.
- **이미지 크기 조절 (Resize)**: `cv2.resize(src, dsize, fx, fy, interpolation)`
    - `dsize=(width, height)`: 절대 크기 지정.
    - `fx`, `fy`: 가로, 세로 비율 지정.
    - `interpolation`: 보간법 지정 (`cv2.INTER_NEAREST`, `cv2.INTER_LINEAR` 등).
- **이미지 뒤집기 (Flipping)**: `cv2.flip(img, flipCode)`
    - `flipCode = 0`: 상하 반전
    - `flipCode > 0` (주로 1): 좌우 반전
    - `flipCode < 0` (주로 -1): 상하좌우 반전
- **이미지 회전 (Rotation)**:
    - **90도 단위**: `cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)` 등 간단한 상수를 사용합니다.
    - **임의 각도**: `cv2.getRotationMatrix2D()`로 회전 행렬을 구한 뒤, `cv2.warpAffine()` 함수를 적용하여 회전시킵니다.
- **히스토그램 (Histogram)**: `hist = cv2.calcHist([img], [channel_idx], None, [bins], [range])` 함수로 픽셀 값의 분포를 계산하고, `matplotlib`으로 시각화할 수 있습니다.

## 오늘의 연습문제 📝
- **문제 1**: OpenCV에서 컬러 이미지는 어떤 자료구조로 표현되며, 컬러 이미지의 채널 순서는 무엇인가요? Matplotlib을 사용하여 이미지를 올바르게 표시하기 위해 어떤 함수를 사용해야 하는지 설명하시오.
- **문제 2**: `cv2.resize()` 함수를 사용하여 이미지를 2배 확대할 때, `interpolation` 인자로 `cv2.INTER_NEAREST`와 `cv2.INTER_LINEAR`를 사용했을 때의 코드 차이점과, 두 방법의 결과물이 시각적으로 어떻게 다를지 예측하여 서술하시오.
- **문제 3**: 컬러 이미지(`example.jpg`)를 로드한 후, (1) 이미지를 Grayscale로 변환하고, (2) 변환된 이미지에 임계값(threshold) 128을 적용하여 흑백(Binary) 이미지로 만드는 OpenCV 코드의 주요 단계를 서술하시오.
