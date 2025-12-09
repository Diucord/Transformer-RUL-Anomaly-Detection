# Transformer-based Predictive Maintenance System  
### Reconstruction Error 기반 이상탐지 + Warning Level + RUL(남은 수명) 예측

This project leverages the **Transformer model** to analyze bearing vibration data (IMS Bearing Dataset), and utilizes the **Reconstruction Error** for **Fault Precursor Detection (Warning)** and **Remaining Useful Life (RUL)** estimation, establishing a complete **Predictive Maintenance (PdM)** system.

---
## 1. Core Features | 주요 기능

### 1.1. Transformer-based Time Series Forecasting | Transformer 기반 시계열 예측
**Architecture Components:**
* FeatureExtractor (CNN) for multi-scale feature learning.
* Trend Layer and Seasonality Layer for time-series decomposition.
* Stable Block-Sparse Self-Attention mechanism.
* Robust Transformer Encoder (Residual Connection + LayerNorm).

### 1.2. Reconstruction Error-based Anomaly Detection | 재구성 오차 기반 이상 탐지
**Detection Logic:**
* Anomaly is triggered when the reconstruction error between the predicted value and the actual value rises significantly.
* Implements a 3-Tiered Alert System: Normal (정상) → Warning (경고) → Critical (심각)

### 1.3. Automatic Failure Point Estimation | 고장 시점 자동 추정
**Definition:**
* The point of failure is automatically recognized as the first occurrence of the Critical alert level.

### 1.4. RUL (Remaining Useful Life) Calculation | 남은 수명 계산
**RUL Logic:**
* The calculation is precisely based on the actual time intervals (5/10 minutes) recorded in the IMS Dataset.
* The output is the "Remaining time until failure (in minutes)" from the detected critical point.

### 1.5. RMS-based Statistical Analysis (Academic Rigor) | RMS 기반 통계 분석 (학술적 정당성)
**Statistical Validation:**
* RMS Trend Analysis
* Kolmogorov-Smirnov Normality Test  
* Levene Variance Stability Test (Homogeneity of Variance)

### 1.6. Performance Comparison | 성능 비교
**Baseline:**
* Compares the Transformer's forecasting and anomaly detection performance against a Moving Average (MA) statistical baseline model.

### 1.7. Modular Structure | 유지 보수 용이한 모듈 구조
**Design:**
* Project is organized into modular scripts (data_preprocessing.py, model_transformer.py, training.py, evaluation.py) for easy maintenance and expansion.

---
## 2. Dataset Details | 데이터셋 상세 정보
**IMS Bearing Dataset (Case Western Reserve University)**

### 2.1. Test 1 Details (1st_test):
* **Initial Channels:** 8 channels (reduced to 4 aggregated channels).
* **Intervals:** Files 1-43 are 5 minutes; files 44 onwards are 10 minutes.
* **Failure Mode:** Inner Race Defect (Bearing 3), Roller Element Defect (Bearing 4)

### 2.2. Test 2 & 3 Details (2nd_test, 3rd_test):
* **Channels:** 4 basic channels
* **Intervals:** All files are 10 minutes
* **Failure Modes:** Outer Race Failure (2nd Test - Bearing 1), Outer Race Failure (3rd Test - Bearing 3)

**Note:** These specific time rules are accurately applied in the RUL calculation and RMS trend plotting for precise prognostics.

---
## 3. System Architecture | 시스템 구조
```bash
    Raw IMS Data
        ⬇
    [Preprocessing]             (data_preprocessing.py)
    - 8→4 Channel Aggregation (1st_test)
    - Downsampling & Scaling
    - .npy Save (Efficient Storage)
        ⬇
    [Statistical Analysis]      (data_analysis.py)
    - RMS Trend, KS Test, Levene Test
        ⬇
    [Model Training]            (training.py, model_transformer.py)
    - AdvancedAutoInformerModel (Trend + Seasonality + Attention)
        ⬇
    [Evaluation & Prognostics]  (evaluation.py)
    - Reconstruction Error Calculation (Anomaly Score)
    - Warning Level Classification
    - Failure Point Detection (First Critical)
    - RUL Calculation (Time-axis based)
        ⬇
    [Visualization + Reporting] (main.py, result_output.py)
```

---
## 4. Key Mathematical Concepts | 핵심 수리적 개념

### 4.1. Reconstruction Error (MSE)
Reconstruction error is calculated between the actual vibration signal ($x_{t}$) and the model’s predicted next timestep ($\hat{x}_{t}$), across $C$ channels:
```bash
\[
\text{error}_t = \frac{1}{C} \sum_{c=1}^C (x_{t,c} - \hat{x}_{t,c})^2
\]
```
This error naturally serves as an anomaly score as it increases when the bearing begins to degrade.

### 4.2. Warning Level Classification  
A moving-average window smooths the error to avoid jittery false alarms and defines the alert levels based on a determined threshold ($\tau$).
```bash
- \text{avg} < \tau \rightarrow \text{Normal}
- \tau \le \text{avg} < 1.5\tau \rightarrow \text{Warning}
- \text{avg} \ge 1.5\tau \rightarrow \text{Critical}
```
This classification system mirrors standard industrial predictive maintenance practices.

### 4.3. RUL (Remaining Useful Life) Calculation  
Based on the IMS dataset’s real-time intervals (5/10 minutes):
```bash
\[
\text{RUL} = \text{Time}_{\text{end}} - \text{Time}_{\text{failure}}
\]
```
This provides the remaining time until failure in minutes.

---
## 5. Installation & Execution | 설치 및 실행

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/Diucord/Predictive-Maintenance-Transformer.git
cd Predictive-Maintenance-Transformer

# 2. Install dependencies
pip install -r requirements.txt
```

### Virtual Environment Configuration (Recommended)
```bash
    # 1) Create conda environment
    conda create -n pm_transformer python=3.10 -y

    # 2) Activate
    conda activate pm_transformer

    # 3) Install packages
    pip install numpy==1.26.4 pandas==2.1.4 matplotlib==3.8.2 scipy==1.11.4 scikit-learn==1.3.2 tqdm==4.66.1

    # 4) Install PyTorch (CPU version)
    pip install torch==2.1.0

    # If you have NVIDIA GPU, use:
    # pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

### Execution
프로젝트의 전체 파이프라인(데이터 전처리, 모델 학습, RUL 계산 및 시각화)을 실행하는 메인 명령어입니다.
```bash
python main.py
```

---
## 6. Output Examples | 출력 예시
The execution generates detailed plots and reports in the ./predictive_results directory.

1) RMS Trend Plot
- 시간축(분) 기반 RMS 상승 패턴을 통해 베어링 열화 추세를 확인합니다.

2) Prediction vs Actuals
- Transformer 모델의 시계열 예측 성능을 시각적으로 검증합니다.

3) Reconstruction Error + Warning Level
- Normal(0) / Warning(1) / Critical(2)

4) Critical 경보 최초 등장 시점을 고장 시점으로 탐지하여 표시합니다.

---
## Author
- Seyoon Oh
- Korea University : School of Industrial & Management Engineering
- Email : osy7336@korea.ac.kr