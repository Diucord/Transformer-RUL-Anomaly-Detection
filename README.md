# Transformer-Based Forecasting Framework for Bearing Degradation Analysis, Failure Onset Detection, and RUL Estimation
### A Fully Reproducible Implementation on the IMS Bearing Dataset

This project leverages the **Transformer model** to analyze bearing vibration data (IMS Bearing Dataset), and utilizes the **Reconstruction Error** for **Fault Precursor Detection (Warning)** and **Remaining Useful Life (RUL)** estimation, establishing a complete **Predictive Maintenance (PdM)** system.

---
## 0. Overview
This repository implements a Transformer-based forecasting framework for:

- Anomaly detection via reconstruction error
- Failure onset detection (consecutive exceedance criterion)
- Remaining Useful Life (RUL) estimation
- Cross-domain bearing degradation analysis

The system performs:

1. Data preprocessing (channel reduction, downsampling, normalization)
2. Transformer forecasting
3. Reconstruction error analysis
4. Normal-region extraction via variance spike
5. Adaptive thresholds based on Gaussian statistics
6. Multi-level warning system (None/Low/Medium/High)
7. Failure onset & RUL prediction
8. Full cross-test evaluation (9 combinations)

---
## 1. Research Problem & Methodology Summary

Traditional vibration-based PHM systems rely on fixed thresholds, RMS monitoring, or handcrafted features. These approaches fail to capture:

* early-stage microscopic degradation,
* non-stationarity,
* noise variance differences across environments,
* cross-domain generalization behavior.

This project addresses these limitations through:

---
## 2. Dataset Details 
**IMS Bearing Dataset (Case Western Reserve University)**

### 2.1 Test 1 Details (1st_test):
* **Initial Channels:** 8 channels (reduced to 4 aggregated channels).
* **Intervals:** Files 1-43 are 5 minutes; files 44 onwards are 10 minutes.
* **Failure Mode:** Inner Race Defect (Bearing 3), Roller Element Defect (Bearing 4)

### 2.2 Test 2 & 3 Details (2nd_test, 3rd_test):
* **Channels:** 4 basic channels
* **Intervals:** All files are 10 minutes
* **Failure Modes:** Outer Race Failure (2nd Test - Bearing 1), Outer Race Failure (3rd Test - Bearing 3)

**Note:** These specific time rules are accurately applied in the RUL calculation and RMS trend plotting for precise prognostics.

---
## 3. System Architecture 
```bash
    Transformer-RUL-Anomaly-Detection/
    │
    ├── main.py                     # End-to-end experimental pipeline
    ├── training.py                 # Model training loop (MSE loss, AdamW, scheduler)
    ├── evaluation.py               # Warning system, RUL estimation, plots
    ├── model_transformer.py        # Advanced AutoInformer-style Transformer model
    ├── data_preprocessing.py       # Channel reduction, downsampling, sliding windows
    ├── data_analysis.py            # RMS, K-S test, Levene test, baseline comparison
    ├── utils.py                    # Time-axis generation, smoothing, warning logic
    │
    ├── config.json                 # All experiment settings (thresholds, paths, etc.)
    ├── analysis_results/           # Auto-generated evaluation results
    └── README.md                   # (This document)
```

### 4. Preprocessing Pipeline
**Channel Reduction**

reduced = (ch1 + ch2) / 2

**Downsampling**

x_ds = x[::10] # 20kHz → 2kHz

**Sliding Window**

X = [x1 ... x(T−1)]

y = xT

**Normalization**

Min–max or z-score per test condition.


---
## 5. Transformer Forecasting Model

### Objective

x(1:t-1) → x̂_t

### Reconstruction Error
Reconstruction error is calculated between the actual vibration signal ($x_{t}$) and the model’s predicted next timestep ($\hat{x}_{t}$), across $C$ channels:

\[
\text{error}_t = \frac{1}{C} \sum_{c=1}^C (x_{t,c} - \hat{x}_{t,c})^2
\]

This error naturally serves as an anomaly score as it increases when the bearing begins to degrade.

### Normal Region Detection (Rolling Variance)

v_t = Var( error(t-w : t) )

Variance spike point defines the end of the normal region.

### Adaptive Threshold Calculation

**Gaussian fit:**

μ, σ = GaussianFit(error_normal)

**Thresholds:**

T_low = μ + 3σ 

T_med = μ + 4.5σ

T_high = μ + 6σ

### Multi-Level Warning Systemn  
A moving-average window smooths the error to avoid jittery false alarms and defines the alert levels based on a determined threshold ($\tau$).

* **None**   : error ≤ T_low
* **Low**    : T_low < error ≤ T_med
* **Medium** : T_med < error ≤ T_high
* **High**   : error > T_high

This classification system mirrors standard industrial predictive maintenance practices.

### Failure Onset Detection

FailureOnset = min { t : error(t : t+k-1) > T_high }


### RUL (Remaining Useful Life) Calculation  

**IMS sampling:**
Test1 → [first 43 files = 5 min], [others = 10 min]

Test2/3 → all 10 min

RUL = sum( Δt(j) ) for j = FailureOnset+1 ... End


### Cross-Domain Evaluation Matrix
```bash
Train → Test:
1 → 1 1 → 2 1 → 3
2 → 1 2 → 2 2 → 3
3 → 1 3 → 2 3 → 3
```
Each experiment outputs:

* MSE, MAE
* Failure index
* RUL estimate
* Warning sequence
* Forecasting plots
* Error curves
* Normal region detection
* Gaussian threshold plots
* Logs & statistics

---
## 6. Installation & Execution 

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
## Author
- Seyoon Oh
- Korea University - School of Industrial & Management Engineering
- Email : osy7336@korea.ac.kr