# Neural-GC Extended

An extended version of [Neural-GC](https://github.com/iancovert/Neural-GC) with support for **continuous Granger causality strength matrices** and experiments on the **SMAP dataset**.

---

## What's New

| Feature | Description |
|---|---|
| `extract_gc_strength()` | Computes a continuous GC strength matrix using L2 norms of input-layer weights |
| `train_SMAP.py` | Full training pipeline for the SMAP multivariate time series dataset |
| `train_3node.ipynb` | Training and analysis on simulated 3-node DAG data |
| `train_4node.ipynb` | Training and analysis on simulated 4-node DAG data |
| `plot_SMAP.ipynb` | Visualization of binary and continuous GC matrices for SMAP |
| `results/` | Saved models and GC matrices for reproducibility |

---

## Installation

```bash
git clone https://github.com/Choirong/Neural-GC-extended.git
cd Neural-GC-extended
```

**1. Install PyTorch** (choose based on your CUDA version)

```bash
# CUDA 11.8
pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch==2.0.1
```

**2. Install remaining dependencies**

```bash
pip install -r requirements.txt
```

> Tested with Python 3.10, PyTorch 2.0.1, CUDA 11.8

---

## Usage

### SMAP dataset

Place `SMAP_train.pkl` in the `simulated/` directory, then:

```bash
python train_SMAP.py
```

Results (binary GC, continuous GC strength, model weights) are saved to `results/SMAP(...)`.

### Simulated DAG data

Open and run the notebooks:

```
train_3node.ipynb   — 3-node simulated Granger causality experiment
train_4node.ipynb   — 4-node simulated Granger causality experiment
plot_SMAP.ipynb     — visualize GC matrices for SMAP
```

### Original demos (from Neural-GC)

```
cmlp_lagged_var_demo.ipynb
clstm_lorenz_demo.ipynb
crnn_lorenz_demo.ipynb
```

---

## 코드 구성

### `models/` — 핵심 모델

| 파일 | 설명 |
|---|---|
| `cmlp.py` | **핵심 파일.** 시계열 변수마다 독립적인 MLP를 하나씩 두는 cMLP 구조. 입력층 가중치 행렬에 Group LASSO 페널티를 적용해 변수 간 인과 관계를 학습. ISTA(근위 경사하강법) 기반 학습 함수 포함. |
| `clstm.py` | cMLP와 동일한 구조지만 LSTM 셀 사용. 장기 시간 의존성 모델링에 유리. |
| `crnn.py` | cMLP와 동일한 구조지만 기본 RNN 셀 사용. |
| `model_helper.py` | 활성화 함수 유틸리티. |

> cMLP / cLSTM / cRNN 모두 "변수 수만큼 서브 네트워크를 두고, 입력층 가중치의 L2 노름이 0이면 해당 변수는 Granger 비인과"라는 동일한 원리로 동작.

---

### 학습 스크립트

| 파일 | 설명 |
|---|---|
| `train_SMAP.py` | SMAP 실제 데이터셋에 cMLP를 학습하고, Binary GC + Continuous GC Strength 행렬을 저장. **(추가됨)** |
| `synthetic.py` | VAR(Vector Autoregression), Lorenz 등 시뮬레이션 데이터 생성기. Ground truth GC 행렬 포함. |

---

### 노트북

| 파일 | 설명 |
|---|---|
| `train_3node.ipynb` | 3-node 시뮬레이션 DAG 데이터 학습 및 GC 행렬 분석. **(추가됨)** |
| `train_4node.ipynb` | 4-node 시뮬레이션 DAG 데이터 학습 및 GC 행렬 분석. **(추가됨)** |
| `plot_SMAP.ipynb` | SMAP 결과의 Binary / Continuous GC 행렬 시각화. **(추가됨)** |
| `cmlp_lagged_var_demo.ipynb` | cMLP 기본 사용 예제 (원본) |
| `clstm_lorenz_demo.ipynb` | cLSTM Lorenz 시스템 예제 (원본) |
| `crnn_lorenz_demo.ipynb` | cRNN Lorenz 시스템 예제 (원본) |

---

### `results/` — 실험 결과

각 실험 설정별 디렉토리에 아래 파일이 저장됨:

| 파일 | 내용 |
|---|---|
| `gc_binary.npy` | Binary GC 행렬 (0/1) |
| `gc_strength.npy` | Continuous GC Strength 행렬 (실수값) |
| `cmlp_model.pt` | 학습된 모델 가중치 |
| `loss_history.npy` | 학습 loss 기록 |
| `elapsed_time.npy` | 학습 소요 시간 |

---

## Acknowledgements

This repository is based on [Neural-GC](https://github.com/iancovert/Neural-GC) by Ian Covert et al., licensed under the MIT License.

> Alex Tank, Ian Covert, Nicholas Foti, Ali Shojaie, Emily Fox.
> "Neural Granger Causality."
> *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2021.
