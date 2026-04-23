import numpy as np
import torch

# Neural-GC repo 루트 기준
from models.cmlp import cMLP
from models.cmlp import train_model_ista

# ---------------------------
# 1. 데이터 로드
# ---------------------------
path = "/Users/ipa/workspace/Neural-GC/simulated/three_nodes_data.npy"
X = np.load(path).astype(np.float32)
X = torch.tensor(X).unsqueeze(0)   # batch 차원 추가

print("X shape:", X.shape)   # expected: (2000, 3)

# ---------------------------
# 2. 하이퍼파라미터
# ---------------------------
p = X.shape[-1]        # number of variables = 3
lag = 5               # 먼저 5로 시작
hidden = [100]        # hidden layer
lam = 0.1           # sparsity penalty
lr = 1e-2
max_iter = 2000       # 너무 오래 걸리면 1000으로 줄여도 됨
check_every = 100

# ---------------------------
# 3. 모델 생성
# ---------------------------
cmlp = cMLP(
    num_series=p,
    lag=lag,
    hidden=hidden,
    activation='relu'
)

# ---------------------------
# 4. 학습
# ---------------------------
train_model_ista(
    cmlp,
    X,
    lam=lam,
    lr=lr,
    max_iter=max_iter,
    check_every=check_every,
    verbose=1
)

GC = cmlp.GC().cpu().data.numpy()

print("Estimated Granger Causality Matrix:")
print(GC)