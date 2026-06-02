"""
116-Node DAG Causal Simulation
================================
4노드 MATLAB 코드 설계 원칙 확장.

4노드 기존 방식:
  source → target : tanh 또는 sin
  target → target : X² (VAR이 못 잡는 nonlinear)

116노드 확장:
  target → target : tanh(x)² + running mean centering
  이유: 순수 X²는 체인이 길어지면 발산.
        tanh(x)²는 출력이 [0,1]로 bounded.
        단, tanh(x)²는 항상 양수라 mean 편향 발생.
        → running mean centering으로 mean을 0에 가깝게 유지.
        → U자형 양쪽 모두 활용 → VAR이 못 잡는 비선형 보장.

조건:
  (1) Variance bounded     : AR 계수 0.4, tanh(x)²로 bounded 보장
  (2) Linearly independent : Source 노드는 서로 독립적 AR 프로세스
  (3) X² nonlinear term    : target→target에 tanh(x)² + running mean centering 적용
"""

import numpy as np
import os
from collections import Counter

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
N_NODES   = 116
T         = 2000
N_SIM     = 10
SPARSITY  = 0.05
AR_COEF   = 0.4
STD_SRC   = 1.0
STD_TGT   = 0.1
COUPLING  = 0.8
BASE_SEED = 42

OUTPUT_DIR = "./simulation_116nodes"
os.makedirs(f"{OUTPUT_DIR}/signal", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/gt",     exist_ok=True)

# ─────────────────────────────────────────────
# Step 1: DAG Adjacency Matrix 생성
# ─────────────────────────────────────────────
def generate_dag_adjacency(n_nodes, sparsity, seed=0):
    rng = np.random.default_rng(seed)
    total_possible = n_nodes * (n_nodes - 1) // 2
    n_edges = int(total_possible * sparsity)
    adj = np.zeros((n_nodes, n_nodes), dtype=int)
    np.fill_diagonal(adj, 1)
    rows, cols = np.tril_indices(n_nodes, k=-1)
    chosen = rng.choice(len(rows), size=n_edges, replace=False)
    for idx in chosen:
        adj[rows[idx], cols[idx]] = 1
    return adj

# ─────────────────────────────────────────────
# Step 2: Source / Target 노드 구분
# ─────────────────────────────────────────────
def get_node_types(adj):
    n = adj.shape[0]
    source_nodes, target_nodes = [], []
    for i in range(n):
        if adj[i, :i].sum() == 0:
            source_nodes.append(i)
        else:
            target_nodes.append(i)
    return source_nodes, target_nodes

# ─────────────────────────────────────────────
# Step 3: Nonlinear 함수 타입 할당
#   source → target : 0=tanh, 1=sin 랜덤
#   target → target : 2=tanh(x)²  고정
# ─────────────────────────────────────────────
def assign_nonlinear_types(adj, source_set, seed):
    rng = np.random.default_rng(seed)
    n = adj.shape[0]
    nl = {}
    for i in range(n):
        for j in range(i):
            if adj[i, j] == 1:
                if j in source_set:
                    nl[(i, j)] = rng.integers(0, 2)  # tanh or sin
                else:
                    nl[(i, j)] = 2                    # tanh(x)²
    return nl

def apply_nonlinear(fn_type, x):
    if fn_type == 0:
        return np.tanh(x)       # [-1, 1]
    elif fn_type == 1:
        return np.sin(x)        # [-1, 1]
    else:
        return np.tanh(x) ** 2  # [0, 1], VAR이 못 잡는 nonlinear

# ─────────────────────────────────────────────
# Step 4: 시계열 데이터 생성
#   running mean centering:
# ─────────────────────────────────────────────
def generate_time_series(adj, nl_types, source_nodes, T,
                          ar_coef, std_src, std_tgt, coupling, sim_seed):
    rng = np.random.default_rng(sim_seed)
    n = adj.shape[0]
    source_set = set(source_nodes)
    std = np.array([std_src if i in source_set else std_tgt for i in range(n)])
    parents_list = [np.where(adj[i, :i] == 1)[0] for i in range(n)]

    X = np.zeros((n, T))
    X[:, 0] = std * rng.standard_normal(n)

    # running mean state: S_j(0)=0, N_j(0)=0
    tanh2_sum   = np.zeros(n)
    tanh2_count = np.zeros(n)

    for t in range(1, T):
        eps = std * rng.standard_normal(n)
        for i in range(n):
            val = ar_coef * X[i, t-1]
            parents = parents_list[i]
            n_parents = len(parents)
            if n_parents > 0:
                scale = coupling / np.sqrt(n_parents)
                for j in parents:
                    fn = nl_types[(i, j)]
                    out = apply_nonlinear(fn, X[j, t-1])
                    if fn == 2:
                        # 충분한 sample이 모인 후에만 centering 적용
                        if tanh2_count[j] > 10:
                            out = out - tanh2_sum[j] / tanh2_count[j]
                        tanh2_sum[j]   += np.tanh(X[j, t-1]) ** 2
                        tanh2_count[j] += 1
                    val += scale * out
            X[i, t] = val + eps[i]

    return X  # (N, T)

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=== 116-Node DAG Simulation ===")
    print(f"Nodes: {N_NODES}, T: {T}, N_sim: {N_SIM}, Sparsity: {SPARSITY*100:.1f}%\n")

    adj = generate_dag_adjacency(N_NODES, SPARSITY, seed=BASE_SEED)
    source_nodes, target_nodes = get_node_types(adj)
    source_set = set(source_nodes)
    nl_types = assign_nonlinear_types(adj, source_set, seed=BASE_SEED)

    total_possible = N_NODES * (N_NODES - 1) // 2
    n_edges = int(adj.sum()) - N_NODES
    actual_sparsity = n_edges / total_possible * 100
    parents_counts = [len(np.where(adj[i,:i]==1)[0]) for i in range(N_NODES)]

    print(f"Source 노드  : {len(source_nodes)}개")
    print(f"Target 노드  : {len(target_nodes)}개")
    print(f"실제 연결 수  : {n_edges}개 ({actual_sparsity:.2f}%)")
    print(f"부모 수 분포  : mean={np.mean(parents_counts):.1f}, max={max(parents_counts)}")

    type_names = {0: "tanh", 1: "sin", 2: "tanh(x)²"}
    counts = Counter(nl_types.values())
    print(f"Nonlinear    : " + ", ".join([f"{type_names[k]}={v}" for k, v in sorted(counts.items())]))
    print(f"  → source→target: tanh/sin | target→target: tanh(x)² + running mean centering\n")

    np.save(f"{OUTPUT_DIR}/gt/gt_adjacency.npy", adj)
    print("GT adjacency matrix 저장 완료\n")

    for k in range(1, N_SIM + 1):
        sim_seed = BASE_SEED + k
        X = generate_time_series(
            adj, nl_types, source_nodes, T,
            AR_COEF, STD_SRC, STD_TGT, COUPLING, sim_seed
        )
        np.save(f"{OUTPUT_DIR}/signal/signal_{k:03d}.npy", X.T)  # shape: (T, N)
        per_node_std = X.std(axis=1)
        print(f"  Sim {k:02d}/{N_SIM} | "
              f"mean={X.mean():.4f}, std={X.std():.4f}, "
              f"max={X.max():.4f}, min={X.min():.4f} | "
              f"node_std: [{per_node_std.min():.3f}, {per_node_std.max():.3f}]")

    print(f"\n=== 완료 ===")
    print(f"저장 위치    : {OUTPUT_DIR}/")
    print(f"  signal/  : signal_001.npy ~ signal_{N_SIM:03d}.npy  (shape: T={T} x N={N_NODES})")
    print(f"  gt/      : gt_adjacency.npy                          (shape: {N_NODES} x {N_NODES})")
