import os
import time
import numpy as np
import torch
import h5py

from models.cmlp import cMLP, train_model_ista


# ============================================================
# 0. 설정
# ============================================================
USE_DEVICE = "cuda:6"
# "cuda:6" -> GPU 서버 cuda:6
# "cpu"    -> 맥미니 CPU
# "mps"    -> Apple Silicon GPU

h5_path = "simulated/260420_data/three_nodes/signal.h5"

binary_save_path = "results/3node_lag1_lambda_sweep_binary_scores.npy"
strength_save_path = "results/3node_lag1_lambda_sweep_strength_scores.npy"
lambda_save_path = "results/3node_lag1_lambdas.npy"

test_start = 1704
test_end = 2000

lag = 1
hidden = [100]
lr = 0.01
max_iter = 1500
check_every = 100
activation = "relu"

# lambda sweep: 1e-4 ~ 1까지 로그 간격 10개
lambdas = np.logspace(np.log10(0.05), np.log10(10.0), 10)


# ============================================================
# 1. 디바이스 선택
# ============================================================
def get_device(device_name: str):
    if device_name.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_name)
        print("CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")

    if device_name == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("MPS is not available. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device("cpu")


device = get_device(USE_DEVICE)
print(f"Using device: {device}")


# ============================================================
# 2. 데이터 로드
# ============================================================
with h5py.File(h5_path, "r") as f:
    X_np = f["signal"][()].astype(np.float32)
    # 원본: (simulation, feature, timestep) = (2000, 3, 2000)

X_np = X_np.transpose(0, 2, 1)
# 변환: (simulation, timestep, feature) = (2000, 2000, 3)

X_test_np = X_np[:, test_start:test_end, :]
# test: (simulation, test_timestep, feature) = (2000, 296, 3)

print(f"Loaded X_np shape: {X_np.shape}")
print(f"Test X shape: {X_test_np.shape}")
print(f"Lambdas: {lambdas}")


# ============================================================
# 3. simulation 하나 + lambda 하나 실행
# ============================================================
def extract_gc_strength(cmlp):
    strengths = []
    for net in cmlp.networks:
        first_conv = None
        for module in net.modules():
            if isinstance(module, torch.nn.Conv1d):
                first_conv = module
                break
        if first_conv is None:
            raise RuntimeError("Conv1d layer를 찾지 못했습니다.")
        w = first_conv.weight.detach().cpu()
        s = torch.norm(w, dim=(0, 2))
        strengths.append(s.numpy())
    return np.stack(strengths, axis=0)


def run_single_simulation_with_lambda(X_single_np, lam):
    """
    X_single_np: (test_timestep, feature)
    return: (gc_binary, gc_strength), each shape (feature, feature)
    """

    X_single = torch.tensor(
        X_single_np,
        dtype=torch.float32
    ).unsqueeze(0).to(device)
    # (1, test_timestep, feature)

    p = X_single.shape[-1]

    cmlp = cMLP(
        num_series=p,
        lag=lag,
        hidden=hidden,
        activation=activation,
    ).to(device)

    train_model_ista(
        cmlp,
        X_single,
        lam=float(lam),
        lr=lr,
        max_iter=max_iter,
        check_every=check_every,
        verbose=0,
    )

    gc_binary = cmlp.GC().detach().cpu().numpy()       # (p, p)
    gc_strength = extract_gc_strength(cmlp)            # (p, p)

    return gc_binary, gc_strength


# ============================================================
# 4. lambda sweep + 전체 simulation 반복
# ============================================================
num_lambdas = len(lambdas)
num_simulations = X_test_np.shape[0]
p = X_test_np.shape[-1]

results_binary = np.zeros((num_lambdas, num_simulations, p, p), dtype=np.float32)
results_strength = np.zeros((num_lambdas, num_simulations, p, p), dtype=np.float32)
# 결과: (lambda, simulation, row, column) = (10, 2000, 3, 3)

os.makedirs("results", exist_ok=True)
np.save(lambda_save_path, lambdas)

start_time = time.time()

for lam_idx, lam in enumerate(lambdas):
    lambda_start = time.time()

    print(f"\n========== Lambda {lam_idx + 1}/{num_lambdas}: {lam:.6g} ==========")

    for sim_idx in range(num_simulations):
        sim_start = time.time()

        gc_binary, gc_strength = run_single_simulation_with_lambda(
            X_test_np[sim_idx],
            lam=lam,
        )
        results_binary[lam_idx, sim_idx] = gc_binary
        results_strength[lam_idx, sim_idx] = gc_strength

        sim_elapsed = time.time() - sim_start

        if (sim_idx + 1) % 10 == 0 or sim_idx == 0:
            total_elapsed = time.time() - start_time
            print(
                f"[lambda {lam_idx + 1}/{num_lambdas} | "
                f"sim {sim_idx + 1}/{num_simulations}] "
                f"sim_time={sim_elapsed:.2f}s, "
                f"total={total_elapsed / 60:.2f}min"
            )

            # 중간 저장
            np.save(binary_save_path, results_binary)
            np.save(strength_save_path, results_strength)

    lambda_elapsed = time.time() - lambda_start
    print(
        f"Lambda {lam:.6g} done: "
        f"{lambda_elapsed:.2f}s ({lambda_elapsed / 60:.2f}min)"
    )

    np.save(binary_save_path, results_binary)
    np.save(strength_save_path, results_strength)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"\nTotal time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} min)")
print(f"Final results shape: {results_binary.shape}")

np.save(binary_save_path, results_binary)
np.save(strength_save_path, results_strength)
np.save(lambda_save_path, lambdas)

print(f"Saved binary to:   {binary_save_path}")
print(f"Saved strength to: {strength_save_path}")
print(f"Saved lambdas to:  {lambda_save_path}")
