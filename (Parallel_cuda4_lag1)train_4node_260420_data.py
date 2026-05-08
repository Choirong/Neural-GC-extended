import os
import time
import numpy as np
import torch
import h5py
import torch.multiprocessing as mp

from models.cmlp import cMLP, train_model_ista


# ============================================================
# 0. 설정
# ============================================================
USE_DEVICE = "cuda:4"

h5_path = "simulated/260420_data/four_nodes/signal.h5"

binary_save_path   = "results/4node_lag1_lambda_sweep_binary_scores.npy"
strength_save_path = "results/4node_lag1_lambda_sweep_strength_scores.npy"
lambda_save_path   = "results/4node_lag1_lambdas.npy"

test_start  = 1704
test_end    = 2000

lag         = 1
hidden      = [100]
lr          = 0.01
max_iter    = 1500
check_every = 100
activation  = "relu"

lambdas = np.logspace(np.log10(0.05), np.log10(10.0), 10)

N_WORKERS = 4  # 병렬 프로세스 수


# ============================================================
# 1. 함수 정의
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


def run_single_simulation_with_lambda(X_single_np, lam, device):
    X_single = torch.tensor(
        X_single_np,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

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

    gc_binary   = cmlp.GC().detach().cpu().numpy()
    gc_strength = extract_gc_strength(cmlp)

    return gc_binary, gc_strength


def worker_fn(worker_id, sim_start_idx, sim_end_idx, X_test_np, lambdas, device_str):
    """각 워커: 담당 sim 구간 × 전체 lambda 처리 후 파일로 저장"""
    device = torch.device(device_str)

    num_lambdas    = len(lambdas)
    num_sims_local = sim_end_idx - sim_start_idx
    p              = X_test_np.shape[-1]

    results_binary   = np.zeros((num_lambdas, num_sims_local, p, p), dtype=np.float32)
    results_strength = np.zeros((num_lambdas, num_sims_local, p, p), dtype=np.float32)

    start_time = time.time()

    for lam_idx, lam in enumerate(lambdas):
        for local_idx, sim_idx in enumerate(range(sim_start_idx, sim_end_idx)):
            gc_binary, gc_strength = run_single_simulation_with_lambda(
                X_test_np[sim_idx], lam, device
            )
            results_binary[lam_idx, local_idx]   = gc_binary
            results_strength[lam_idx, local_idx] = gc_strength

            if (local_idx + 1) % 10 == 0 or local_idx == 0:
                elapsed = time.time() - start_time
                print(
                    f"[Worker {worker_id} | lambda {lam_idx+1}/{num_lambdas} | "
                    f"sim {sim_idx+1} ({local_idx+1}/{num_sims_local})] "
                    f"total={elapsed/60:.2f}min",
                    flush=True
                )

    np.save(f"results/worker/4node_lag1_worker{worker_id}_binary.npy",   results_binary)
    np.save(f"results/worker/4node_lag1_worker{worker_id}_strength.npy", results_strength)
    print(f"[Worker {worker_id}] Done. {(time.time()-start_time)/60:.2f}min", flush=True)


# ============================================================
# 2. 메인
# ============================================================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    with h5py.File(h5_path, "r") as f:
        X_np = f["signal"][()].astype(np.float32)

    X_np   = X_np.transpose(0, 2, 1)
    X_test = X_np[:, test_start:test_end, :]

    print(f"Device: {USE_DEVICE}")
    print(f"Test X shape: {X_test.shape}")
    print(f"Lambdas: {np.round(lambdas, 4)}")
    print(f"N_WORKERS: {N_WORKERS}")

    os.makedirs("results", exist_ok=True)
    os.makedirs("results/worker", exist_ok=True)
    np.save(lambda_save_path, lambdas)

    num_sims = X_test.shape[0]
    chunk    = num_sims // N_WORKERS
    ranges   = [
        (i * chunk, (i + 1) * chunk if i < N_WORKERS - 1 else num_sims)
        for i in range(N_WORKERS)
    ]
    print(f"Sim ranges: {ranges}")

    processes  = []
    total_start = time.time()

    for wid, (s, e) in enumerate(ranges):
        proc = mp.Process(
            target=worker_fn,
            args=(wid, s, e, X_test, lambdas, USE_DEVICE)
        )
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()

    total_elapsed = time.time() - total_start
    print(f"\nAll workers done. Total: {total_elapsed/60:.2f}min")

    # ============================================================
    # 3. 결과 합치기
    # ============================================================
    num_lambdas = len(lambdas)
    p_dim       = X_test.shape[-1]

    all_binary   = np.zeros((num_lambdas, num_sims, p_dim, p_dim), dtype=np.float32)
    all_strength = np.zeros((num_lambdas, num_sims, p_dim, p_dim), dtype=np.float32)

    for wid, (s, e) in enumerate(ranges):
        all_binary[:, s:e]   = np.load(f"results/worker/4node_lag1_worker{wid}_binary.npy")
        all_strength[:, s:e] = np.load(f"results/worker/4node_lag1_worker{wid}_strength.npy")

    np.save(binary_save_path,   all_binary)
    np.save(strength_save_path, all_strength)
    np.save(lambda_save_path,   lambdas)

    print(f"Merged binary shape:   {all_binary.shape}")
    print(f"Saved binary to:       {binary_save_path}")
    print(f"Saved strength to:     {strength_save_path}")
    print(f"Saved lambdas to:      {lambda_save_path}")
