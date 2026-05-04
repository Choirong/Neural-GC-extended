import os
import time
import pickle
import numpy as np
import torch

from models.cmlp import cMLP, train_model_ista


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

    strengths = np.stack(strengths, axis=0)
    return strengths


def main():
    # ---------------------------
    # 0. device 설정
    # ---------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------------------------
    # 1. 데이터 로드
    # ---------------------------
    data_path = "/data/mydongh/Neural-GC/simulated/SMAP_train.pkl"
    with open(data_path, "rb") as f:
        X = pickle.load(f)

    X = np.asarray(X, dtype=np.float32)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)

    print("X shape:", X.shape)

    # ---------------------------
    # 2. 하이퍼파라미터
    # ---------------------------
    p = X.shape[-1]
    lag = 5
    hidden = [100]
    lam = 0.06
    lr = 1e-2
    max_iter = 1500
    check_every = 100

    # ---------------------------
    # 3. 모델 생성
    # ---------------------------
    cmlp = cMLP(
        num_series=p,
        lag=lag,
        hidden=hidden,
        activation="relu"
    ).to(device)

    # ---------------------------
    # 4. 학습
    # ---------------------------
    start_time = time.time()

    loss_history = train_model_ista(
        cmlp,
        X,
        lam=lam,
        lr=lr,
        max_iter=max_iter,
        check_every=check_every,
        verbose=1
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nTraining time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} min)")

    GC = cmlp.GC().detach().cpu().numpy()
    GC_strength = extract_gc_strength(cmlp)

    print("Estimated Granger Causality Matrix:")
    print(GC)

    print("\nContinuous Granger Causality Strength:")
    print(np.round(GC_strength, 4))

    # ---------------------------
    # 5. 저장
    # ---------------------------
    save_dir = "/data/mydongh/Neural-GC/results/SMAP(lag5,iter1500,lam0.06)"
    os.makedirs(save_dir, exist_ok=True)

    torch.save(cmlp.state_dict(), os.path.join(save_dir, "cmlp_model.pt"))
    np.save(os.path.join(save_dir, "elapsed_time.npy"), np.array(elapsed_time))
    np.save(os.path.join(save_dir, "gc_binary.npy"), GC)
    np.save(os.path.join(save_dir, "gc_strength.npy"), GC_strength)

    if loss_history is not None:
        try:
            loss_history_np = np.array([float(x) for x in loss_history])
            np.save(os.path.join(save_dir, "loss_history.npy"), loss_history_np)
        except Exception as e:
            print("loss_history 저장 실패:", e)

    print(f"\nSaved all results to: {save_dir}")


if __name__ == "__main__":
    main()