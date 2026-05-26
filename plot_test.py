import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import auc

SIM_IDX = 0

configs = [
    {
        "label":        "3-node, lag=1",
        "binary_path":  "results/3node_lag1_lambda_sweep_binary_scores.npy",
        "lambda_path":  "results/3node_lag1_lambdas.npy",
        "gt_path":      "simulated/260420_data/three_nodes/result.mat",
    },
    {
        "label":        "3-node, lag=2",
        "binary_path":  "results/3node_lag2_lambda_sweep_binary_scores.npy",
        "lambda_path":  "results/3node_lag2_lambdas.npy",
        "gt_path":      "simulated/260420_data/three_nodes/result.mat",
    },
    {
        "label":        "4-node, lag=1",
        "binary_path":  "results/4node_lag1_lambda_sweep_binary_scores.npy",
        "lambda_path":  "results/4node_lag1_lambdas.npy",
        "gt_path":      "simulated/260420_data/four_nodes/result.mat",
    },
    {
        "label":        "4-node, lag=2",
        "binary_path":  "results/4node_lag2_lambda_sweep_binary_scores.npy",
        "lambda_path":  "results/4node_lag2_lambdas.npy",
        "gt_path":      "simulated/260420_data/four_nodes/result.mat",
    },
]

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
fig.suptitle(f"Neural-GC Baseline — ROC Curve (λ sweep)\nSimulation #{SIM_IDX}", fontsize=13)

for ax, cfg in zip(axes.flat, configs):
    lambdas       = np.load(cfg["lambda_path"])
    binary_scores = np.load(cfg["binary_path"])   # (n_lambdas, n_sims, p, p)

    with h5py.File(cfg["gt_path"], "r") as f:
        W_prior = np.array(f["W_prior"])           # (n_sims, p, p)

    GT = (W_prior[SIM_IDX] > 0).astype(np.float32)  # (p, p)

    n_lambdas = binary_scores.shape[0]
    p         = binary_scores.shape[2]
    off_diag  = ~np.eye(p, dtype=bool)

    tpr_list = np.zeros(n_lambdas)
    fpr_list = np.zeros(n_lambdas)

    gt_flat = GT[off_diag]

    for li in range(n_lambdas):
        pred_flat = binary_scores[li, SIM_IDX][off_diag]

        TP = np.sum((pred_flat == 1) & (gt_flat == 1))
        FP = np.sum((pred_flat == 1) & (gt_flat == 0))
        TN = np.sum((pred_flat == 0) & (gt_flat == 0))
        FN = np.sum((pred_flat == 0) & (gt_flat == 1))

        tpr_list[li] = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        fpr_list[li] = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    # 같은 FPR에서 최대 TPR만 유지 후 AUC 계산
    sort_idx   = np.argsort(fpr_list)
    fpr_sorted = fpr_list[sort_idx]
    tpr_sorted = tpr_list[sort_idx]

    unique_fpr, inv = np.unique(fpr_sorted, return_inverse=True)
    unique_tpr = np.array([tpr_sorted[inv == i].max() for i in range(len(unique_fpr))])

    roc_auc = auc(unique_fpr, unique_tpr)

    ax.plot(unique_fpr, unique_tpr,
            marker="o", markersize=6, linewidth=2, color="steelblue",
            label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")

    # 각 unique operating point에 λ 레이블 표시
    for uf, ut in zip(unique_fpr, unique_tpr):
        matches = np.where((fpr_list == uf) & (tpr_list == ut))[0]
        lam_label = "/".join([f"{lambdas[m]:.2g}" for m in matches[:2]])
        ax.annotate(f"λ={lam_label}",
                    xy=(uf, ut),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color="steelblue")

    ax.set_title(cfg["label"], fontsize=11)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = f"results/roc_test_sim{SIM_IDX}.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.show()
