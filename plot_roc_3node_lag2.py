import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import auc

# ── 1. 데이터 로드 ──────────────────────────────────────────────
lambdas       = np.load("results/3node_lag2_lambdas.npy")
binary_scores = np.load("results/3node_lag2_lambda_sweep_binary_scores.npy")

with h5py.File("simulated/260420_data/three_nodes/result.mat", "r") as f:
    W_prior = np.array(f["W_prior"])

GT = (W_prior > 0).astype(np.float32)

n_lambdas, n_sims, p, _ = binary_scores.shape
off_diag = ~np.eye(p, dtype=bool)

# ── 2. 각 λ에서 TPR / FPR 계산 ──────────────────────────────────
tpr_binary = np.zeros(n_lambdas)
fpr_binary = np.zeros(n_lambdas)

for li in range(n_lambdas):
    pred_all = binary_scores[li][:, off_diag].flatten()
    gt_all   = GT[:, off_diag].flatten()

    TP = np.sum((pred_all == 1) & (gt_all == 1))
    FP = np.sum((pred_all == 1) & (gt_all == 0))
    TN = np.sum((pred_all == 0) & (gt_all == 0))
    FN = np.sum((pred_all == 0) & (gt_all == 1))

    tpr_binary[li] = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    fpr_binary[li] = FP / (FP + TN) if (FP + TN) > 0 else 0.0

sort_idx = np.argsort(fpr_binary)
fpr_sorted = fpr_binary[sort_idx]
tpr_sorted = tpr_binary[sort_idx]
roc_auc = auc(fpr_sorted, tpr_sorted)

# ── 3. 시각화 ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
fig.suptitle("Neural-GC Baseline — 3-node, lag=2\nROC Curve (λ sweep)", fontsize=13)

ax.plot(fpr_sorted, tpr_sorted,
        marker="o", markersize=6, linewidth=2, color="steelblue",
        label=f"Neural-GC binary (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
ax.set_xlabel("False Positive Rate (FPR)")
ax.set_ylabel("True Positive Rate (TPR)")
ax.legend(loc="lower right")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.grid(True, alpha=0.3)

for li in sort_idx:
    ax.annotate(f"λ={lambdas[li]:.2g}",
                xy=(fpr_binary[li], tpr_binary[li]),
                xytext=(4, 4), textcoords="offset points",
                fontsize=7, color="steelblue")

plt.tight_layout()
out_path = "results/roc_3node_lag2.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
print(f"AUC: {roc_auc:.4f}")
plt.show()
