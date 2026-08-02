import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, f1_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix

FEATURES = ["precipitation_sum", "rain_cum_3d", "rain_cum_5d", "rain_cum_7d"]
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})

train = pd.read_csv("train.csv", parse_dates=["time"]).sort_values("time")
test = pd.read_csv("test.csv", parse_dates=["time"]).sort_values("time")
X_train, y_train = train[FEATURES].values, train["label"].values
X_test, y_test = test[FEATURES].values, test["label"].values

with open("results/metrics_test_results.json") as f:
    meta = json.load(f)
best_C = meta["best_C"]
best_rf = meta["best_rf_params"]

# --- LR: log-loss theo so vong lap (warm_start) ---
iters = list(range(10, 310, 10))
train_losses, val_losses = [], []
split = int(len(X_train) * 0.85)
Xtr, Xva = X_train[:split], X_train[split:]
ytr, yva = y_train[:split], y_train[split:]

model = LogisticRegression(C=best_C, class_weight="balanced",
                            solver="lbfgs", warm_start=True, max_iter=10)
for it in iters:
    model.max_iter = 10
    model.fit(Xtr, ytr)
    train_losses.append(log_loss(ytr, model.predict_proba(Xtr)[:, 1], labels=[0, 1]))
    val_losses.append(log_loss(yva, model.predict_proba(Xva)[:, 1], labels=[0, 1]))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(iters, train_losses, label="Train log-loss", marker="o", ms=3)
ax.plot(iters, val_losses, label="Validation log-loss", marker="s", ms=3)
ax.set_xlabel("So vong lap toi uu hoa (lbfgs)")
ax.set_ylabel("Log-loss")
ax.set_title("Duong cong hoi tu - Logistic Regression")
ax.legend()
fig.tight_layout()
fig.savefig("figures/lr_convergence.png")
plt.close(fig)

# --- RF: F1 / ROC-AUC theo so luong cay ---
n_trees_grid = [10, 25, 50, 75, 100, 150, 200, 300]
rf_f1s, rf_aucs = [], []
for n in n_trees_grid:
    rf = RandomForestClassifier(n_estimators=n, max_depth=best_rf["max_depth"],
                                 min_samples_leaf=best_rf["min_samples_leaf"],
                                 class_weight="balanced", max_features="sqrt",
                                 random_state=42).fit(X_train, y_train)
    proba = rf.predict_proba(X_test)[:, 1]
    rf_f1s.append(f1_score(y_test, (proba >= 0.5).astype(int)))
    rf_aucs.append(roc_auc_score(y_test, proba))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(n_trees_grid, rf_f1s, label="F1-score (Test)", marker="o")
ax.plot(n_trees_grid, rf_aucs, label="ROC-AUC (Test)", marker="s")
ax.set_xlabel("So luong cay (n_estimators)")
ax.set_ylabel("Diem so")
ax.set_title("Duong cong hoi tu - Random Forest")
ax.legend()
fig.tight_layout()
fig.savefig("figures/rf_convergence.png")
plt.close(fig)

# --- confusion matrix + ROC + PR (2 mo hinh) ---
npz = np.load("results/test_predictions.npz")
proba_lr, proba_rf = npz["proba_lr"], npz["proba_rf"]

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, proba, name in zip(axes, [proba_lr, proba_rf], ["Logistic Regression", "Random Forest"]):
    cm = confusion_matrix(y_test, (proba >= 0.5).astype(int))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Khong ngap", "Co nguy co"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Khong ngap", "Co nguy co"])
    ax.set_xlabel("Du doan"); ax.set_ylabel("Thuc te")
    ax.set_title(name)
fig.suptitle("Ma tran nham lan tren tap Test (nguong = 0.5)")
fig.tight_layout()
fig.savefig("figures/confusion_matrices.png")
plt.close(fig)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
for proba, name in zip([proba_lr, proba_rf], ["Logistic Regression", "Random Forest"]):
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax1.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, proba):.3f})")
    prec, rec, _ = precision_recall_curve(y_test, proba)
    ax2.plot(rec, prec, label=name)
ax1.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curve"); ax1.legend()
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall Curve"); ax2.legend()
fig.tight_layout()
fig.savefig("figures/roc_pr_curves.png")
plt.close(fig)

print("Da luu 4 bieu do vao figures/")
