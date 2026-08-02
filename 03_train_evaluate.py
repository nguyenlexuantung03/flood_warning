import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score, precision_score, recall_score,
    confusion_matrix
)

FEATURES = ["precipitation_sum", "rain_cum_3d", "rain_cum_5d", "rain_cum_7d"]
RANDOM_STATE = 42

train = pd.read_csv("train.csv", parse_dates=["time"]).sort_values("time")
test = pd.read_csv("test.csv", parse_dates=["time"]).sort_values("time")

X_train, y_train = train[FEATURES].values, train["label"].values
X_test, y_test = test[FEATURES].values, test["label"].values

# walk-forward CV theo moc thoi gian, k=3
# chia deu theo so dong bi loi (fold 2018-2019 khong co dot lu nao),
# nen chon ranh gioi thu cong theo ngay de fold nao cung co mau duong
FOLD_CUTS = [
    ("2017-06-30", "2019-06-30"),
    ("2019-06-30", "2021-12-31"),
    ("2021-12-31", "2023-12-31"),
]


def walk_forward_folds_by_date(time_series, cuts):
    folds = []
    for train_end, val_end in cuts:
        tr_idx = np.where(time_series <= train_end)[0]
        va_idx = np.where((time_series > train_end) & (time_series <= val_end))[0]
        folds.append((tr_idx, va_idx))
    return folds


folds = walk_forward_folds_by_date(train["time"], FOLD_CUTS)
for i, (tr_idx, va_idx) in enumerate(folds, 1):
    n_pos_val = y_train[va_idx].sum()
    print(f"Fold {i}: train={len(tr_idx)} | val={len(va_idx)} | mau duong val = {n_pos_val}")

LR_GRID = [0.01, 0.1, 1, 10]
RF_GRID = [
    {"n_estimators": n, "max_depth": d, "min_samples_leaf": m}
    for n in [100, 200, 300] for d in [5, 10, None] for m in [1, 5, 10]
]


def cv_score(model_fn, param):
    f1s = []
    for tr_idx, va_idx in folds:
        model = model_fn(param)
        model.fit(X_train[tr_idx], y_train[tr_idx])
        pred = model.predict(X_train[va_idx])
        f1s.append(f1_score(y_train[va_idx], pred, zero_division=0))
    return np.mean(f1s)


lr_results = {}
for C in LR_GRID:
    fn = lambda c=C: LogisticRegression(
        class_weight="balanced", penalty="l2", C=c, max_iter=1000, solver="lbfgs"
    )
    lr_results[C] = cv_score(fn, C)
best_C = max(lr_results, key=lr_results.get)

rf_results = {}
for p in RF_GRID:
    fn = lambda pp=p: RandomForestClassifier(
        n_estimators=pp["n_estimators"], max_depth=pp["max_depth"],
        min_samples_leaf=pp["min_samples_leaf"], class_weight="balanced",
        max_features="sqrt", random_state=RANDOM_STATE
    )
    rf_results[json.dumps(p)] = cv_score(fn, p)
best_rf_params = json.loads(max(rf_results, key=rf_results.get))

print("\nSieu tham so tot nhat (walk-forward CV, F1 trung binh):")
print(f"Logistic Regression: C = {best_C}  (F1_cv = {lr_results[best_C]:.4f})")
print(f"Random Forest: {best_rf_params}  (F1_cv = {rf_results[json.dumps(best_rf_params)]:.4f})")

final_lr = LogisticRegression(
    class_weight="balanced", penalty="l2", C=best_C, max_iter=1000, solver="lbfgs"
).fit(X_train, y_train)

final_rf = RandomForestClassifier(
    n_estimators=best_rf_params["n_estimators"], max_depth=best_rf_params["max_depth"],
    min_samples_leaf=best_rf_params["min_samples_leaf"], class_weight="balanced",
    max_features="sqrt", random_state=RANDOM_STATE
).fit(X_train, y_train)

results = {}
for name, model in [("Logistic Regression", final_lr), ("Random Forest", final_rf)]:
    proba = model.predict_proba(X_test)[:, 1]
    pred_05 = (proba >= 0.5).astype(int)
    results[name] = {
        "accuracy": accuracy_score(y_test, pred_05),
        "precision": precision_score(y_test, pred_05, zero_division=0),
        "recall": recall_score(y_test, pred_05, zero_division=0),
        "f1": f1_score(y_test, pred_05, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "confusion_matrix": confusion_matrix(y_test, pred_05).tolist(),
    }

print("\nKet qua tren tap Test (nguong 0.5):")
for name, r in results.items():
    print(f"{name}: Acc={r['accuracy']:.3f} P={r['precision']:.3f} R={r['recall']:.3f} "
          f"F1={r['f1']:.3f} AUC={r['roc_auc']:.3f}")
    print(f"  Confusion matrix (true x pred): {r['confusion_matrix']}")

with open("results/metrics_test_results.json", "w") as f:
    json.dump({
        "best_C": best_C, "best_rf_params": best_rf_params,
        "lr_cv_scores": lr_results, "results": results
    }, f, indent=2, default=str)

np.savez("results/test_predictions.npz",
         y_test=y_test,
         proba_lr=final_lr.predict_proba(X_test)[:, 1],
         proba_rf=final_rf.predict_proba(X_test)[:, 1])

print("\nDa luu ket qua vao results/metrics_test_results.json")
