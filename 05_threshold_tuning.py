import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score

npz = np.load("results/test_predictions.npz")
y_test, proba_lr, proba_rf = npz["y_test"], npz["proba_lr"], npz["proba_rf"]

print(f"So mau duong trong tap Test: {y_test.sum()} / {len(y_test)}\n")

for name, proba in [("Logistic Regression", proba_lr), ("Random Forest", proba_rf)]:
    prec, rec, thr = precision_recall_curve(y_test, proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best_f1_idx = np.argmax(f1s[:-1])
    best_f1_thr = thr[best_f1_idx]

    high_recall_mask = rec[:-1] >= 0.90
    if high_recall_mask.any():
        idx = np.where(high_recall_mask)[0]
        best_recall_idx = idx[np.argmax(prec[:-1][idx])]
        recall_thr = thr[best_recall_idx]
    else:
        recall_thr = None

    print(f"=== {name} ===")
    print(f"  Nguong toi uu F1: {best_f1_thr:.3f} "
          f"-> P={prec[best_f1_idx]:.3f} R={rec[best_f1_idx]:.3f} F1={f1s[best_f1_idx]:.3f}")

    pred05 = (proba >= 0.5).astype(int)
    print(f"  So sanh voi nguong 0.5: P={precision_score(y_test, pred05):.3f} "
          f"R={recall_score(y_test, pred05):.3f} F1={f1_score(y_test, pred05):.3f}")

    if recall_thr is not None:
        predrecall = (proba >= recall_thr).astype(int)
        print(f"  Nguong uu tien Recall>=0.90: {recall_thr:.3f} "
              f"-> P={precision_score(y_test, predrecall):.3f} "
              f"R={recall_score(y_test, predrecall):.3f} "
              f"F1={f1_score(y_test, predrecall):.3f}")
    print()
