import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.calibration import calibration_curve

from topcer.models import GRUMultiTask, ModelConfig


def pick_logits_correct(out: object) -> torch.Tensor:
    if isinstance(out, dict):
        if "logit_correct" in out:
            return out["logit_correct"]
        for k in ["logits_correct", "correct_logits", "logits", "logit"]:
            if k in out:
                return out[k]
        raise KeyError(f"Tidak menemukan logits correctness. Keys tersedia: {list(out.keys())}")

    if isinstance(out, (tuple, list)):
        return out[0]

    if torch.is_tensor(out):
        return out

    raise TypeError(f"Output model tidak dikenali: {type(out)}")


def align_and_flatten(probs: np.ndarray, y_true: np.ndarray, mode="all"):
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)

    if mode == "last":
        # ambil last timestep
        probs_1d = probs[:, -1] if probs.ndim == 2 else probs.squeeze()
        y_1d = y_true[:, -1] if y_true.ndim == 2 else y_true.squeeze()

    else:
        # default: all timesteps
        if probs.ndim == 2 and y_true.ndim == 2:
            probs_1d = probs.reshape(-1)
            y_1d = y_true.reshape(-1)
        elif probs.ndim == 2 and y_true.ndim == 1:
            probs_1d = probs[:, -1]
            y_1d = y_true
        else:
            probs_1d = probs.squeeze()
            y_1d = y_true.squeeze()

    # mask padding label -1 (ubah jika padding Anda berbeda)
    if np.any(y_1d == -1):
        mask = (y_1d != -1)
        probs_1d = probs_1d[mask]
        y_1d = y_1d[mask]

    return probs_1d, y_1d


def safe_auc(y_true_1d: np.ndarray, probs_1d: np.ndarray):
    uniq = np.unique(y_true_1d)
    if uniq.size < 2:
        return None
    return roc_auc_score(y_true_1d, probs_1d)


def plot_roc(y_true, probs, title_suffix=""):
    auc = safe_auc(y_true, probs)
    if auc is None:
        print("ROC/AUC tidak bisa dihitung (y_true hanya 1 kelas).")
        return
    fpr, tpr, _ = roc_curve(y_true, probs)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (AUC = {auc:.4f}){title_suffix}")
    plt.grid(True)
    plt.show()


def plot_pr(y_true, probs, title_suffix=""):
    uniq = np.unique(y_true)
    if uniq.size < 2:
        print("PR curve tidak bisa dihitung (y_true hanya 1 kelas).")
        return
    prec, rec, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve (AP = {ap:.4f}){title_suffix}")
    plt.grid(True)
    plt.show()


def plot_calibration(y_true, probs, n_bins=10, title_suffix=""):
    uniq = np.unique(y_true)
    if uniq.size < 2:
        print("Calibration curve tidak bisa dihitung (y_true hanya 1 kelas).")
        return
    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy="uniform")
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1])
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Calibration Curve (bins={n_bins}){title_suffix}")
    plt.grid(True)
    plt.show()


def plot_score_hist(y_true, probs, bins=30, title_suffix=""):
    plt.figure()
    plt.hist(probs[y_true == 0], bins=bins, alpha=0.7, label="y=0")
    plt.hist(probs[y_true == 1], bins=bins, alpha=0.7, label="y=1")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title(f"Score Distribution by Class{title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_metrics_vs_threshold(y_true, probs, title_suffix=""):
    thresholds = np.linspace(0.01, 0.99, 99)
    f1s, ps, rs = [], [], []
    for t in thresholds:
        y_pred = (probs >= t).astype(int)
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
        ps.append(precision_score(y_true, y_pred, zero_division=0))
        rs.append(recall_score(y_true, y_pred, zero_division=0))

    plt.figure()
    plt.plot(thresholds, f1s, label="F1")
    plt.plot(thresholds, ps, label="Precision")
    plt.plot(thresholds, rs, label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Metrics vs Threshold{title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    test_npz = "data/sequences/seq_test.npz"
    ckpt_path = "models/state_seq_model.pt"
    threshold = 0.5

    # ====== LOAD DATA ======
    data = np.load(test_npz, allow_pickle=True)
    X = data["X"]

    # y correctness
    if "y_correct" in data.files:
        y = data["y_correct"]
    elif "y" in data.files:
        y = data["y"]
    else:
        raise RuntimeError(f"Tidak menemukan label correctness di NPZ. Keys: {data.files}")

    # kc ids
    kc_key_candidates = ["kc_id", "kc_ids", "kc", "kc_seq", "kc_index", "kc_indices"]
    kc_key = next((k for k in kc_key_candidates if k in data.files), None)
    if kc_key is None:
        raise RuntimeError(
            f"Tidak menemukan kc_id di NPZ. Keys: {data.files}\n"
            "Cek scripts/build_sequences.py untuk nama field KC yang disimpan."
        )
    kc_id = data[kc_key]

    # ====== LOAD MODEL ======
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt["config"]
    sd = ckpt["model_state"]

    cfg_dict["input_dim"] = int(X.shape[-1])
    cfg = ModelConfig(**cfg_dict)

    model = GRUMultiTask(cfg)
    model.load_state_dict(sd, strict=True)
    model.eval()

    # ====== INFERENCE (sekali saja) ======
    X_t = torch.tensor(X, dtype=torch.float32)
    kc_t = torch.tensor(kc_id, dtype=torch.long)

    with torch.no_grad():
        out = model(X_t, kc_t)

    logits_correct = pick_logits_correct(out)
    probs = torch.sigmoid(logits_correct).cpu().numpy()  # shape (B,T)

    # ====== EVALUATION for each mode ======
    for mode in ["all", "last"]:
        title_suffix = f" | mode={mode}"

        probs_1d, y_1d = align_and_flatten(probs, y, mode=mode)
        y_1d = y_1d.astype(int)
        y_pred = (probs_1d >= threshold).astype(int)

        acc = accuracy_score(y_1d, y_pred)
        auc = safe_auc(y_1d, probs_1d)
        brier = brier_score_loss(y_1d, probs_1d)

        print("\n" + "=" * 60)
        print(f"MODE: {mode}")
        print(f"Correctness Accuracy : {acc:.4f}")
        if auc is None:
            print("Correctness AUC      : (tidak bisa dihitung; hanya 1 kelas pada y_true)")
        else:
            print(f"Correctness AUC      : {auc:.4f}")
        print(f"Brier score          : {brier:.4f}")

        print(f"\n=== Classification Report (threshold = {threshold:.2f}) ===")
        print(classification_report(y_1d, y_pred, digits=4, zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(y_1d, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        plt.figure()
        disp.plot(values_format="d")
        plt.title("Confusion Matrix" + title_suffix)
        plt.grid(False)
        plt.show()

        # Plots
        plot_roc(y_1d, probs_1d, title_suffix=title_suffix)
        plot_pr(y_1d, probs_1d, title_suffix=title_suffix)
        plot_calibration(y_1d, probs_1d, n_bins=10, title_suffix=title_suffix)
        plot_score_hist(y_1d, probs_1d, bins=30, title_suffix=title_suffix)
        plot_metrics_vs_threshold(y_1d, probs_1d, title_suffix=title_suffix)


if __name__ == "__main__":
    main()
