
from typing import Dict, Tuple
import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _criterion(task: str, num_classes: int):
    if task == "multi-label" or num_classes == 1:
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss()

def _softmax_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)

def _sigmoid_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _save_confusion_matrix(y_true, y_pred, classes, out_png: str):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(5,5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def evaluate(model, loader, device, task: str, num_classes: int):
    model.eval()
    ys, ps = [], []
    y_preds = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            if task == "multi-label" or num_classes == 1:
                prob = _sigmoid_logits(logits).cpu().numpy()
                ys.append(y.cpu().numpy())
                y_pred = (prob > 0.5).astype(int)
            else:
                prob = _softmax_logits(logits).cpu().numpy()
                ys.append(y.cpu().numpy())
                y_pred = prob.argmax(1)
            ps.append(prob)
            y_preds.append(y_pred)
    import numpy as _np
    y_true = _np.concatenate(ys, axis=0)
    prob = _np.concatenate(ps, axis=0)
    y_pred = _np.concatenate(y_preds, axis=0)
    if task == "multi-class":
        acc = float(accuracy_score(y_true, y_pred))
        try:
            y_true_ovr = _np.eye(num_classes)[y_true.astype(int)]
            auc = float(roc_auc_score(y_true_ovr, prob, multi_class="ovr"))
        except Exception:
            auc = float("nan")
    else:
        acc = float((y_true == y_pred).mean())
        try:
            auc = float(roc_auc_score(y_true, prob))
        except Exception:
            auc = float("nan")
    return {"acc": acc, "auc": auc, "y_true": y_true.tolist(), "y_pred": y_pred.tolist()}

def train_and_eval(
    model: torch.nn.Module,
    loaders: Tuple,
    device: torch.device,
    task: str,
    num_classes: int,
    epochs: int = 1,
    lr: float = 1e-3,
    out_dir: str = "outputs",
):
    train_loader, val_loader, test_loader = loaders
    model.to(device)
    criterion = _criterion(task, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val = -1.0
    best_state = None

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y if task != "multi-class" else y.long())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate(model, val_loader, device, task, num_classes)
        if val_metrics["acc"] > best_val:
            best_val = val_metrics["acc"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    _ensure_dir(out_dir)
    if best_state is not None:
        torch.save(best_state, os.path.join(out_dir, "best.pt"))
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    test_metrics = evaluate(model, test_loader, device, task, num_classes)

    classes = [str(i) for i in range(num_classes)]
    _save_confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"], classes, os.path.join(out_dir, "confusion_matrix.png"))

    metrics = {
        "val_acc_best": float(best_val),
        "test_acc": float(test_metrics["acc"]),
        "test_auc": float(test_metrics["auc"]),
        "epochs": int(epochs),
        "lr": float(lr),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
