
from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

try:
    import torchvision.transforms as T
except Exception:
    T = None

import medmnist
from medmnist import INFO

def get_dataset_info(name: str):
    key = name.lower()
    if key not in INFO:
        raise ValueError(f"Unknown dataset: {name}. Try 'pathmnist' or 'organmnist3d'.")
    info = INFO[key]
    n_classes = int(info["n_classes"])
    task = info["task"]  # "multi-class" / "multi-label" / "binary-class"
    is_3d = key.endswith("3d")
    n_channels = int(info.get("n_channels", 1))
    return {"key": key, "n_classes": n_classes, "task": task, "is_3d": is_3d, "n_channels": n_channels, "info": info}

def _build_transform_2d(n_channels: int):
    if T is None:
        return None
    mean = [0.5] * n_channels
    std = [0.5] * n_channels
    return T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])

def _to_tensor_3d(x: np.ndarray) -> torch.Tensor:
    # x: (D,H,W); convert to (1, D, H, W)
    if x.ndim != 3:
        raise ValueError(f"Unexpected 3D input shape: {x.shape}")
    x = x.astype("float32") / 255.0
    x = np.expand_dims(x, 0)
    return torch.from_numpy(x)

def _label_to_long(y: np.ndarray) -> torch.Tensor:
    if y.ndim >= 1:
        y = y.squeeze()
    return torch.tensor(int(y), dtype=torch.long)

def _label_to_float(y: np.ndarray) -> torch.Tensor:
    y = y.astype("float32").squeeze()
    return torch.from_numpy(y)

def _get_dataset_class(key: str):
    clsname = key[0].upper() + key[1:]
    return getattr(medmnist, clsname)

def _wrap_split(ds, is_3d: bool, task: str):
    class _Wrap(torch.utils.data.Dataset):
        def __len__(self): return len(ds)
        def __getitem__(self, idx):
            img, label = ds[idx]
            if is_3d:
                if hasattr(img, "numpy"):
                    img = img.numpy()
                x = _to_tensor_3d(img)
            else:
                if hasattr(img, "size") and not isinstance(img, np.ndarray):
                    if T is None:
                        arr = np.array(img).astype("float32")/255.0
                        if arr.ndim == 2:
                            arr = np.expand_dims(arr, 0)
                        else:
                            arr = np.transpose(arr, (2,0,1))
                        x = torch.from_numpy(arr)
                    else:
                        x = img if isinstance(img, torch.Tensor) else torch.from_numpy(np.array(img)).permute(2,0,1).float()/255.0
                else:
                    arr = img.astype("float32")/255.0
                    if arr.ndim == 2:
                        arr = np.expand_dims(arr, 0)
                    else:
                        arr = np.transpose(arr, (2,0,1))
                    x = torch.from_numpy(arr)
            if task == "multi-class":
                y = _label_to_long(label)
            else:
                y = _label_to_float(label)
            return x, y
    return _Wrap()

def get_dataloaders(
    name: str,
    batch_size: int = 64,
    num_workers: int = 2,
    download: bool = True,
    limit_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    # Return train/val/test dataloaders and dataset meta.
    meta = get_dataset_info(name)
    key = meta["key"]
    DatasetClass = _get_dataset_class(key)

    transform = _build_transform_2d(meta["n_channels"]) if not meta["is_3d"] else None

    train_ds = DatasetClass(split="train", transform=transform, download=download, as_rgb=meta["n_channels"]==3)
    val_ds   = DatasetClass(split="val",   transform=transform, download=download, as_rgb=meta["n_channels"]==3)
    test_ds  = DatasetClass(split="test",  transform=transform, download=download, as_rgb=meta["n_channels"]==3)

    train_ds = _wrap_split(train_ds, is_3d=meta["is_3d"], task=meta["task"])
    val_ds   = _wrap_split(val_ds,   is_3d=meta["is_3d"], task=meta["task"])
    test_ds  = _wrap_split(test_ds,  is_3d=meta["is_3d"], task=meta["task"])

    if limit_samples is not None:
        train_ds = Subset(train_ds, list(range(min(limit_samples, len(train_ds)))))
        quarter = max(limit_samples//4, 1)
        val_ds   = Subset(val_ds,   list(range(min(quarter, len(val_ds)))))
        test_ds  = Subset(test_ds,  list(range(min(quarter, len(test_ds)))))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, meta
