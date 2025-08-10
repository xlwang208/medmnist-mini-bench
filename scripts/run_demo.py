
import os, argparse, json, datetime as dt, yaml
import torch

from bench import get_dataloaders, get_dataset_info, build_model, train_and_eval, set_seed

def parse_args():
    p = argparse.ArgumentParser(description="Minimal MedMNIST demo")
    p.add_argument("--dataset", type=str, default="pathmnist", help="pathmnist | organmnist3d")
    p.add_argument("--model", type=str, default=None, help="cnn | cnn3d (default auto by dataset)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--limit-samples", type=int, default=None, help="use a small subset for speed")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", type=str, default=None, help="YAML config file")
    return p.parse_args()

def main():
    args = parse_args()
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            setattr(args, k, v)

    set_seed(args.seed)

    meta = get_dataset_info(args.dataset)
    in_channels = meta["n_channels"] if not meta["is_3d"] else 1
    num_classes = meta["n_classes"]
    task = meta["task"]

    model_kind = args.model or ("cnn3d" if meta["is_3d"] else "cnn")
    model = build_model(model_kind, in_channels=in_channels, num_classes=num_classes)

    train_loader, val_loader, test_loader, _ = get_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=True,
        limit_samples=args.limit_samples,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join("outputs", args.dataset, model_kind, ts)

    metrics = train_and_eval(
        model=model,
        loaders=(train_loader, val_loader, test_loader),
        device=device,
        task=task,
        num_classes=num_classes,
        epochs=args.epochs,
        lr=args.lr,
        out_dir=out_dir,
    )

    print(json.dumps(metrics, indent=2))
    print(f"Saved to: {out_dir}")

if __name__ == "__main__":
    main()
