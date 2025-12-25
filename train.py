import argparse
import pathlib
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_registry() -> Dict[str, Tuple]:
    """
    Returns mapping of model names to (builder_fn, weights_enum_or_None).
    """
    return {
        "resnet18": (models.resnet18, getattr(models, "ResNet18_Weights", None)),
        "resnet50": (models.resnet50, getattr(models, "ResNet50_Weights", None)),
        "vit_b_16": (models.vit_b_16, getattr(models, "ViT_B_16_Weights", None)),
        "vit_b_32": (models.vit_b_32, getattr(models, "ViT_B_32_Weights", None)),
    }


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    reg = model_registry()
    if name not in reg:
        raise ValueError(f"Unsupported model '{name}'. Choices: {', '.join(reg)}")

    build, weights_enum = reg[name]
    weights = weights_enum.DEFAULT if pretrained and weights_enum else None
    model = build(weights=weights)

    if name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    return model


def build_dataloaders(
    data_root: str,
    dataset: str,
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_tfms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    if dataset == "cifar10":
        train_ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tfms)
        test_ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tfms)
        num_classes = 10
    elif dataset == "cifar100":
        train_ds = datasets.CIFAR100(root=data_root, train=True, download=True, transform=train_tfms)
        test_ds = datasets.CIFAR100(root=data_root, train=False, download=True, transform=test_tfms)
        num_classes = 100
    else:
        raise ValueError("dataset must be cifar10 or cifar100")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    return train_loader, test_loader, num_classes


def build_optimizer_scheduler(model: nn.Module, name: str, epochs: int, batch_size: int, base_lr: float = None):
    if name.startswith("resnet"):
        lr = base_lr or 0.1 * batch_size / 128
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    else:
        lr = base_lr or 5e-4 * batch_size / 128
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return optimizer, scheduler


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, correct = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    model.train()
    return correct / total if total else 0.0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: amp.GradScaler,
    device: torch.device,
    label_smoothing: float = 0.1,
):
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast():
            logits = model(images)
            loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)


def save_checkpoint(model: nn.Module, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(to_save, path)


def parse_args():
    ap = argparse.ArgumentParser(description="Train ResNet or ViT on CIFAR-10/100 and save checkpoint.")
    ap.add_argument("--model", type=str, default="resnet18", choices=list(model_registry().keys()))
    ap.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--data-root", type=str, default="data")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None, help="Override base learning rate.")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-pretrained", action="store_true", help="Do not start from torchvision pretrained weights.")
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    ap.add_argument("--output", type=str, default=None, help="Path to save checkpoint. Defaults to checkpoints/{model}_{dataset}.pth")
    return ap.parse_args()


def main():
    args = parse_args()
    device = DEFAULT_DEVICE
    print(f"Using device: {device}")

    train_loader, test_loader, num_classes = build_dataloaders(
        data_root=args.data_root,
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = build_model(args.model, num_classes=num_classes, pretrained=not args.no_pretrained).to(device)

    if torch.cuda.device_count() > 1 and device.type == "cuda":
        # Simple data parallel for faster training on multiple GPUs.
        model = nn.DataParallel(model)

    optimizer, scheduler = build_optimizer_scheduler(
        model, name=args.model, epochs=args.epochs, batch_size=args.batch_size, base_lr=args.lr
    )
    scaler = amp.GradScaler()

    best_acc = 0.0
    output_path = pathlib.Path(args.output or f"checkpoints/{args.model}_{args.dataset}.pth")

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        target = model.module if isinstance(model, nn.DataParallel) else model
        target.load_state_dict(state)
        best_acc = evaluate(model, test_loader, device)
        print(f"Resumed from {args.resume} with eval acc {best_acc*100:.2f}%")

    output_path = pathlib.Path(args.output or f"checkpoints/{args.model}_{args.dataset}.pth")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        acc = evaluate(model, test_loader, device)
        scheduler.step()
        if acc > best_acc:
            best_acc = acc
            save_checkpoint(model, output_path)
        print(
            f"epoch {epoch:03d}/{args.epochs} | loss={train_loss:.4f} | acc={acc*100:.2f}% "
            f"| best={best_acc*100:.2f}% | lr={scheduler.get_last_lr()[0]:.5f}"
        )

    print(f"Best checkpoint saved to {output_path} with accuracy {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
