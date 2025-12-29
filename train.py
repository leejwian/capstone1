import argparse
import pathlib
from typing import Dict, Tuple

import torch
import torch.nn.functional as functional
from torch import nn, optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_registry() -> Dict[str, Tuple]:
    """
    Return a mapping from model name to a tuple of
    (model_builder_function, pretrained_weights_enum).
    """
    return {
        "resnet18": (models.resnet18, getattr(models, "ResNet18_Weights", None)),
        "resnet50": (models.resnet50, getattr(models, "ResNet50_Weights", None)),
        "vit_b_16": (models.vit_b_16, getattr(models, "ViT_B_16_Weights", None)),
        "vit_b_32": (models.vit_b_32, getattr(models, "ViT_B_32_Weights", None)),
    }


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Build a classification model with a modified classification head.

    Args:
        model_name: Name of the model architecture.
        num_classes: Number of output classes.
        pretrained: Whether to load torchvision pretrained weights.

    Returns:
        Instantiated PyTorch model.
    """
    registry = get_model_registry()
    if model_name not in registry:
        raise ValueError(f"Unsupported model '{model_name}'. Choices: {', '.join(registry)}")

    builder, weights_enum = registry[model_name]
    weights = weights_enum.DEFAULT if pretrained and weights_enum else None
    model = builder(weights=weights)

    if model_name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    return model


def build_dataloaders(
    data_root: str,
    dataset_name: str,
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Build training and test dataloaders for CIFAR datasets.

    Args:
        data_root: Root directory for datasets.
        dataset_name: Either 'cifar10' or 'cifar100'.
        batch_size: Batch size.
        num_workers: Number of DataLoader workers.

    Returns:
        train_loader, test_loader, num_classes
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    if dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=train_transforms,
        )
        test_dataset = datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=test_transforms,
        )
        num_classes = 10
    elif dataset_name == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=data_root,
            train=True,
            download=True,
            transform=train_transforms,
        )
        test_dataset = datasets.CIFAR100(
            root=data_root,
            train=False,
            download=True,
            transform=test_transforms,
        )
        num_classes = 100
    else:
        raise ValueError("dataset_name must be either 'cifar10' or 'cifar100'")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader, num_classes


def build_optimizer_and_scheduler(
    model: nn.Module,
    model_name: str,
    epochs: int,
    batch_size: int,
    base_lr: float | None = None,
):
    """
    Build optimizer and learning-rate scheduler.

    Args:
        model: Model to optimize.
        model_name: Model architecture name.
        epochs: Number of training epochs.
        batch_size: Batch size.
        base_lr: Optional base learning rate override.

    Returns:
        optimizer, scheduler
    """
    if model_name.startswith("resnet"):
        learning_rate = base_lr or 0.1 * batch_size / 128
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
        )
    else:
        learning_rate = base_lr or 5e-4 * batch_size / 128
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.05,
        )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )
    return optimizer, scheduler


@torch.no_grad()
def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    """
    Evaluate classification accuracy.

    Args:
        model: Trained model.
        data_loader: Evaluation DataLoader.
        device: Target device.

    Returns:
        Top-1 accuracy.
    """
    model.eval()
    total_samples = 0
    correct_predictions = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        predictions = logits.argmax(dim=1)

        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)

    model.train()
    return correct_predictions / total_samples if total_samples > 0 else 0.0


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: amp.GradScaler,
    device: torch.device,
    label_smoothing: float = 0.1,
) -> float:
    """
    Train the model for one epoch.

    Returns:
        Average training loss.
    """
    total_loss = 0.0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with amp.autocast():
            logits = model(images)
            loss = functional.cross_entropy(
                logits,
                labels,
                label_smoothing=label_smoothing,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)

    return total_loss / len(data_loader.dataset)


def save_checkpoint(model: nn.Module, checkpoint_path: pathlib.Path) -> None:
    """
    Save model state dictionary to disk.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = (
        model.module.state_dict()
        if isinstance(model, nn.DataParallel)
        else model.state_dict()
    )
    torch.save(state_dict, checkpoint_path)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train ResNet or ViT on CIFAR-10/100 and save checkpoints."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=list(get_model_registry().keys()),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100"],
    )
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable torchvision pretrained weights.",
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Checkpoint path (default: checkpoints/{model}_{dataset}.pth)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = DEFAULT_DEVICE
    print(f"Using device: {device}")

    train_loader, test_loader, num_classes = build_dataloaders(
        data_root=args.data_root,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
    ).to(device)

    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)

    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        base_lr=args.lr,
    )
    scaler = amp.GradScaler()

    best_accuracy = 0.0
    output_path = pathlib.Path(
        args.output or f"checkpoints/{args.model}_{args.dataset}.pth"
    )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        target_model = model.module if isinstance(model, nn.DataParallel) else model
        target_model.load_state_dict(checkpoint)
        best_accuracy = evaluate(model, test_loader, device)
        print(f"Resumed from {args.resume} | acc={best_accuracy * 100:.2f}%")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )
        accuracy = evaluate(model, test_loader, device)
        scheduler.step()

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_checkpoint(model, output_path)

        print(
            f"epoch {epoch:03d}/{args.epochs} | "
            f"loss={train_loss:.4f} | "
            f"acc={accuracy * 100:.2f}% | "
            f"best={best_accuracy * 100:.2f}% | "
            f"lr={scheduler.get_last_lr()[0]:.5f}"
        )

    print(
        f"Best checkpoint saved to {output_path} "
        f"with accuracy {best_accuracy * 100:.2f}%"
    )


if __name__ == "__main__":
    main()