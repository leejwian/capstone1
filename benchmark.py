import pathlib
import time
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_model_registry() -> Dict[str, Tuple]:
    """
    Return a mapping from model name to
    (model_builder_function, pretrained_weights_enum).
    """
    return {
        "resnet18": (models.resnet18, getattr(models, "ResNet18_Weights", None)),
        "resnet50": (models.resnet50, getattr(models, "ResNet50_Weights", None)),
        "vit_b_16": (models.vit_b_16, getattr(models, "ViT_B_16_Weights", None)),
        "vit_b_32": (models.vit_b_32, getattr(models, "ViT_B_32_Weights", None)),
    }


def load_model(
    model_name: str,
    state_path: Optional[str] = None,
    pretrained: bool = True,
    num_classes: Optional[int] = None,
) -> nn.Module:
    """
    Load a model and optionally restore a checkpoint.
    Automatically handles dynamic-quantized checkpoints.
    """
    registry = get_model_registry()
    if model_name not in registry:
        raise ValueError(f"Unsupported model '{model_name}'")

    builder, weights_enum = registry[model_name]
    weights = weights_enum.DEFAULT if pretrained and weights_enum else None

    state_dict = None
    is_quantized = False
    if state_path:
        state_dict = torch.load(state_path, map_location="cpu")
        is_quantized = any(
            "_packed_params" in key or ".scale" in key or ".zero_point" in key
            for key in state_dict.keys()
        )

    model = builder(weights=weights).eval()

    if num_classes is not None:
        if model_name.startswith("resnet"):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    if is_quantized:
        model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8,
        )
        target_device = "cpu"
    else:
        target_device = DEVICE

    model = model.to(target_device)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and non-zero parameters in a model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    nonzero_params = sum(
        (p != 0).sum().item()
        for p in model.parameters()
        if p.is_floating_point()
    )
    return total_params, nonzero_params


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    batch_size: int = 1,
    repeats: int = 5,
    warmup: int = 1,
) -> Tuple[float, float]:
    """
    Measure average inference latency and throughput.
    """
    device = next(model.parameters()).device
    inputs = torch.randn(batch_size, 3, 224, 224, device=device)

    for _ in range(warmup):
        model(inputs)

    start_time = time.time()
    for _ in range(repeats):
        model(inputs)
    elapsed = time.time() - start_time

    avg_latency = elapsed / repeats
    throughput = batch_size / avg_latency
    return avg_latency, throughput


@torch.inference_mode()
def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    max_samples: int = 0,
) -> float:
    """
    Evaluate top-1 accuracy.
    """
    device = next(model.parameters()).device
    correct, total = 0, 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        if max_samples and total >= max_samples:
            break

    return correct / total if total else 0.0


def get_file_size_mb(path: pathlib.Path) -> float:
    """
    Return file size in megabytes.
    """
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for quick benchmarking without real data.
    """

    def __init__(self, length: int, num_classes: int):
        self.length = length
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        image = torch.randn(3, 224, 224)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return image, label


def build_dataloader(
    data_root: str,
    dataset: str,
    batch_size: int,
    download: bool,
    synthetic_samples: int,
    max_eval_samples: int,
) -> DataLoader:
    """
    Build evaluation DataLoader (real or synthetic).
    """
    if synthetic_samples > 0:
        num_classes = 100 if dataset == "cifar100" else 10
        dataset_obj = SyntheticDataset(synthetic_samples, num_classes)
        return DataLoader(
            dataset_obj,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    try:
        if dataset == "cifar10":
            dataset_obj = datasets.CIFAR10(
                root=data_root,
                train=False,
                transform=transform,
                download=False,
            )
        elif dataset == "cifar100":
            dataset_obj = datasets.CIFAR100(
                root=data_root,
                train=False,
                transform=transform,
                download=False,
            )
        else:
            raise ValueError("Unsupported dataset")
    except RuntimeError:
        if not download:
            raise
        dataset_obj = datasets.CIFAR10(
            root=data_root,
            train=False,
            transform=transform,
            download=True,
        )

    if max_eval_samples and max_eval_samples < len(dataset_obj):
        dataset_obj = torch.utils.data.Subset(
            dataset_obj,
            list(range(max_eval_samples)),
        )

    return DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
