import argparse
import pathlib
import time
from typing import Dict, Tuple, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def model_registry() -> Dict[str, Tuple]:
    return {
        "resnet18": (models.resnet18, getattr(models, "ResNet18_Weights", None)),
        "resnet50": (models.resnet50, getattr(models, "ResNet50_Weights", None)),
        "vit_b_16": (models.vit_b_16, getattr(models, "ViT_B_16_Weights", None)),
        "vit_b_32": (models.vit_b_32, getattr(models, "ViT_B_32_Weights", None)),
    }


def load_model(name: str, state_path: str = None, pretrained: bool = True, num_classes: int = None) -> nn.Module:
    reg = model_registry()
    if name not in reg:
        raise ValueError(f"Unsupported model {name}")
    build, weights_enum = reg[name]
    weights = weights_enum.DEFAULT if pretrained and weights_enum else None

    state = None
    quantized_state = False
    if state_path:
        state = torch.load(state_path, map_location="cpu")
        quantized_state = any("_packed_params" in k or ".scale" in k or ".zero_point" in k for k in state.keys())

    model = build(weights=weights).eval()
    if num_classes:
        if name.startswith("resnet"):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        else:
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)
    if quantized_state:
        model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        target_device = "cpu"  # dynamic quantized modules are CPU-only
    else:
        target_device = DEVICE

    model = model.to(target_device)
    if state is not None:
        model.load_state_dict(state)
    return model


def count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum((p != 0).sum().item() for p in model.parameters() if p.is_floating_point())
    return total, nonzero


@torch.no_grad()
def measure_latency(model: nn.Module, batch_size: int = 1, repeats: int = 5, warmup: int = 1) -> Tuple[float, float]:
    model_device = next(model.parameters()).device
    x = torch.randn(batch_size, 3, 224, 224, device=model_device)
    for _ in range(warmup):
        model(x)
    start = time.time()
    for _ in range(repeats):
        model(x)
    duration = time.time() - start
    avg_latency = duration / repeats
    throughput = batch_size / avg_latency
    return avg_latency, throughput


@torch.inference_mode()
def eval_accuracy(model: nn.Module, loader: DataLoader, max_samples: int = 0) -> float:
    model_device = next(model.parameters()).device
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(model_device), labels.to(model_device)
        preds = model(images).argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        if max_samples and total >= max_samples:
            break
    return correct / total if total else 0.0


def file_size_mb(path: pathlib.Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, length: int, num_classes: int = 10):
        self.length = length
        self.num_classes = num_classes

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.randn(3, 224, 224)
        y = torch.randint(0, self.num_classes, (1,)).item()
        return x, y


def build_loader(
    data_root: str,
    dataset: str = "cifar10",
    batch_size: int = 64,
    download: bool = True,
    synthetic_samples: int = 0,
    max_eval_samples: int = 0,
):
    if synthetic_samples > 0:
        ds = SyntheticDataset(synthetic_samples, num_classes=100 if dataset == "cifar100" else 10)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    try:
        if dataset == "cifar10":
            ds = datasets.CIFAR10(root=data_root, train=False, transform=tfm, download=False)
        elif dataset == "cifar100":
            ds = datasets.CIFAR100(root=data_root, train=False, transform=tfm, download=False)
        else:
            raise ValueError("Unsupported dataset")
    except RuntimeError:
        if not download:
            raise
        # Retry with download if missing or corrupted.
        if dataset == "cifar10":
            ds = datasets.CIFAR10(root=data_root, train=False, transform=tfm, download=True)
        else:
            ds = datasets.CIFAR100(root=data_root, train=False, transform=tfm, download=True)
    if max_eval_samples and max_eval_samples < len(ds):
        # Use a subset to speed up quick benchmarks.
        indices = list(range(max_eval_samples))
        ds = torch.utils.data.Subset(ds, indices)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())


def run_pair(
    name: str,
    orig_path: Optional[pathlib.Path],
    compressed_path: pathlib.Path,
    data_root: str,
    dataset: str,
    download: bool,
    synthetic_samples: int,
    num_classes: int,
    max_eval_samples: int,
    eval_batch_size: int,
    latency_repeats: int,
    latency_warmup: int,
    skip_accuracy: bool,
):
    if not compressed_path.exists():
        raise FileNotFoundError(f"Compressed checkpoint not found: {compressed_path}")

    print(f"\n=== {name} ===")
    # Original model
    if orig_path is not None:
        orig = load_model(name, pretrained=False, state_path=str(orig_path), num_classes=num_classes)
        orig_size_mb = file_size_mb(orig_path)
    else:
        orig = load_model(name, pretrained=True, state_path=None, num_classes=num_classes)
        orig_size_mb = 0.0
    orig_params, orig_nz = count_params(orig)
    orig_lat, orig_tp = measure_latency(orig, repeats=latency_repeats, warmup=latency_warmup)
    loader = build_loader(
        data_root,
        dataset=dataset,
        download=download,
        synthetic_samples=synthetic_samples,
        batch_size=eval_batch_size,
        max_eval_samples=max_eval_samples,
    )
    if skip_accuracy:
        orig_acc = None
    else:
        print(f"Evaluating accuracy for original {name} on {len(loader.dataset)} samples...")
        orig_acc = eval_accuracy(orig, loader, max_samples=max_eval_samples)
    print(
        f"original: params={orig_params:,}, nonzero={orig_nz:,}, sparsity={(1 - orig_nz/orig_params)*100:.2f}%, "
        f"latency={orig_lat*1000:.2f} ms, throughput={orig_tp:.2f} img/s, "
        f"acc={'skipped' if orig_acc is None else f'{orig_acc*100:.2f}%'}, "
        f"file_size={orig_size_mb:.2f} MB"
    )

    # Compressed model
    comp = load_model(name, pretrained=False, state_path=str(compressed_path), num_classes=num_classes)
    comp_params, comp_nz = count_params(comp)
    comp_lat, comp_tp = measure_latency(comp, repeats=latency_repeats, warmup=latency_warmup)
    if skip_accuracy:
        comp_acc = None
    else:
        print(f"Evaluating accuracy for compressed {name} on {len(loader.dataset)} samples...")
        comp_acc = eval_accuracy(comp, loader, max_samples=max_eval_samples)
    size_mb = file_size_mb(compressed_path)
    print(
        f"compressed: params={comp_params:,}, nonzero={comp_nz:,}, sparsity={(1 - comp_nz/comp_params)*100:.2f}%, "
        f"latency={comp_lat*1000:.2f} ms, throughput={comp_tp:.2f} img/s, "
        f"acc={'skipped' if comp_acc is None else f'{comp_acc*100:.2f}%'}, "
        f"file_size={size_mb:.2f} MB"
    )


def parse_args():
    ap = argparse.ArgumentParser(description="Benchmark original vs compressed models.")
    ap.add_argument("--data-root", type=str, default="data", help="Dataset root (auto-download if missing).")
    ap.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--num-classes", type=int, default=10, help="Number of classes for classifier heads.")
    ap.add_argument("--resnet-path", type=str, default="compressed_models/compressed_resnet18.pt")
    ap.add_argument(
        "--resnet-orig",
        type=str,
        default="checkpoints/resnet18_cifar10.pth",
        help="Path to original ResNet checkpoint. If omitted, uses pretrained weights.",
    )
    ap.add_argument("--vit-path", type=str, default="compressed_models/compressed_vit_b_16.pt")
    ap.add_argument(
        "--vit-orig",
        type=str,
        default="checkpoints/vit_b_16_cifar10.pth",
        help="Path to original ViT checkpoint. If omitted, uses pretrained weights.",
    )
    ap.add_argument("--no-download", action="store_true", help="Do not download dataset if missing.")
    ap.add_argument("--max-eval-samples", type=int, default=500, help="Cap evaluation samples for faster runs (0 = full dataset).")
    ap.add_argument("--eval-batch-size", type=int, default=128, help="Batch size for evaluation loader.")
    ap.add_argument("--latency-repeats", type=int, default=5, help="Forward passes for latency averaging.")
    ap.add_argument("--latency-warmup", type=int, default=1, help="Warmup passes before measuring latency.")
    ap.add_argument("--skip-accuracy", action="store_true", help="Skip accuracy evaluation to save time.")
    ap.add_argument(
        "--levels",
        type=int,
        action="append",
        default=None,
        help="If set, benchmark these compression levels using pattern paths instead of single paths.",
    )
    ap.add_argument(
        "--resnet-pattern",
        type=str,
        default="compressed_models/compressed_resnet18_l{level}.pt",
        help="Path pattern for resnet compressed checkpoints when using --levels.",
    )
    ap.add_argument(
        "--vit-pattern",
        type=str,
        default="compressed_models/compressed_vit_b_16_l{level}.pt",
        help="Path pattern for vit compressed checkpoints when using --levels.",
    )
    ap.add_argument(
        "--synthetic-samples",
        type=int,
        default=0,
        help="If >0, use synthetic data with this many samples instead of a real dataset.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    levels = args.levels or []
    if levels:
        for lvl in levels:
            print(f"\n### Compression level {lvl} ###")
            run_pair(
                "resnet18",
                pathlib.Path(args.resnet_orig) if args.resnet_orig else None,
                pathlib.Path(args.resnet_pattern.format(level=lvl)),
                args.data_root,
                args.dataset,
                download=not args.no_download,
                synthetic_samples=args.synthetic_samples,
                num_classes=args.num_classes,
                max_eval_samples=args.max_eval_samples,
                eval_batch_size=args.eval_batch_size,
                latency_repeats=args.latency_repeats,
                latency_warmup=args.latency_warmup,
                skip_accuracy=args.skip_accuracy,
            )
            run_pair(
                "vit_b_16",
                pathlib.Path(args.vit_orig) if args.vit_orig else None,
                pathlib.Path(args.vit_pattern.format(level=lvl)),
                args.data_root,
                args.dataset,
                download=not args.no_download,
                synthetic_samples=args.synthetic_samples,
                num_classes=args.num_classes,
                max_eval_samples=args.max_eval_samples,
                eval_batch_size=args.eval_batch_size,
                latency_repeats=args.latency_repeats,
                latency_warmup=args.latency_warmup,
                skip_accuracy=args.skip_accuracy,
            )
    else:
        run_pair(
            "resnet18",
            pathlib.Path(args.resnet_orig) if args.resnet_orig else None,
            pathlib.Path(args.resnet_path),
            args.data_root,
            args.dataset,
            download=not args.no_download,
            synthetic_samples=args.synthetic_samples,
            num_classes=args.num_classes,
            max_eval_samples=args.max_eval_samples,
            eval_batch_size=args.eval_batch_size,
            latency_repeats=args.latency_repeats,
            latency_warmup=args.latency_warmup,
            skip_accuracy=args.skip_accuracy,
        )
        run_pair(
            "vit_b_16",
            pathlib.Path(args.vit_orig) if args.vit_orig else None,
            pathlib.Path(args.vit_path),
            args.data_root,
            args.dataset,
            download=not args.no_download,
            synthetic_samples=args.synthetic_samples,
            num_classes=args.num_classes,
            max_eval_samples=args.max_eval_samples,
            eval_batch_size=args.eval_batch_size,
            latency_repeats=args.latency_repeats,
            latency_warmup=args.latency_warmup,
            skip_accuracy=args.skip_accuracy,
        )


if __name__ == "__main__":
    main()
