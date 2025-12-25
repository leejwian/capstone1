import argparse
import pathlib
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn
from torch.nn.utils import prune
from torchvision import models


def _model_registry() -> Dict[str, Tuple]:
    """
    Returns mapping of model names to (builder_fn, weights_enum_or_None).
    """
    return {
        "resnet18": (models.resnet18, getattr(models, "ResNet18_Weights", None)),
        "resnet50": (models.resnet50, getattr(models, "ResNet50_Weights", None)),
        "vit_b_16": (models.vit_b_16, getattr(models, "ViT_B_16_Weights", None)),
        "vit_b_32": (models.vit_b_32, getattr(models, "ViT_B_32_Weights", None)),
    }


def build_model(name: str, pretrained: bool, num_classes: Optional[int] = None) -> nn.Module:
    registry = _model_registry()
    if name not in registry:
        raise ValueError(f"Unsupported model '{name}'. Choices: {', '.join(registry)}")

    builder, weights_enum = registry[name]
    weights = weights_enum.DEFAULT if pretrained and weights_enum is not None else None
    model = builder(weights=weights)
    if num_classes:
        if name.startswith("resnet"):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        else:
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)
    model.eval()
    return model


def load_checkpoint(model: nn.Module, checkpoint: Optional[str]) -> nn.Module:
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state)
    return model


def apply_pruning(model: nn.Module, amount: float) -> nn.Module:
    if amount <= 0:
        return model

    to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            to_prune.append((module, "weight"))

    for module, param_name in to_prune:
        prune.l1_unstructured(module, name=param_name, amount=amount)

    # Remove pruning reparameterization so the graph stays clean after saving.
    for module, param_name in to_prune:
        prune.remove(module, param_name)

    return model


def apply_quantization(model: nn.Module) -> nn.Module:
    # Dynamic quantization targets Linear layers, which helps both ResNet heads and ViT blocks.
    return torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )


def save_model(model: nn.Module, output_path: pathlib.Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)


def compression_defaults(model_name: str, level: int) -> Tuple[float, bool]:
    """
    Returns (prune_amount, quantize) defaults for a given model and compression level 1-5.
    Levels are more aggressive as the number increases, spaced to make differences visible.
    """
    if level < 1 or level > 5:
        raise ValueError("compression level must be between 1 and 5")

    if model_name.startswith("resnet"):
        # ResNet: widen gaps so each step changes sparsity/latency more noticeably.
        prune_levels = [0.10, 0.25, 0.40, 0.55, 0.70]
        # Quantize from level 3 and higher to make speed/size differences clearer (CPU-only after quantize).
        quantize = level >= 3
    elif model_name.startswith("vit"):
        # ViT: keep early levels mild, later levels more aggressive; quantize from level 4 up.
        prune_levels = [0.05, 0.10, 0.15, 0.25, 0.35]
        quantize = level >= 4
    else:
        prune_levels = [0.1] * 5
        quantize = False
    return prune_levels[level - 1], quantize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prune and quantize ResNet or Vision Transformer models."
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        choices=list(_model_registry().keys()),
        help="Model architecture to compress. Can be passed multiple times. Defaults to resnet18 and vit_b_16.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Skip loading torchvision pretrained weights.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a local checkpoint (.pt/.pth) to load before compression.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Set classifier head to this many classes before loading checkpoint (e.g., 10 for CIFAR-10).",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=3,
        choices=[1, 2, 3, 4, 5],
        help="Compression strength level (1-5). Maps to model-specific prune/quantize defaults. Overrides --prune-amount if not provided.",
    )
    parser.add_argument(
        "--prune-amount",
        type=float,
        default=None,
        help="Fraction of weights to prune per layer (0.0-1.0). If omitted, uses --compression-level defaults or 0.2.",
    )
    q_group = parser.add_mutually_exclusive_group()
    q_group.add_argument(
        "--quantize",
        dest="quantize",
        action="store_true",
        help="Apply dynamic quantization to linear layers.",
    )
    q_group.add_argument(
        "--no-quantize",
        dest="quantize",
        action="store_false",
        help="Disable quantization even if compression level suggests it.",
    )
    parser.set_defaults(quantize=None)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the compressed model when compressing a single model.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="compressed_models",
        help="Directory to save compressed models when compressing multiple architectures or when --output is omitted.",
    )
    return parser.parse_args()


def compress_models(
    model_names: Iterable[str],
    pretrained: bool,
    checkpoint: Optional[str],
    prune_amount: Optional[float],
    quantize: Optional[bool],
    compression_level: Optional[int],
    num_classes: Optional[int],
    output: Optional[str],
    output_dir: str,
) -> None:
    registry = _model_registry()
    model_list = list(model_names)

    for name in model_list:
        if name not in registry:
            raise ValueError(f"Unsupported model '{name}'. Choices: {', '.join(registry)}")

    if checkpoint and len(model_list) > 1:
        raise ValueError("When compressing multiple models, omit --checkpoint or run one model at a time.")

    for name in model_list:
        model = build_model(name, pretrained=pretrained, num_classes=num_classes)
        model = load_checkpoint(model, checkpoint)

        level_prune, level_quant = compression_defaults(name, compression_level) if compression_level else (0.2, False)
        effective_prune = prune_amount if prune_amount is not None else level_prune
        effective_quant = quantize if quantize is not None else level_quant

        model = apply_pruning(model, amount=effective_prune)

        if effective_quant:
            model = apply_quantization(model)

        if output and len(model_list) == 1:
            output_path = pathlib.Path(output)
        else:
            output_path = pathlib.Path(output_dir) / f"compressed_{name}.pt"

        save_model(model, output_path)

        print(f"Compressed model saved to {output_path}")
        print(
            f"Options -> model: {name}, prune_amount: {effective_prune}, "
            f"quantized: {effective_quant}, checkpoint: {checkpoint or 'none'}, level: {compression_level or 'none'}"
        )


def main() -> None:
    args = parse_args()

    default_models = ["resnet18", "vit_b_16"]
    models_to_run = args.models or default_models

    compress_models(
        model_names=models_to_run,
        pretrained=not args.no_pretrained,
        checkpoint=args.checkpoint,
        prune_amount=args.prune_amount,
        quantize=args.quantize,
        compression_level=args.compression_level,
        num_classes=args.num_classes,
        output=args.output,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
