import argparse
import pathlib
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn
from torch.nn.utils import prune
from torchvision import models


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


def build_model(
    model_name: str,
    pretrained: bool,
    num_classes: Optional[int] = None,
) -> nn.Module:
    """
    Build a torchvision model and optionally replace its classification head.

    Args:
        model_name: Model architecture name.
        pretrained: Whether to load pretrained weights.
        num_classes: Number of output classes for the classifier head.

    Returns:
        Instantiated PyTorch model in eval mode.
    """
    registry = get_model_registry()
    if model_name not in registry:
        raise ValueError(
            f"Unsupported model '{model_name}'. Choices: {', '.join(registry)}"
        )

    builder, weights_enum = registry[model_name]
    weights = weights_enum.DEFAULT if pretrained and weights_enum else None
    model = builder(weights=weights)

    if num_classes is not None:
        if model_name.startswith("resnet"):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        else:
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)

    model.eval()
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: Optional[str]) -> nn.Module:
    """
    Load a checkpoint into the model if provided.
    """
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
    return model


def apply_pruning(model: nn.Module, prune_amount: float) -> nn.Module:
    """
    Apply unstructured L1 pruning to all Conv2d and Linear layers.
    """
    prune_amount = max(0.0, prune_amount)
    if prune_amount <= 0.0:
        return model

    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, "weight"))

    for module, param_name in parameters_to_prune:
        prune.l1_unstructured(module, name=param_name, amount=prune_amount)

    # Remove reparameterization for clean state_dict saving
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)

    return model


def apply_module_pruning(
    module: nn.Module,
    param_name: str,
    prune_amount: float,
    method: str = "l1",
) -> nn.Module:
    """
    Apply pruning to a single module parameter.
    """
    if method == "l1":
        prune.l1_unstructured(
            module,
            name=param_name,
            amount=prune_amount,
        )
        prune.remove(module, param_name)
    else:
        # TODO: add additional pruning methods
        raise NotImplementedError(f"Pruning method '{method}' is not supported.")

    return module


def apply_quantization(model: nn.Module) -> nn.Module:
    """
    Apply post-training dynamic quantization to Linear layers.
    """
    return torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )


def save_model(model: nn.Module, output_path: pathlib.Path) -> None:
    """
    Save model state dictionary to disk.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)


def compression_defaults(
    model_name: str,
    compression_level: int,
) -> Tuple[float, bool]:
    """
    Return default (prune_amount, quantize) settings
    for a given model and compression level (1–5).
    """
    if compression_level < 1 or compression_level > 5:
        raise ValueError("compression_level must be between 1 and 5")

    if model_name.startswith("resnet"):
        prune_levels = [0.10, 0.25, 0.40, 0.55, 0.70]
        quantize = compression_level >= 3
    elif model_name.startswith("vit"):
        prune_levels = [0.05, 0.10, 0.15, 0.25, 0.35]
        quantize = compression_level >= 4
    else:
        prune_levels = [0.1] * 5
        quantize = False

    return prune_levels[compression_level - 1], quantize


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Prune and quantize ResNet or Vision Transformer models."
    )
    parser.add_argument(
        "--model",
        dest="model_names",
        action="append",
        choices=list(get_model_registry().keys()),
        help=(
            "Model architecture to compress. Can be passed multiple times. "
            "Defaults to resnet18 and vit_b_16."
        ),
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable loading torchvision pretrained weights.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to load before compression.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Override classifier head output dimension.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=3,
        choices=[1, 2, 3, 4, 5],
        help="Compression strength level (1–5).",
    )
    parser.add_argument(
        "--prune-amount",
        type=float,
        default=None,
        help="Fraction of weights to prune per layer (0.0–1.0).",
    )

    quant_group = parser.add_mutually_exclusive_group()
    quant_group.add_argument(
        "--quantize",
        dest="quantize",
        action="store_true",
        help="Enable dynamic quantization.",
    )
    quant_group.add_argument(
        "--no-quantize",
        dest="quantize",
        action="store_false",
        help="Disable quantization.",
    )
    parser.set_defaults(quantize=None)

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for single-model compression.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="compressed_models",
        help="Directory for saving compressed models.",
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
    """
    Compress one or more models using pruning and optional quantization.
    """
    registry = get_model_registry()
    model_list = list(model_names)

    for model_name in model_list:
        if model_name not in registry:
            raise ValueError(
                f"Unsupported model '{model_name}'. Choices: {', '.join(registry)}"
            )

    if checkpoint and len(model_list) > 1:
        raise ValueError(
            "Checkpoint loading is only supported for single-model compression."
        )

    for model_name in model_list:
        model = build_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )
        model = load_checkpoint(model, checkpoint)

        level_prune, level_quant = (
            compression_defaults(model_name, compression_level)
            if compression_level
            else (0.2, False)
        )

        effective_prune = prune_amount if prune_amount is not None else level_prune
        effective_quant = quantize if quantize is not None else level_quant

        model = apply_pruning(model, effective_prune)

        if effective_quant:
            model = apply_quantization(model)

        if output and len(model_list) == 1:
            output_path = pathlib.Path(output)
        else:
            output_path = pathlib.Path(output_dir) / f"compressed_{model_name}.pt"

        save_model(model, output_path)

        print(f"Compressed model saved to {output_path}")
        print(
            f"Options -> model: {model_name}, "
            f"prune_amount: {effective_prune}, "
            f"quantized: {effective_quant}, "
            f"checkpoint: {checkpoint or 'none'}, "
            f"level: {compression_level or 'none'}"
        )


def main() -> None:
    args = parse_args()

    default_models = ["resnet18", "vit_b_16"]
    models_to_run = args.model_names or default_models

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
