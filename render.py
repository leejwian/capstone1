"""
Gradio 기반 모델 경량화 및 벤치마크 UI.

사전학습 Torchvision 모델 및 사용자 정의 모델에 대해
Pruning 및 Quantization을 적용하고 성능을 비교한다.
"""
import pathlib
from typing import Optional, List

import torch
from torch import nn
import gradio as gr

from compress import (
    apply_pruning,
    apply_module_pruning,
    apply_quantization,
    save_model,
    compression_defaults,
)
from benchmark import (
    count_params,
    measure_latency,
    file_size_mb,
    eval_accuracy,
    build_loader,
)
from models import (
    CustomModelLoader,
    default_model_loader,
)

# ---------------------------------------------------------------------
# Device 설정
# Dynamic quantization은 CUDA에서 제한이 있어 CPU로 고정
# ---------------------------------------------------------------------
DEVICE = "cpu"
print(f"Device: {DEVICE}")

# 기본 모델 로더
model_loader = default_model_loader


def compress_and_benchmark(
    model_source: str,
    model_name_or_path: str,
    compression_mode: str,
    selected_modules: Optional[List[str]],
    compression_level: int,
    prune_amount: Optional[float],
    apply_quantization_flag: bool,
    checkpoint_path: Optional[str],
    output_filename: str,
) -> str:
    """
    모델을 경량화하고 벤치마크 결과를 문자열로 반환한다.

    Args:
        model_source: "pretrained" 또는 "custom"
        model_name_or_path: 모델 이름 또는 파일 경로
        compression_mode: "level" 또는 "modules"
        selected_modules: 모듈 단위 경량화 시 선택된 모듈 목록
        compression_level: Compression level (1–5)
        prune_amount: 수동 pruning 비율 (None이면 level 기본값 사용)
        apply_quantization_flag: Quantization 강제 적용 여부
        checkpoint_path: 로드할 체크포인트 경로
        output_filename: 저장 파일 이름

    Returns:
        벤치마크 결과 문자열
    """
    try:
        output_dir = pathlib.Path("compressed_models")
        output_dir.mkdir(parents=True, exist_ok=True)

        # --------------------------------------------------------------
        # 모델 로드
        # --------------------------------------------------------------
        if model_source == "pretrained":
            orig_model = model_loader.load_model(
                model_name_or_path,
                pretrained=True,
                num_classes=10,
                checkpoint=None,
            )
            display_name = model_name_or_path
        else:
            custom_loader = CustomModelLoader()
            orig_model = custom_loader.load_model(
                model_name_or_path,
                checkpoint=None,
            )
            display_name = pathlib.Path(model_name_or_path).name

        orig_model = orig_model.to(DEVICE)

        # --------------------------------------------------------------
        # 원본 모델 벤치마크
        # --------------------------------------------------------------
        orig_params, orig_nz = count_params(orig_model)
        orig_latency, orig_throughput = measure_latency(
            orig_model,
            batch_size=1,
            repeats=10,
            warmup=1,
        )

        try:
            loader = build_loader(
                "data",
                dataset="cifar10",
                batch_size=64,
                synthetic_samples=500,
            )
            orig_accuracy = eval_accuracy(orig_model, loader)
        except Exception as err:
            print(f"Accuracy measurement skipped: {err}")
            orig_accuracy = None

        # --------------------------------------------------------------
        # 결과 문자열 (원본 모델)
        # --------------------------------------------------------------
        header = (
            "Compression Level"
            if compression_mode == "level"
            else "Selected Modules"
        )
        header_value = (
            compression_level
            if compression_mode == "level"
            else selected_modules
        )

        result = (
            "=" * 70 + "\n"
            f"Model: {display_name.upper()} | "
            f"{header}: {header_value}\n"
            + "=" * 70 + "\n\n"
        )

        result += (
            "[ORIGINAL MODEL]\n"
            f"  Parameters: {orig_params:,}\n"
            f"  Non-zero: {orig_nz:,}\n"
            f"  Sparsity: {(1 - orig_nz / orig_params) * 100:.2f}%\n"
            f"  Latency: {orig_latency * 1000:.2f} ms\n"
            f"  Throughput: {orig_throughput:.2f} img/s\n"
            f"  Accuracy: "
            f"{'skipped' if orig_accuracy is None else f'{orig_accuracy * 100:.2f}%'}\n\n"
        )

        # --------------------------------------------------------------
        # 경량화 설정
        # --------------------------------------------------------------
        level_prune, level_quant = compression_defaults(
            display_name,
            compression_level,
        )
        effective_prune = (
            level_prune if compression_mode == "level" else prune_amount
        )
        effective_quant = (
            apply_quantization_flag if apply_quantization_flag else level_quant
        )

        # --------------------------------------------------------------
        # 경량화 대상 모델 생성
        # --------------------------------------------------------------
        if model_source == "pretrained":
            comp_model = model_loader.load_model(
                model_name_or_path,
                pretrained=True,
                num_classes=10,
                checkpoint=checkpoint_path,
            )
        else:
            comp_model = custom_loader.load_model(
                model_name_or_path,
                checkpoint=checkpoint_path,
            )

        # --------------------------------------------------------------
        # Pruning 적용
        # --------------------------------------------------------------
        if compression_mode == "level":
            comp_model = apply_pruning(comp_model, amount=effective_prune)
        elif compression_mode == "modules":
            if not selected_modules:
                return "No modules selected for module-based compression."

            module_map = dict(comp_model.named_modules())
            for module_name in selected_modules:
                module = module_map.get(module_name)
                if module is not None and hasattr(module, "weight"):
                    apply_module_pruning(
                        module,
                        param_name="weight",
                        amount=effective_prune,
                        method="l1",
                    )
        else:
            return f"Unknown compression mode: {compression_mode}"

        # --------------------------------------------------------------
        # Quantization 적용
        # --------------------------------------------------------------
        if effective_quant:
            comp_model = apply_quantization(comp_model)

        comp_model = comp_model.to(DEVICE)

        # --------------------------------------------------------------
        # 압축 모델 벤치마크
        # --------------------------------------------------------------
        comp_params, comp_nz = count_params(comp_model)
        comp_latency, comp_throughput = measure_latency(
            comp_model,
            batch_size=1,
            repeats=10,
            warmup=1,
        )

        try:
            comp_accuracy = eval_accuracy(comp_model, loader)
        except Exception as err:
            print(f"Compressed accuracy skipped: {err}")
            comp_accuracy = None

        result += (
            "[COMPRESSED MODEL]\n"
            f"  Parameters: {comp_params:,}\n"
            f"  Non-zero: {comp_nz:,}\n"
            f"  Sparsity: {(1 - comp_nz / comp_params) * 100:.2f}%\n"
            f"  Latency: {comp_latency * 1000:.2f} ms\n"
            f"  Throughput: {comp_throughput:.2f} img/s\n"
            f"  Accuracy: "
            f"{'skipped' if comp_accuracy is None else f'{comp_accuracy * 100:.2f}%'}\n\n"
        )

        # --------------------------------------------------------------
        # 개선도 계산
        # --------------------------------------------------------------
        param_reduction = (
            (1 - comp_params / orig_params) * 100
            if orig_params > 0
            else 0.0
        )
        latency_improvement = (
            (1 - comp_latency / orig_latency) * 100
            if orig_latency > 0
            else 0.0
        )

        result += (
            "[IMPROVEMENTS]\n"
            f"  Parameter Reduction: {param_reduction:.2f}%\n"
            f"  Latency Improvement: {latency_improvement:.2f}%\n"
            f"  Speedup: {orig_latency / comp_latency:.2f}x\n"
        )

        if orig_accuracy is not None and comp_accuracy is not None:
            accuracy_drop = (comp_accuracy - orig_accuracy) * 100
            result += f"  Accuracy Drop: {accuracy_drop:.2f}%\n"

        # --------------------------------------------------------------
        # 모델 저장
        # --------------------------------------------------------------
        output_path = output_dir / f"{output_filename}.pt"
        save_model(comp_model, output_path)
        size_mb = file_size_mb(output_path)

        result += (
            "\n[COMPRESSION SETTINGS]\n"
            f"  Pruning Amount: {effective_prune:.2f}\n"
            f"  Quantized: {effective_quant}\n"
            f"  Output Size: {size_mb:.2f} MB\n"
            f"  Saved to: {output_path}\n"
            + "=" * 70
        )

        return result

    except Exception as err:
        import traceback

        return f"Error: {err}\n\n{traceback.format_exc()}"
