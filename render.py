import pathlib
from typing import Optional

import torch
from torch import nn
import gradio as gr

# 외부 모듈 import
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
    TorchvisionModelLoader,
    CustomModelLoader,
    default_model_loader,
)

# Device 설정
#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#cuda로 dynamic_qunatiz로 사용시 오류가 있어서 cpu로 고정합니다.
DEVICE = "cpu"
print(f"Device: {DEVICE}")

# 모델 로더 초기화
model_loader = default_model_loader


def compress_and_benchmark(
    model_source: str,
    model_name_or_path: str,
    compression_mode: str,
    selected_modules: Optional[list],
    compression_level: int,
    prune_amount: Optional[float],
    apply_quantization_flag: bool,
    checkpoint_path: Optional[str],
    output_filename: str,
) -> str:
    """
    모델을 경량화하고 벤치마크 결과를 표시
    
    Args:
        model_source: 모델 출처 ("pretrained" 또는 "checkpoint")
        model_name_or_path: 모델 이름 (pretrained) 또는 경로 (checkpoint)
        compression_mode: "level" or "modules"
        selected_modules: 경량화할 modules 목록
        compression_level: Compression level (1-5)
        prune_amount: 수동 pruning amount (None이면 level 기본값 사용)
        apply_quantization_flag: Quantization 적용 여부
        checkpoint_path: 경량화 전 로드할 체크포인트 경로
        output_filename: 출력 파일 이름
    """
    try:
        output_dir = pathlib.Path("compressed_models")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 로드 방식 결정
        if model_source == "pretrained":
            # Torchvision 사전학습 모델
            orig_model = model_loader.load_model(
                model_name_or_path,
                pretrained=True,
                num_classes=10,
                checkpoint=None
            )
            display_name = model_name_or_path
        else:
            # 커스텀 경로의 모델
            custom_loader = CustomModelLoader()
            try:
                orig_model = custom_loader.load_model(
                    model_name_or_path,
                    checkpoint=None
                )
            except Exception as e:
                return f"커스텀 모델 로드 실패: {str(e)}"
            display_name = pathlib.Path(model_name_or_path).name
        
        orig_model = orig_model.to(DEVICE)
        
        # 원본 모델 벤치마크
        orig_params, orig_nz = count_params(orig_model)
        orig_lat, orig_tp = measure_latency(orig_model, batch_size=1, repeats=10, warmup=1)
        
        # 원본 모델 accuracy 측정
        try:
            loader = build_loader("data", dataset="cifar10", batch_size=64, synthetic_samples=500)
            orig_acc = eval_accuracy(orig_model, loader)
        except Exception as acc_err:
            orig_acc = None
            print(f"Accuracy measurement skipped: {acc_err}")
        
        result = "=" * 70 + "\n"
        result += f"Model: {display_name.upper()} | {"Compression Level" if compression_mode=="level" else "Selected Modules"}: {compression_level if compression_mode=="level" else selected_modules}\n"
        result += "=" * 70 + "\n\n"
        
        result += "[ORIGINAL MODEL]\n"
        result += f"  Parameters: {orig_params:,}\n"
        result += f"  Non-zero: {orig_nz:,}\n"
        result += f"  Sparsity: {(1 - orig_nz/orig_params)*100:.2f}%\n"
        result += f"  Latency: {orig_lat*1000:.2f} ms\n"
        result += f"  Throughput: {orig_tp:.2f} img/s\n"
        result += f"  Accuracy: {'skipped' if orig_acc is None else f'{orig_acc*100:.2f}%'}\n\n"
        
        # 경량화 파라미터 결정
        level_prune, level_quant = compression_defaults(display_name, compression_level)
        effective_prune = level_prune if compression_mode=="level" else prune_amount
        effective_quant = apply_quantization_flag if apply_quantization_flag else level_quant

        # 경량화된 모델 생성
        if model_source == "pretrained":
            comp_model = model_loader.load_model(
                model_name_or_path,
                pretrained=True,
                num_classes=10,
                checkpoint=checkpoint_path
            )
        else:
            comp_model = custom_loader.load_model(
                model_name_or_path,
                checkpoint=checkpoint_path
            )

        # Apply compression depending on selected mode
        if compression_mode == "level":
            comp_model = apply_pruning(comp_model, amount=effective_prune)
        elif compression_mode == "modules":
            # selected_modules contains module names to compress
            if not selected_modules:
                return "No modules selected for module-based compression."
            # find modules by name and apply module-level pruning
            name2module = {name: m for name, m in comp_model.named_modules()}
            for mname in selected_modules:
                if mname not in name2module:
                    # ignore unknown
                    continue
                module = name2module[mname]
                if hasattr(module, "weight"):
                    apply_module_pruning(module, param_name="weight", amount=effective_prune, method="l1")
        else:
            return f"Unknown compression mode: {compression_mode}"

        if effective_quant:
            comp_model = apply_quantization(comp_model)
        
        comp_model = comp_model.to(DEVICE)
        comp_params, comp_nz = count_params(comp_model)
        comp_lat, comp_tp = measure_latency(comp_model, batch_size=1, repeats=10, warmup=1)
        
        # 압축된 모델 accuracy 측정
        try:
            comp_acc = eval_accuracy(comp_model, loader)
        except Exception as acc_err:
            comp_acc = None
            print(f"Compressed model accuracy measurement failed: {acc_err}")
        
        result += "[COMPRESSED MODEL]\n"
        result += f"  Parameters: {comp_params:,}\n"
        result += f"  Non-zero: {comp_nz:,}\n"
        result += f"  Sparsity: {(1 - comp_nz/comp_params)*100:.2f}%\n"
        result += f"  Latency: {comp_lat*1000:.2f} ms\n"
        result += f"  Throughput: {comp_tp:.2f} img/s\n"
        result += f"  Accuracy: {'skipped' if comp_acc is None else f'{comp_acc*100:.2f}%'}\n\n"
        
        # 개선도 계산
        param_reduction = (1 - comp_params/orig_params) * 100 if orig_params > 0 else 0
        latency_improvement = (1 - comp_lat/orig_lat) * 100 if orig_lat > 0 else 0
        accuracy_drop = None
        if orig_acc is not None and comp_acc is not None:
            accuracy_drop = (comp_acc - orig_acc) * 100
        
        result += "[IMPROVEMENTS]\n"
        result += f"  Parameter Reduction: {param_reduction:.2f}%\n"
        result += f"  Latency Improvement: {latency_improvement:.2f}%\n"
        result += f"  Speedup: {orig_lat/comp_lat:.2f}x\n"
        if accuracy_drop is not None:
            result += f"  Accuracy Drop: {accuracy_drop:.2f}%\n"
        result += "\n"
        
        # 모델 저장
        output_path = output_dir / f"{output_filename}.pt"
        save_model(comp_model, output_path)
        size_mb = file_size_mb(output_path)
        
        result += "[COMPRESSION SETTINGS]\n"
        result += f"  Pruning Amount: {effective_prune:.2f}\n"
        result += f"  Quantized: {effective_quant}\n"
        result += f"  Output Size: {size_mb:.2f} MB\n"
        result += f"  Saved to: {output_path}\n"
        result += "=" * 70
        
        return result
        
    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\n{traceback.format_exc()}"


def get_available_models() -> list:
    """사용 가능한 모델 목록 반환"""
    registry = model_loader.get_available_models()
    return list(registry.keys())


def get_available_checkpoints() -> list:
    """사용 가능한 체크포인트 파일 목록 반환"""
    checkpoint_dir = pathlib.Path("checkpoints")
    if not checkpoint_dir.exists():
        return []
    return [str(f) for f in checkpoint_dir.glob("*.pth") if f.is_file()]


def get_model_module_names(model_source: str, model_name_or_path: str, checkpoint_path: Optional[str] = None) -> list:
    """주어진 모델에 대해 weight를 가진 모듈 이름 리스트 반환 (CPU에서 로드)
    """
    try:
        if model_source == "pretrained":
            model = model_loader.load_model(model_name_or_path, pretrained=True, num_classes=10, checkpoint=checkpoint_path)
        else:
            loader = CustomModelLoader()
            model = loader.load_model(model_name_or_path, checkpoint=checkpoint_path)
        model = model.to("cpu")
        names = [name for name, mod in model.named_modules() if hasattr(mod, "weight") and name != ""]
        return names
    except Exception:
        return []


# ==================== Gradio Interface ====================
with gr.Blocks(title="Model Compression & Benchmark UI") as demo:
    gr.Markdown("# Model Compression & Benchmark UI")
    gr.Markdown(
        """
        **이미지 분류 모델의 Pruning & Quantization**
        """
    )
    
    # 탭 인터페이스로 두 가지 모드 제공
    with gr.Tabs():
        # ===== 탭 1: 사전학습 모델 =====
        with gr.TabItem("Pretrained Models"):
            gr.Markdown("### 사전학습된 Torchvision 모델 경량화")
            
            with gr.Row():
                model_select = gr.Dropdown(
                    choices=get_available_models(),
                    value=get_available_models()[0] if get_available_models() else "resnet18",
                    label="Model Architecture",
                    info="Select from Torchvision pretrained models"
                )
                
                compression_level = gr.Slider(
                    minimum=1,
                    maximum=5,
                    step=1,
                    value=3,
                    label="Compression Level",
                    info="1 (mild) ~ 5 (aggressive)",
                    interactive=True
                )
                
                module_select_pre = gr.Dropdown(
                    choices=[],
                    multiselect=True,
                    visible=False,
                    label="Select Modules (multi)",
                    info="Choose modules to compress",
                    interactive=True
                )
                
                compression_mode = gr.Radio(
                    choices=["level", "modules"],
                    value="level",
                    label="Compression Mode",
                    info="Choose level-based or per-module compression"
                )
                
            
            with gr.Row(visible=False) as modules_row:
                prune_amount_input = gr.Number(
                    value=None,
                    step=0.01,
                    label="Pruning Amount (optional)",
                    info="Override level default (0.0-1.0). Leave empty to use level default.",
                )
                
                quantize_checkbox = gr.Checkbox(
                    value=False,
                    label="Apply Quantization",
                    info="Enable/disable quantization (int8)"
                )
            
            with gr.Row():
                checkpoint_select = gr.Dropdown(
                    choices=get_available_checkpoints(),
                    value=None,
                    label="Checkpoint (optional)",
                    info="Load pre-trained checkpoint before compression",
                    multiselect=False
                )
            
            with gr.Row():
                output_filename_1 = gr.Textbox(
                    value="compressed_model",
                    label="Output Filename",
                    info="Saved as {filename}.pt in compressed_models/"
                )
            
            with gr.Row():
                compress_btn_1 = gr.Button("🔄 Compress & Benchmark", variant="primary", scale=2)
                refresh_btn = gr.Button("🔄 Refresh Checkpoints", scale=1)
            
            output_text_1 = gr.Textbox(
                label="Results",
                interactive=False,
                lines=28,
                max_lines=45
            )
            
            def update_pretrained_modules(model_name):
                return gr.update(choices=get_model_module_names('pretrained', model_name))

            model_select.change(fn=update_pretrained_modules, inputs=model_select, outputs=module_select_pre)

            def toggle_pre_mode(mode):
                if mode == "modules":
                    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

            # compression_level visible when level mode; module_select_pre visible when modules mode
            compression_mode.change(toggle_pre_mode, inputs=compression_mode, outputs=[module_select_pre, compression_level, modules_row])

            compress_btn_1.click(
                fn=compress_and_benchmark,
                inputs=[
                    gr.State("pretrained"),
                    model_select,
                    compression_mode,
                    module_select_pre,
                    compression_level,
                    prune_amount_input,
                    quantize_checkbox,
                    checkpoint_select,
                    output_filename_1,
                ],
                outputs=output_text_1,
            )
            
            refresh_btn.click(
                fn=lambda: gr.update(choices=get_available_checkpoints()),
                outputs=checkpoint_select,
            )
        
        # ===== 탭 2: 커스텀 모델 =====
        with gr.TabItem("Custom Model"):
            gr.Markdown("### 커스텀 모델 경량화")
            gr.Markdown(
                """
                **지원 형식**: `.pt`, `.pth` 파일
                
                **예시**: `/path/to/my_model.pt`
                """
            )
            
            with gr.Row():
                model_path_input = gr.Textbox(
                    label="Model File Path",
                    placeholder="e.g., /path/to/model.pt",
                    info="Full path to your custom model file"
                )
            
            with gr.Row():
                compression_level_2 = gr.Number(
                    minimum=1,
                    maximum=5,
                    step=1,
                    value=3,
                    label="Compression Level",
                    info="1 (mild) ~ 5 (aggressive)"
                )
                
                module_select_custom = gr.Dropdown(
                    choices=[],
                    multiselect=True,
                    visible=False,
                    label="Select Modules (multi)",
                    info="Choose modules to compress from custom model"
                )
                
                compression_mode_2 = gr.Radio(
                    choices=["level", "modules"],
                    value="level",
                    label="Compression Mode",
                    info="Choose level-based or per-module compression"
                )
                
            
            with gr.Row():
                prune_amount_input_2 = gr.Number(
                    value=None,
                    label="Pruning Amount (optional)",
                    info="0.0-1.0, or leave empty for level default",
                    interactive=True
                )
                
                quantize_checkbox_2 = gr.Checkbox(
                    value=False,
                    label="Apply Quantization",
                    info="Enable/disable quantization (int8)"
                )
            
            with gr.Row():
                checkpoint_select_2 = gr.Textbox(
                    label="Checkpoint Path (optional)",
                    placeholder="e.g., /path/to/checkpoint.pth",
                    info="Checkpoint to load before compression"
                )
            
            with gr.Row():
                output_filename_2 = gr.Textbox(
                    value="compressed_custom_model",
                    label="Output Filename",
                    info="Saved as {filename}.pt in compressed_models/"
                )
            
            compress_btn_2 = gr.Button("🔄 Compress & Benchmark", variant="primary")
            
            output_text_2 = gr.Textbox(
                label="Results",
                interactive=False,
                lines=28,
                max_lines=45
            )
            
            def update_custom_modules(path):
                return gr.update(choices=get_model_module_names('custom', path))

            model_path_input.change(fn=update_custom_modules, inputs=model_path_input, outputs=module_select_custom)

            def toggle_custom_mode(mode):
                if mode == "modules":
                    return gr.update(visible=True), gr.update(visible=False)
                return gr.update(visible=False), gr.update(visible=True)

            compression_mode_2.change(toggle_custom_mode, inputs=compression_mode_2, outputs=[module_select_custom, compression_level_2])

            compress_btn_2.click(
                fn=compress_and_benchmark,
                inputs=[
                    gr.State("custom"),
                    model_path_input,
                    compression_mode_2,
                    module_select_custom,
                    compression_level_2,
                    prune_amount_input_2,
                    quantize_checkbox_2,
                    checkpoint_select_2,
                    output_filename_2,
                ],
                outputs=output_text_2,
            )
    
    # 하단 정보
    gr.Markdown(
        """
        ---
        ### Compression Level Guide
        
        **ResNet (ResNet18/50/101):**
        - Level 1: 10% pruning
        - Level 2: 25% pruning
        - Level 3: 40% pruning + quantization
        - Level 4: 55% pruning + quantization
        - Level 5: 70% pruning + quantization
        
        **Vision Transformer (ViT):**
        - Level 1: 5% pruning
        - Level 2: 10% pruning
        - Level 3: 15% pruning
        - Level 4: 25% pruning + quantization
        - Level 5: 35% pruning + quantization
        
        **Other Models (MobileNet, EfficientNet, etc.):**
        - Default: 10% pruning per level (quantization from level 3)
        """
    )

# Launch interface
if __name__ == "__main__":
    demo.launch(share=True)