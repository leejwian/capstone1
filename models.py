"""
모델 인터페이스 및 로더 - 어떤 이미지 모델도 지원 가능한 추상화 계층
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Union
import pathlib

import torch
from torch import nn
from torchvision import models


class ImageModelLoader(ABC):
    """어떤 이미지 모델이든 로드할 수 있는 추상 인터페이스"""
    
    @abstractmethod
    def load_model(
        self,
        name: str,
        pretrained: bool = True,
        num_classes: Optional[int] = None,
        checkpoint: Optional[str] = None,
    ) -> nn.Module:
        """모델을 로드하여 반환"""
        pass
    
    @abstractmethod
    def get_available_models(self) -> Dict[str, Tuple]:
        """사용 가능한 모델 목록 반환"""
        pass


class TorchvisionModelLoader(ImageModelLoader):
    """Torchvision 모델 로더"""
    
    def __init__(self, custom_registry: Optional[Dict[str, Tuple]] = None):
        """
        Args:
            custom_registry: 기본 모델에 추가할 커스텀 모델 딕셔너리
                            {model_name: (builder_fn, weights_enum)}
        """
        self._registry = self._default_registry()
        if custom_registry:
            self._registry.update(custom_registry)
    
    def _default_registry(self) -> Dict[str, Tuple]:
        """기본 모델 레지스트리"""
        return {
            "resnet18": (models.resnet18, getattr(models, "ResNet18_Weights", None)),
            "resnet50": (models.resnet50, getattr(models, "ResNet50_Weights", None)),
            "resnet101": (models.resnet101, getattr(models, "ResNet101_Weights", None)),
            "vit_b_16": (models.vit_b_16, getattr(models, "ViT_B_16_Weights", None)),
            "vit_b_32": (models.vit_b_32, getattr(models, "ViT_B_32_Weights", None)),
            "mobilenet_v2": (models.mobilenet_v2, getattr(models, "MobileNet_V2_Weights", None)),
            "efficientnet_b0": (models.efficientnet_b0, getattr(models, "EfficientNet_B0_Weights", None)),
        }
    
    def get_available_models(self) -> Dict[str, Tuple]:
        """사용 가능한 모델 목록 반환"""
        return self._registry.copy()
    
    def add_model(self, name: str, builder_fn, weights_enum):
        """새로운 모델을 레지스트리에 추가"""
        self._registry[name] = (builder_fn, weights_enum)
    
    def load_model(
        self,
        name: str,
        pretrained: bool = True,
        num_classes: Optional[int] = None,
        checkpoint: Optional[str] = None,
    ) -> nn.Module:
        """모델을 로드"""
        if name not in self._registry:
            raise ValueError(
                f"Unsupported model '{name}'. "
                f"Available: {', '.join(self._registry.keys())}"
            )
        
        builder, weights_enum = self._registry[name]
        weights = weights_enum.DEFAULT if pretrained and weights_enum else None
        model = builder(weights=weights)
        
        # 분류기 헤드 수정 (ResNet과 ViT 모두 지원)
        if num_classes:
            model = self._replace_classifier(model, num_classes)
        
        model.eval()
        
        # 체크포인트 로드
        if checkpoint:
            self._load_checkpoint(model, checkpoint)
        
        return model
    
    @staticmethod
    def _replace_classifier(model: nn.Module, num_classes: int) -> nn.Module:
        """모델의 분류기 헤드를 교체"""
        # ResNet 계열
        if hasattr(model, "fc"):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        # Vision Transformer 계열
        elif hasattr(model, "heads"):
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)
        # EfficientNet 계열
        elif hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Sequential):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes)
            else:
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
        # MobileNet 계열
        elif hasattr(model, "classifier"):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        
        return model
    
    @staticmethod
    def _load_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
        """체크포인트를 모델에 로드"""
        checkpoint_file = pathlib.Path(checkpoint_path)
        if checkpoint_file.exists():
            state = torch.load(checkpoint_file, map_location="cpu")
            model.load_state_dict(state)
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")


class CustomModelLoader(ImageModelLoader):
    """커스텀 모델 파일(.pt, .pth) 로더"""
    
    def __init__(self, model_class_factory=None):
        """
        Args:
            model_class_factory: 커스텀 모델 클래스를 반환하는 callable
        """
        self.model_class_factory = model_class_factory
    
    def get_available_models(self) -> Dict[str, Tuple]:
        """커스텀 모델은 파일 경로로 로드되므로 비어있음"""
        return {}
    
    def load_model(
        self,
        path: str,
        pretrained: bool = True,
        num_classes: Optional[int] = None,
        checkpoint: Optional[str] = None,
    ) -> nn.Module:
        """파일에서 커스텀 모델 로드"""
        model_path = pathlib.Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # 모델 클래스가 제공되면 먼저 인스턴스 생성
        if self.model_class_factory:
            model = self.model_class_factory()
        else:
            # 또는 저장된 상태 딕셔너리만으로 로드 시도
            model = None
        
        # 상태 딕셔너리 로드
        state = torch.load(model_path, map_location="cpu")
        
        if isinstance(state, dict) and "state_dict" in state:
            # 전체 체크포인트 형식 (state_dict + 추가 정보)
            actual_state = state["state_dict"]
        else:
            actual_state = state
        
        if model:
            model.load_state_dict(actual_state)
        else:
            raise ValueError(
                "Model class factory not provided. "
                "Cannot load custom model architecture."
            )
        
        model.eval()
        return model


# 기본 TorchvisionModelLoader 인스턴스
default_model_loader = TorchvisionModelLoader()


def get_model_loader(
    loader_type: str = "torchvision",
    custom_registry: Optional[Dict] = None,
) -> ImageModelLoader:
    """
    모델 로더 팩토리
    
    Args:
        loader_type: "torchvision" 또는 "custom"
        custom_registry: Torchvision 로더에 추가할 커스텀 모델
    
    Returns:
        ImageModelLoader 인스턴스
    """
    if loader_type == "torchvision":
        return TorchvisionModelLoader(custom_registry)
    elif loader_type == "custom":
        return CustomModelLoader()
    else:
        raise ValueError(f"Unknown loader type: {loader_type}")
