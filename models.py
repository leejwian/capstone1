"""
모델 인터페이스 및 로더 모듈.

Torchvision 및 커스텀 이미지 분류 모델을
일관된 인터페이스로 로드하기 위한 추상화 계층을 제공한다.
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Callable
import pathlib

import torch
from torch import nn
from torchvision import models


class ImageModelLoader(ABC):
    """
    이미지 분류 모델 로딩을 위한 추상 인터페이스.
    """

    @abstractmethod
    def load_model(
        self,
        name: str,
        pretrained: bool = True,
        num_classes: Optional[int] = None,
        checkpoint: Optional[str] = None,
    ) -> nn.Module:
        """
        모델을 로드하여 반환한다.

        Args:
            name: 모델 이름 또는 경로
            pretrained: 사전 학습 가중치 사용 여부
            num_classes: 분류 클래스 수
            checkpoint: 로드할 체크포인트 경로

        Returns:
            로드된 PyTorch 모델
        """
        raise NotImplementedError

    @abstractmethod
    def get_available_models(self) -> Dict[str, Tuple]:
        """
        사용 가능한 모델 목록을 반환한다.

        Returns:
            모델 레지스트리 딕셔너리
        """
        raise NotImplementedError


class TorchvisionModelLoader(ImageModelLoader):
    """
    Torchvision 기반 이미지 모델 로더.
    """

    def __init__(
        self,
        custom_registry: Optional[Dict[str, Tuple[Callable, Optional[object]]]] = None,
    ) -> None:
        """
        Args:
            custom_registry:
                기본 레지스트리에 추가할 커스텀 모델 정보
                {model_name: (builder_fn, weights_enum)}
        """
        self._registry: Dict[str, Tuple] = self._build_default_registry()

        if custom_registry:
            self._registry.update(custom_registry)

    @staticmethod
    def _build_default_registry() -> Dict[str, Tuple]:
        """
        Torchvision 기본 모델 레지스트리를 생성한다.

        Returns:
            모델 레지스트리 딕셔너리
        """
        return {
            "resnet18": (
                models.resnet18,
                getattr(models, "ResNet18_Weights", None),
            ),
            "resnet50": (
                models.resnet50,
                getattr(models, "ResNet50_Weights", None),
            ),
            "resnet101": (
                models.resnet101,
                getattr(models, "ResNet101_Weights", None),
            ),
            "vit_b_16": (
                models.vit_b_16,
                getattr(models, "ViT_B_16_Weights", None),
            ),
            "vit_b_32": (
                models.vit_b_32,
                getattr(models, "ViT_B_32_Weights", None),
            ),
            "mobilenet_v2": (
                models.mobilenet_v2,
                getattr(models, "MobileNet_V2_Weights", None),
            ),
            "efficientnet_b0": (
                models.efficientnet_b0,
                getattr(models, "EfficientNet_B0_Weights", None),
            ),
        }

    def get_available_models(self) -> Dict[str, Tuple]:
        """
        사용 가능한 Torchvision 모델 목록을 반환한다.
        """
        return self._registry.copy()

    def add_model(
        self,
        name: str,
        builder_fn: Callable,
        weights_enum: Optional[object],
    ) -> None:
        """
        레지스트리에 새로운 모델을 추가한다.

        Args:
            name: 모델 이름
            builder_fn: 모델 생성 함수
            weights_enum: 사전 학습 가중치 enum
        """
        self._registry[name] = (builder_fn, weights_enum)

    def load_model(
        self,
        name: str,
        pretrained: bool = True,
        num_classes: Optional[int] = None,
        checkpoint: Optional[str] = None,
    ) -> nn.Module:
        """
        Torchvision 모델을 로드한다.
        """
        if name not in self._registry:
            raise ValueError(
                f"Unsupported model '{name}'. "
                f"Available models: {', '.join(self._registry.keys())}"
            )

        builder_fn, weights_enum = self._registry[name]
        weights = weights_enum.DEFAULT if pretrained and weights_enum else None
        model = builder_fn(weights=weights)

        if num_classes is not None:
            self._replace_classifier(model, num_classes)

        if checkpoint:
            self._load_checkpoint(model, checkpoint)

        model.eval()
        return model

    @staticmethod
    def _replace_classifier(model: nn.Module, num_classes: int) -> None:
        """
        모델의 분류기 헤드를 num_classes에 맞게 교체한다.
        """
        if hasattr(model, "fc"):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            return

        if hasattr(model, "heads"):
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)
            return

        if hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Sequential):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes)
            else:
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)

    @staticmethod
    def _load_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
        """
        체크포인트 파일을 모델에 로드한다.
        """
        checkpoint_file = pathlib.Path(checkpoint_path)

        if not checkpoint_file.exists():
            print(f"Warning: checkpoint not found at {checkpoint_path}")
            return

        state_dict = torch.load(checkpoint_file, map_location="cpu")
        model.load_state_dict(state_dict)


class CustomModelLoader(ImageModelLoader):
    """
    사용자 정의 모델 파일(.pt, .pth) 로더.
    """

    def __init__(
        self,
        model_class_factory: Optional[Callable[[], nn.Module]] = None,
    ) -> None:
        """
        Args:
            model_class_factory:
                커스텀 모델 인스턴스를 생성하는 callable
        """
        self.model_class_factory = model_class_factory

    def get_available_models(self) -> Dict[str, Tuple]:
        """
        커스텀 모델은 사전 정의된 목록이 없으므로 빈 딕셔너리를 반환한다.
        """
        return {}

    def load_model(
        self,
        path: str,
        pretrained: bool = True,
        num_classes: Optional[int] = None,
        checkpoint: Optional[str] = None,
    ) -> nn.Module:
        """
        파일 경로로부터 커스텀 모델을 로드한다.
        """
        model_path = pathlib.Path(path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if not self.model_class_factory:
            raise ValueError(
                "Model class factory not provided. "
                "Cannot load custom model architecture."
            )

        model = self.model_class_factory()
        state = torch.load(model_path, map_location="cpu")

        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        model.load_state_dict(state)
        model.eval()
        return model


# 기본 Torchvision 모델 로더 인스턴스
default_model_loader = TorchvisionModelLoader()


def get_model_loader(
    loader_type: str = "torchvision",
    custom_registry: Optional[Dict[str, Tuple]] = None,
) -> ImageModelLoader:
    """
    모델 로더 팩토리 함수.

    Args:
        loader_type: "torchvision" 또는 "custom"
        custom_registry: Torchvision 로더에 추가할 커스텀 모델

    Returns:
        ImageModelLoader 인스턴스
    """
    if loader_type == "torchvision":
        return TorchvisionModelLoader(custom_registry)

    if loader_type == "custom":
        return CustomModelLoader()

    raise ValueError(f"Unknown loader type: {loader_type}")
