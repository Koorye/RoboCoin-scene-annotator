import torch
from abc import ABC, abstractmethod
from PIL import Image

from .configuration_detectors import (
    BaseDetectorConfig, 
    DetectorConfig,
    GroundingDinoDetectorConfig,
)
from .detection_result import DetectionResult


class BaseDetector(ABC):

    config_class = BaseDetectorConfig
    name = "base_detector"

    def __init__(
        self,
        config: BaseDetectorConfig,
    ):
        self.config = config
        self.model = self._load_model()
        self._count = 0

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def _detect(self, image, prompt) -> DetectionResult:
        pass

    def detect(self, image, prompt) -> DetectionResult:
        result = self._detect(image, prompt)
        if self._count < self.config.visualize_first:
            result.visualize()
        self._count += 1
        return result


class GroundingDinoDetector(BaseDetector):

    config_class = GroundingDinoDetectorConfig
    name = "grounding_dino"

    def __init__(
        self,
        config: GroundingDinoDetectorConfig,
    ):
        super().__init__(config=config)
        self.config = config
    
    def _load_model(self):
        from groundingdino.util.inference import load_model

        return load_model(
            model_config_path=self.config.model_config_path,
            model_checkpoint_path=self.config.model_checkpoint,
            device=self.config.device
        )

    @torch.no_grad()
    def _detect(self, image, prompt):
        from groundingdino.util.inference import predict

        image, image_for_detector = self._load_image(image)
        boxes, logits, phrases = predict(
            model=self.model,
            image=image_for_detector,
            caption=prompt,
            box_threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
            device=self.config.device,
        )
        return DetectionResult(
            image=image,
            names=phrases,
            boxes=boxes.cpu().numpy().tolist(),
            logits=logits.cpu().numpy().tolist(),
        )
    
    def _load_image(self, image):
        import groundingdino.datasets.transforms as T
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_pillow = Image.fromarray(image).convert('RGB')
        image_transformed, _ = transform(image_pillow, None)
        return image, image_transformed


def get_detector(config: DetectorConfig) -> BaseDetector:
    if isinstance(config, GroundingDinoDetectorConfig):
        return GroundingDinoDetector(config)
    else:
        raise ValueError(f"Unknown detector type: {config.type}")