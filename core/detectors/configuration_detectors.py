import draccus
from dataclasses import dataclass


@dataclass
class DetectorConfig(draccus.ChoiceRegistry):
    pass


@DetectorConfig.register_subclass("base_detector")
@dataclass
class BaseDetectorConfig(draccus.ChoiceRegistry):
    visualize_first: int = 0 # number of first images to visualize during inference


@DetectorConfig.register_subclass("grounding_dino")
@dataclass
class GroundingDinoDetectorConfig(BaseDetectorConfig):
    model_config_path: str = "configs/grounding_dino/GroundingDINO_SwinT_OGC.py" # path to model config
    model_checkpoint: str = "weights/groundingdino_swint_ogc.pth" # path to model weights
    device: str = "cuda" # device to run the model on
    box_threshold: float = 0.3 # box detection threshold
    text_threshold: float = 0.25 # text confidence threshold