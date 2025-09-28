import draccus
from dataclasses import dataclass


@dataclass
class DetectorConfig(draccus.ChoiceRegistry):
    pass


@DetectorConfig.register_subclass("base_detector")
@dataclass
class BaseDetectorConfig(draccus.ChoiceRegistry):
    visualize: bool = False
    visualize_first: int = 0


@DetectorConfig.register_subclass("grounding_dino")
@dataclass
class GroundingDinoDetectorConfig(BaseDetectorConfig):
    model_config_path: str = "configs/grounding_dino/GroundingDINO_SwinT_OGC.py"
    model_checkpoint: str = "weights/groundingdino_swint_ogc.pth"
    device: str = "cuda"
    box_threshold: float = 0.3
    text_threshold: float = 0.25