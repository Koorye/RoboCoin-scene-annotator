"""
e.g.
python scripts/detect.py \
    --detector.type="grounding_dino" \
    --detector.device=cpu \
    --detector.visualize=True \
    --prompt="pink bowl. blue bowl. iron rack. white cup." \
    --image_dir="results/frames/" \
    --save_dir="results/annotations/"
"""

import draccus
import imageio
import os
import sys
from dataclasses import dataclass
from tqdm import tqdm
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.detectors import (
    DetectorConfig,
    GroundingDinoDetectorConfig,
    get_detector,
)


@dataclass
class InferenceConfig:
    detector: DetectorConfig
    prompt: str
    image_path: Optional[str] = None
    image_dir: Optional[str] = None
    save_dir: str = "annotations/"

    def __post_init__(self):
        if (self.image_path is None) == (self.image_dir is None):
            raise ValueError("Either image_path or image_dir must be provided, but not both.")


def load_image(path):
    return imageio.v2.imread(path)


def get_filename(path):
    return os.path.splitext(os.path.basename(path))[0] + ".json"


@draccus.wrap()
def main(config: InferenceConfig):
    detector = get_detector(config.detector)

    os.makedirs(config.save_dir, exist_ok=True)

    if config.image_path:
        image = load_image(config.image_path)
        result = detector.detect(image, config.prompt)
        result.dump_json(os.path.join(config.save_dir, get_filename(config.image_path)))

    elif config.image_dir:
        image_files = [f for f in os.listdir(config.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()
        for image_file in tqdm(image_files, desc="Processing images"):
            image_path = os.path.join(config.image_dir, image_file)
            image = load_image(image_path)
            result = detector.detect(image, config.prompt)
            result.dump_json(os.path.join(config.save_dir, get_filename(image_file)))


if __name__ == '__main__':
    main()