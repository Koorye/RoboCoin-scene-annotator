"""
e.g.
python scripts/detect.py \
    --detector.type="grounding_dino" \
    --detector.device=cpu \
    --detector.visualize_first=5 \
    --repo_id "unitree_g1_food_storage" \
    --prompt_dir="results/prompts" \
    --image_dir="results/frames" \
    --save_dir="results/annotations"
"""

import draccus
import imageio
import os
import sys
from dataclasses import dataclass
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.detectors import (
    DetectorConfig,
    GroundingDinoDetectorConfig,
    get_detector,
)
from utils import (
    ensure_dir,
)


@dataclass
class InferenceConfig:
    detector: DetectorConfig
    repo_id: str = ""
    prompt_dir: str = "prompts/"
    image_dir: str = "frames/"
    save_dir: str = "annotations/"


def load_image(path):
    return imageio.v2.imread(path)


def get_filename(path):
    return os.path.splitext(os.path.basename(path))[0] + ".json"


@draccus.wrap()
def main(config: InferenceConfig):
    detector = get_detector(config.detector)

    with open(os.path.join(config.prompt_dir, config.repo_id + '.txt')) as f:
        prompt = f.read().strip()
    
    image_files = [f for f in os.listdir(os.path.join(config.image_dir, config.repo_id)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()

    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(config.image_dir, config.repo_id, image_file)
        image = load_image(image_path)
        result = detector.detect(image, prompt)
        ensure_dir(os.path.join(config.save_dir, config.repo_id, get_filename(image_file)))
        result.dump_json(os.path.join(config.save_dir, config.repo_id, get_filename(image_file)))


if __name__ == '__main__':
    main()