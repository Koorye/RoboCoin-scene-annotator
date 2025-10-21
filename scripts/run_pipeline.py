"""
Main pipeline to run the full annotation process: 
1. Prompt extraction: extract prompts from task instructions.
2. Frame extraction: extract first frames from videos.
3. Object detection: detect objects in frames using prompts.
4. Scene generation: generate refined scene annotations from detected objects.

e.g.
python scripts/run_pipeline.py \
    --repo_root="/home/koorye/.cache/huggingface/lerobot/" \
    --repo_id="realman_fold_towel" \
    --save_root="results/" \
    --camera="observation.images.cam_high" \
    --detector.type="grounding_dino" \
    --detector.device=cpu \
    --detector.visualize_first=5 \
    --language_model.type="ollama" \
    --language_model.model="deepseek-r1:8b" \
    --language_model.think=False
"""

import draccus
import ollama
import os
import sys
from dataclasses import dataclass
from rich import print

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.detectors import DetectorConfig
from core.language_models import LanguageModelConfig


from detect import main as detect_main
from detect import InferenceConfig

from extract_first_frame import main as extract_main
from extract_first_frame import ExtractConfig

from extract_prompt import main as extract_prompt_main
from extract_prompt import GenerationConfig as PromptGenerationConfig

from generate import main as generate_main
from generate import GenerationConfig


@dataclass
class PipelineConfig:
    repo_id: str = ""
    repo_root: str = ""
    save_root: str = ""
    camera: str = "observation.images.cam_high_rgb"

    detector: DetectorConfig = None
    language_model: LanguageModelConfig = None

    def __post_init__(self):
        self.extract = ExtractConfig(
            repo_dir=os.path.join(self.repo_root, self.repo_id),
            camera=self.camera,
            save_dir=os.path.join(self.save_root, "frames")
        )
        self.prompt = PromptGenerationConfig(
            language_model=self.language_model,
            prompt="You are a text annotation expert. Please help me find the names of all the objects in the instruction (like bowl, plate, block, banana, etc.), including the adjectives describing the objects (like blue bowl, yellow banana) if it exists in the instruction. All names are enclosed in double quotation marks " ". Do not include any non-existent objects or additional information. A detailed list of instructions is as follows:\n",
            repo_dir=os.path.join(self.repo_root, self.repo_id),
            save_dir=os.path.join(self.save_root, "prompts")
        )
        self.inference = InferenceConfig(
            detector=self.detector,
            repo_id=self.repo_id,
            prompt_dir=os.path.join(self.save_root, "prompts"),
            image_dir=os.path.join(self.save_root, "frames"),
            save_dir=os.path.join(self.save_root, "annotations")
        )
        self.generation = GenerationConfig(
            language_model=self.language_model,
            prompt="You are a professional annotator, please provide a concise and clear summary in one sentence, describing the position and relationship of each object. Do not add any extra information! Here are the objects:\n",
            repo_id=self.repo_id,
            json_dir=os.path.join(self.save_root, "annotations"),
            save_dir=os.path.join(self.save_root, "annotations_refined")
        )


@draccus.wrap()
def main(config: PipelineConfig):
    print('=' * 40)
    print("Starting pipeline...")
    print('=' * 40)

    if os.path.exists(os.path.join(config.save_root, "prompts", config.repo_id + ".txt")):
        print(f"[WARN] Prompts for {config.repo_id} already exist, skipping extraction.")
        print("[WRAN] If you want to re-extract, please delete the file and run again.")
        print('=' * 40)
    else:
        print("Extracting prompts...")
        print('=' * 40)
        extract_prompt_main(config.prompt)

    if os.path.exists(os.path.join(config.save_root, "frames", config.repo_id)):
        print(f"[WRAN] Frames for {config.repo_id} already exist, skipping extraction.")
        print("[WARN] If you want to re-extract, please delete the folder and run again.")
        print('=' * 40)
    else:
        input("Please review the extracted prompts and press Enter to continue...")
        print("Extracting first frames...")
        print('=' * 40)
        extract_main(config.extract)

    input("Please review the extracted frames and press Enter to continue...")
    print("Generating annotations...")
    print('=' * 40)
    detect_main(config.inference)

    input("Please review the detected annotations and press Enter to continue...")
    print("Refining annotations...")
    print('=' * 40)
    generate_main(config.generation)

    input("Please review the generated annotations and press Enter to continue...")
    print('=' * 40)
    print("Pipeline completed.")
    print('=' * 40)


if __name__ == '__main__':
    main()