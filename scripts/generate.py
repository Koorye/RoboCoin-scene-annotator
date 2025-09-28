"""
e.g.
python scripts/generate.py \
    --language_model.type="ollama" \
    --language_model.model="deepseek-r1:8b" \
    --language_model.think=False \
    --prompt="You are a professional annotator, please provide a concise and clear summary in one sentence, describing the position and relationship of each object. Do not add any extra information! Here are the objects:\n" \
    --json_dir="results/annotations/" \
    --save_dir="results/annotations_refined/"
    
"""

import draccus
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.language_models import (
    LanguageModelConfig,
    OllamaLanguageModelConfig,
    get_language_model,
)


@dataclass
class GenerationConfig:
    language_model: LanguageModelConfig
    prompt: str
    json_path: Optional[str] = None
    json_dir: Optional[str] = None
    save_dir: str = "annotations_refined/"

    def __post_init__(self):
        if (self.json_path is None) == (self.json_dir is None):
            raise ValueError("Either json_path or json_dir must be provided, but not both.")


def parse_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    data_string = ''
    for obj in data['object']:
        data_string += obj['name']
        for k, v in obj['info'].items():
            data_string += f', {k}: {v}'
        data_string += '\n'
    return data, data_string


@draccus.wrap()
def main(config: GenerationConfig):
    language_model = get_language_model(config.language_model)

    os.makedirs(config.save_dir, exist_ok=True)

    if config.json_path:
        data, prompt = parse_json(config.json_path)
        prompt = config.prompt + prompt
        response = language_model.generate(prompt)
        data['description'] = response
        save_path = os.path.join(config.save_dir, os.path.basename(config.json_path))
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)

    elif config.json_dir:
        json_files = [f for f in os.listdir(config.json_dir) if f.lower().endswith('.json')]
        json_files.sort()
        for json_file in json_files:
            json_path = os.path.join(config.json_dir, json_file)
            data, prompt = parse_json(json_path)
            prompt = config.prompt + prompt
            response = language_model.generate(prompt)
            data['description'] = response
            save_path = os.path.join(config.save_dir, json_file)
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)


if __name__ == '__main__':
    main()