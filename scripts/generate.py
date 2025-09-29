"""
e.g.
python scripts/generate.py \
    --language_model.type="ollama" \
    --language_model.model="deepseek-r1:8b" \
    --language_model.think=False \
    --prompt="You are a professional annotator, please provide a concise and clear summary in one sentence, describing the position and relationship of each object. Do not add any extra information! Here are the objects:\n" \
    --repo_id "unitree_g1_food_storage" \
    --json_dir="results/annotations/" \
    --save_dir="results/annotations_refined/"
    
python scripts/generate.py \
    --language_model.type="webapi" \
    --language_model.model="free:Qwen3-30B-A3B" \
    --language_model.api_url="https://api.suanli.cn/v1/chat/completions" \
    --language_model.api_key="sk-vPTMzbUSQzG6jJiE0QQNENu2f7zwcD8m6FGr8GDTD4Vqy1Dw" \
    --prompt="You are a professional annotator, please provide a concise and clear summary in one sentence, describing the position and relationship of each object. Do not add any extra information! Here are the objects:\n" \
    --repo_id "unitree_g1_food_storage" \
    --json_dir="results/annotations/" \
    --save_dir="results/annotations_refined/"
"""

import draccus
import json
import os
import sys
from dataclasses import dataclass
from tqdm import tqdm


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.language_models import (
    LanguageModelConfig,
    OllamaLanguageModelConfig,
    get_language_model,
)
from utils import(
    ensure_dir
)


@dataclass
class GenerationConfig:
    language_model: LanguageModelConfig
    prompt: str
    repo_id: str = ""
    json_dir: str = "annotations/"
    save_dir: str = "annotations_refined/"


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

    json_files = [f for f in os.listdir(os.path.join(config.json_dir, config.repo_id))
                  if f.lower().endswith('.json')]
    json_files.sort()
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        json_path = os.path.join(config.json_dir, config.repo_id, json_file)
        data, prompt = parse_json(json_path)
        prompt = config.prompt + prompt
        response = language_model.generate(prompt)
        data['description'] = response
        save_path = os.path.join(config.save_dir, config.repo_id, json_file)
        ensure_dir(save_path)
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)


if __name__ == '__main__':
    main()