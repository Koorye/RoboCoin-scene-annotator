"""
e.g.
python scripts/extract_prompt.py \
    --language_model.type="ollama" \
    --language_model.model="deepseek-r1:8b" \
    --language_model.think=False \
    --prompt="You are a text annotation expert. Please help me find the names of all the objects in the instruction (like bowl, plate, block, banana, etc.), including the adjectives describing the objects (like blue bowl, yellow banana) if it exists in the instruction. Separate the objects with a period "." and do not include any non-existent objects or additional information. A detailed list of instructions is as follows:\n" \
    --repo_dir="/home/koorye/.cache/huggingface/lerobot/unitree_g1_food_storage" \
    --save_dir="results/prompts/"
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
    OllamaLanguageModel,
    get_language_model,
)
from utils import (
    ensure_dir,
    get_filename_without_suffix,
    get_lerobot_root,
)


@dataclass
class GenerationConfig:
    language_model: LanguageModelConfig
    prompt: str
    repo_dir: Optional[str] = None
    save_dir: str = "prompts/"


def parse_jsonl(jsonl_path):
    with open(jsonl_path) as f:
        lines = f.readlines()

    out = ''
    for line in lines:
        data = json.loads(line)
        out += f'- {data["task"]}\n'
    return out.strip()


def post_process_response(response):
    # response = response.replace(',', '.')
    # objects = list(sorted(set([obj.strip() for obj in response.split('.')])))
    # objects = filter(lambda x: len(x) > 0, objects)
    # return ' . '.join(objects) + ' .'
    # find all text within double quotation marks
    import re
    pattern = r'"(.*?)"'
    objects = re.findall(pattern, response)
    objects = list(sorted(set([obj.strip() for obj in objects])))
    return ' . '.join(objects) + ' .'


@draccus.wrap()
def main(config: GenerationConfig):
    language_model = get_language_model(config.language_model)
    
    task_path = os.path.join(config.repo_dir, 'meta/tasks.jsonl')
    save_path = os.path.join(config.save_dir, get_filename_without_suffix(config.repo_dir) + '.txt')

    prompt = parse_jsonl(task_path)
    prompt = config.prompt + prompt
    response = language_model.generate(prompt)
    response = post_process_response(response)

    ensure_dir(save_path)
    with open(save_path, 'w') as f:
        f.write(response)
    print('Prompt:', response)


if __name__ == '__main__':
    main()