# RoboCoin Scene Annotator

[English](README.md) | [中文](README_CN.md)

RoboCoin Scene Annotator, an automated robot scene annotation generation tool based on open-vocabulary object detectors and large language models.

## Tool Overview

RoboCoin Scene Annotator adopts an integrated detection and description pipeline:
```mermaid
graph LR;
A[Video] --Extract--> B[Initial Frame]
B --Grounding DINO--> C[Object Detection Boxes & Labels]
C --Language Model--> D[Scene Description]
```

Core Features:
- Open-vocabulary detection capability, no predefined categories required
- Flexible language model integration, supporting local and API modes
- Automated scene description generation
- Visualized annotation results

## Installation Guide

Prerequisites:
- GPU: At least 12GB VRAM recommended
- Network: Access to HuggingFace for downloading pre-trained models

Installation Steps

1. Download repository
   ```bash
   git clone --recursive https://github.com/Koorye/RoboCoin-scene-annotator.git
   ```

2. Install Grounding DINO

   Refer to [Grounding DINO official repository](Link2):

   Install PyTorch (recommended torch 2.5.1)
   ```bash
   pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url Link3
   ```

   Install Grounding DINO repository
   ```bash
   cd third_party/GroundingDINO
   pip install -e .
   cd ..
   ```

   Download pre-trained weights
   ```bash
   mkdir weights
   cd weights
   wget -q Link2/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
   cd ..
   ```

3. Install Ollama (optional, for running language models locally)

   Refer to [Ollama official repository](Link4)

   Linux standard installation (root ):
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

   Linux manual installation:
   ```bash
   curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
   mkdir -p ~/.local
   tar -C ~/.local -xzf ollama-linux-amd64.tgz
   export PATH="$HOME/.local:$PATH"
   source ~/.bashrc
   ollama serve
   ```

4. Install other dependencies
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run main program:
```bash
python scripts/run_pipeline.py [parameters]
```

Parameter Details

Parameter Category | Parameter Name | Type | Description
--- | --- | --- | ---
Basic Parameters | --repo_id | str | Repository identifier 
| | --repo_root | str | Repository root path
| | --save_root | str | Results save root path
| | --camera | str | Camera key name
Detector Configuration | --detector.type | str | Detector type 
||--detector.visualize_first | int | Number of frames to visualize, detection results of first few frames will be drawn for manual inspection 
||--detector.model_config_path | str | Model configuration file path
||--detector.model_checkpoint | str | Model weights path 
||--detector.device | str | Running device 
||--detector.box_threshold | float | Detection box threshold 
||--detector.text_threshold | float | Text threshold 
Language Model Configuration | --language_model.type | str | Language model type
| | --language_model.think | bool | Whether to use thinking mode 
| | --language_model.api_url | str | API endpoint URL, API mode only
| | --language_model.api_key | str | API key, API mode only
| | --language_model.model | str | Model name 

Usage Example:
```bash
python scripts/run_pipeline.py \
    --repo_id example_repo \
    --repo_root /path/to/repo_root \
    --save_root results/ \
    --camera observation.front \
    --detector.type grounding_dino \
    --detector.model_config_path configs/grounding_dino/GroundingDINO_SwinT_OGC.py \
    --detector.model_checkpoint weights/groundingdino_swint_ogc.pth \
    --detector.device cuda:0 \
    --detector.box_threshold 0.3 \
    --detector.text_threshold 0.3 \
    --language_model.type ollama \
    --language_model.model deepseek-r1:13b \
    --language_model.think False
```

### Detailed Process

This program will initiate the scene annotation process, processing videos from the specified repository and saving the generated annotations to the designated path.

1. Prompt Extraction: Extract object list from task description to generate detection prompts, results are saved in `<save_root>/prompts/<repo_id>.txt`.
   Example: `results/prompts/unitree_g1_food_storage.txt`
   ```
   basket . white bowl . cake . donut . white plate .
   ```

2. **Initial Frame Extraction**: Extracts initial frames from videos and visualizes detection results for manual verification. Results are saved in `<save_root>/frames/<repo_id>/` directory.
   Example output:
   ![](examples/first_frames.png)

3. **Object Detection**: Performs object detection using Grounding DINO. Results are saved in `<save_root>/annotations/<repo_id>/` directory.
   Example: `results/annotations/unitree_g1_food_storage/episode_000000.json`
   ```json
      {
     "object": [
       {
         "name": "basket",
         "box": {
           "x_center": 0.5401855111122131,
           "y_center": 0.43615126609802246,
           "width": 0.47688308358192444,
           "height": 0.3583659827709198
         },
         "logit": 0.7744243144989014,
         "info": {
           "position": "center"
         }
       },
       {
         "name": "white bowl white plate",
         "box": {
           "x_center": 0.21382831037044525,
           "y_center": 0.7187483310699463,
           "width": 0.38848257064819336,
           "height": 0.41331198811531067
         },
         "logit": 0.41868603229522705,
         "info": {
           "position": "back left"
         }
       },
       {
         "name": "donut",
         "box": {
           "x_center": 0.19377967715263367,
           "y_center": 0.7328643202781677,
           "width": 0.2978207468986511,
           "height": 0.3233490288257599
         },
         "logit": 0.38621386885643005,
         "info": {
           "position": "back left"
         }
       },
       {
         "name": "white plate",
         "box": {
           "x_center": 0.8474635481834412,
           "y_center": 0.8461897373199463,
           "width": 0.3014012575149536,
           "height": 0.30122512578964233
         },
         "logit": 0.30470189452171326,
         "info": {
           "position": "back right"
         }
       },
       {
         "name": "donut",
         "box": {
           "x_center": 0.2593638002872467,
           "y_center": 0.6321070790290833,
           "width": 0.16489802300930023,
           "height": 0.12321418523788452
         },
         "logit": 0.33250319957733154,
         "info": {
           "position": "back left"
         }
       }
     ]
   }
   ```

4. **Scene Description Generation**: Generates scene descriptions based on detection results. Results are saved in `<save_root>/annotations_refined/<repo_id>/` directory.
Example: `results/annotations_refined/unitree_g1_food_storage/episode_000000.json`
   ```json
   {
     "object": [
       {
         "name": "basket",
         "box": {
           "x_center": 0.5401855111122131,
           "y_center": 0.43615126609802246,
           "width": 0.47688308358192444,
           "height": 0.3583659827709198
         },
         "logit": 0.7744243144989014,
         "info": {
           "position": "center"
         }
       },
       {
         "name": "white bowl white plate",
         "box": {
           "x_center": 0.21382831037044525,
           "y_center": 0.7187483310699463,
           "width": 0.38848257064819336,
           "height": 0.41331198811531067
         },
         "logit": 0.41868603229522705,
         "info": {
           "position": "back left"
         }
       },
       {
         "name": "donut",
         "box": {
           "x_center": 0.19377967715263367,
           "y_center": 0.7328643202781677,
           "width": 0.2978207468986511,
           "height": 0.3233490288257599
         },
         "logit": 0.38621386885643005,
         "info": {
           "position": "back left"
         }
       },
       {
         "name": "white plate",
         "box": {
           "x_center": 0.8474635481834412,
           "y_center": 0.8461897373199463,
           "width": 0.3014012575149536,
           "height": 0.30122512578964233
         },
         "logit": 0.30470189452171326,
         "info": {
           "position": "back right"
         }
       },
       {
         "name": "donut",
         "box": {
           "x_center": 0.2593638002872467,
           "y_center": 0.6321070790290833,
           "width": 0.16489802300930023,
           "height": 0.12321418523788452
         },
         "logit": 0.33250319957733154,
         "info": {
           "position": "back left"
         }
       }
     ],
     "description": "The basket is at the center. The white bowl and white plate (back left) contain the donut located there."
   }
   ```

## Acknowledgments

Thanks to the support of the following excellent projects:
- https://github.com/IDEA-Research/GroundingDINO: Advanced open-vocabulary object detector
- https://github.com/ollama/ollama: Local large language model deployment framework
- Other open-source projects contributing to the computer vision and artificial intelligence fields