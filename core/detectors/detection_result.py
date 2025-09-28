import json
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
import torch
from dataclasses import dataclass
from torchvision.ops import box_convert
from typing import List


@dataclass
class DetectionResult:
    image: np.ndarray
    names: List[str]
    boxes: List[List[float]]
    logits: List[float]

    def __post_init__(self):
        infos = []
        for box, logit, name in zip(self.boxes, self.logits, self.names):
            infos.append({
                "position": self._get_position(box),
            })
        self.infos = infos
    
    def dump_json(self, save_path):
        annotations = []
        for name, box, logit, info in zip(self.names, self.boxes, self.logits, self.infos):
            annotations.append({
                "name": name,
                "box": {
                    "x_center": box[0],
                    "y_center": box[1],
                    "width": box[2],
                    "height": box[3],
                },
                "logit": logit,
                "info": info,
            })

        with open(save_path, 'w') as f:
            json.dump({"object": annotations}, f, indent=2)

    def visualize(self):
        h, w, _ = self.image.shape
        boxes = torch.from_numpy(np.array(self.boxes)) * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        detections = sv.Detections(xyxy=xyxy)

        labels = []
        for name, logit, info in zip(self.names, self.logits, self.infos):
            label = f'{name} {logit:.2f}'
            for key, value in info.items():
                label += f' {value}'
            labels.append(label)

        bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
        label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX, text_scale=1, text_thickness=2)
        annotated_frame = self.image.copy()
        annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        plt.figure(figsize=(5, 5))
        plt.imshow(annotated_frame)
        plt.show()
    
    def _get_position(self, box):
        x_center = box[0]
        y_center = box[1]

        horizontal_pos = "left" if x_center <  0.4 else "right" if x_center > 0.6 else ""
        vertical_pos = "front" if y_center < 0.4 else "back" if y_center > 0.6 else ""

        position = f"{vertical_pos} {horizontal_pos}".strip()
        if position == "":
            position = "center"
        return position