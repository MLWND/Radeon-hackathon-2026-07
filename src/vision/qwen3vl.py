"""
Module 2: Qwen3-VL Integration
Single perception backbone: detect + ground + reason.
No YOLO needed — Qwen3-VL handles everything.
"""
import numpy as np
import json
import time
from typing import Dict, Optional, List


# Prompt for Qwen3-VL grounding + task planning
GROUNDING_PROMPT = """Analyze this image for a robotic pick-and-place task.

User instruction: {instruction}

You must:
1. Identify the object to pick and the target location
2. Provide bounding boxes in [x1, y1, x2, y2] format (normalized 0-1)
3. Estimate 3D positions relative to the robot workspace

Output ONLY valid JSON:
{{
    "task": "pick_place",
    "reasoning": "brief explanation of what you see and plan",
    "object": {{
        "type": "object type",
        "color": "color",
        "bbox": [x1, y1, x2, y2],
        "center_pixel": [cx, cy],
        "estimated_xyz": [x, y, z],
        "confidence": 0.95
    }},
    "target": {{
        "type": "object type",
        "color": "color",
        "bbox": [x1, y1, x2, y2],
        "center_pixel": [cx, cy],
        "estimated_xyz": [x, y, z],
        "confidence": 0.90
    }}
}}"""


class Qwen3VLWrapper:
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-2B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.last_inference_time = 0.0
        self.last_output = None

    def load(self):
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            import torch
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            print(f"Qwen3-VL loaded: {self.model_name}")
        except Exception as e:
            print(f"Qwen3-VL load failed: {e}")
            print("Using rule-based fallback for task parsing")
        return self

    def understand(self, image: np.ndarray, instruction: str) -> Dict:
        start = time.time()

        if self.model is not None:
            result = self._inference(image, instruction)
        else:
            result = self._rule_based(instruction)

        self.last_inference_time = (time.time() - start) * 1000
        self.last_output = result
        return result

    def _inference(self, image: np.ndarray, instruction: str) -> Dict:
        from PIL import Image
        import torch

        pil_img = Image.fromarray(image)
        prompt = GROUNDING_PROMPT.format(instruction=instruction)

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": prompt},
            ]}
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[pil_img], return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        response = self.processor.decode(output[0], skip_special_tokens=True)
        return self._parse_response(response, image.shape)

    def _parse_response(self, response: str, image_shape: tuple) -> Dict:
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(response[start:end])
                return self._validate_and_fill(result, image_shape)
        except json.JSONDecodeError:
            pass
        return self._rule_based(response)

    def _validate_and_fill(self, result: Dict, image_shape: tuple) -> Dict:
        h, w = image_shape[:2]
        for key in ["object", "target"]:
            if key in result:
                item = result[key]
                if "bbox" in item:
                    bbox = item["bbox"]
                    item["center_pixel"] = [
                        int((bbox[0] + bbox[2]) / 2 * w),
                        int((bbox[1] + bbox[3]) / 2 * h),
                    ]
                if "estimated_xyz" not in item:
                    item["estimated_xyz"] = self._pixel_to_xyz(
                        item.get("center_pixel", [w//2, h//2])
                    )
                if "confidence" not in item:
                    item["confidence"] = 0.85
        return result

    def _pixel_to_xyz(self, pixel: List[float]) -> List[float]:
        x = (pixel[0] / 640 - 0.5) * 0.8
        y = (pixel[1] / 480 - 0.5) * 0.6
        z = 0.05
        return [x, y, z]

    def _rule_based(self, instruction: str) -> Dict:
        instruction_lower = instruction.lower()
        colors = ["red", "blue", "green", "yellow", "white", "black"]
        objects = ["cup", "box", "cube", "bottle", "can"]

        found_color = None
        found_obj = None
        for c in colors:
            if c in instruction_lower:
                found_color = c
                break
        for o in objects:
            if o in instruction_lower:
                found_obj = o
                break

        return {
            "task": "pick_place",
            "reasoning": f"Rule-based fallback: detected '{found_color or 'unknown'} {found_obj or 'object'}'",
            "object": {
                "type": found_obj or "cube",
                "color": found_color or "unknown",
                "bbox": [0.2, 0.3, 0.4, 0.6],
                "center_pixel": [192, 230],
                "estimated_xyz": [0.4, 0.0, 0.05],
                "confidence": 0.80,
            },
            "target": {
                "type": "box",
                "color": "blue",
                "bbox": [0.6, 0.3, 0.8, 0.6],
                "center_pixel": [448, 230],
                "estimated_xyz": [-0.3, 0.0, 0.05],
                "confidence": 0.80,
            },
        }

    def get_inference_time(self) -> float:
        return self.last_inference_time

    def get_last_output(self) -> Optional[Dict]:
        return self.last_output
