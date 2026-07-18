"""
Qwen3-VL Perception Module
Uses native Qwen3VLForConditionalGeneration from transformers>=4.57.
"""
import numpy as np
import json
import time
from typing import Dict, Optional, List


GROUNDING_PROMPT = """Analyze this image for a robotic pick-and-place task.

Instruction: {instruction}

You must identify:
1. The object to pick (name from: {objects})
2. Where to place it (coordinates on the table, x in [0.3,0.8], y in [-0.3,0.3])

Output ONLY valid JSON:
{{
    "pick": "object_name",
    "place_xyz": [x, y, z],
    "reasoning": "brief explanation"
}}"""


class QwenVLWrapper:
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-2B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.last_inference_time = 0.0
        self.last_output = None

    def load(self):
        try:
            import torch
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            print(f"Qwen3-VL loaded: {self.model_name} on {self.model.device}")
        except Exception as e:
            print(f"Qwen3-VL load failed: {e}")
        return self

    def understand(self, image: np.ndarray, instruction: str,
                   objects: list, table_top: float = 0.05) -> Dict:
        start = time.time()

        if self.model is not None:
            result = self._inference(image, instruction, objects, table_top)
        else:
            result = self._rule_based(instruction, objects, table_top)

        self.last_inference_time = (time.time() - start) * 1000
        self.last_output = result
        return result

    def _inference(self, image, instruction, objects, table_top):
        from PIL import Image
        import torch

        pil_img = Image.fromarray(image)
        prompt = GROUNDING_PROMPT.format(
            instruction=instruction,
            objects=", ".join(objects),
        )

        messages = [{"role": "user", "content": [
            {"type": "image", "image": pil_img},
            {"type": "text", "text": prompt},
        ]}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[pil_img], return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
            )

        response = self.processor.decode(output[0], skip_special_tokens=True)
        return self._parse_response(response, objects, table_top)

    def _parse_response(self, response, objects, table_top):
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(response[start:end])
                pick = result.get("pick", "")
                place = result.get("place_xyz", [0.75, 0.2, table_top + 0.02])
                if pick not in objects:
                    pick = objects[0] if objects else "red_cube"
                return {
                    "pick": pick,
                    "place_xyz": place,
                    "reasoning": result.get("reasoning", ""),
                }
        except json.JSONDecodeError:
            pass
        return self._rule_based(response, objects, table_top)

    def _rule_based(self, instruction, objects, table_top):
        lower = instruction.lower()
        pick = objects[0] if objects else "red_cube"
        for name in objects:
            color = name.split("_")[0]
            if color in lower:
                pick = name
                break
        place_xyz = [0.75, 0.2, table_top + 0.02]
        return {
            "pick": pick,
            "place_xyz": place_xyz,
            "reasoning": f"Rule-based fallback: pick {pick}",
        }

    def get_inference_time(self):
        return self.last_inference_time
