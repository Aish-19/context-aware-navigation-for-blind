import json
import re
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

_model = None
_processor = None


def get_model():
    global _model
    if _model is None:
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype="auto",
            device_map="auto",
        )
    return _model


def get_processor():
    global _processor
    if _processor is None:
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
    return _processor


def extract_json(text):
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        text = fenced_match.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")

    return json.loads(text[start:end + 1])


def build_prompt():
    return """
You are assisting a visually impaired user walking forward on a sidewalk.

Use the image to choose the safest immediate direction.

Rules:
- Image left means user's left.
- Image right means user's right.
- Only sidewalk and open walking surfaces are walkable.
- Avoid people, poles, vehicles, curbs, walls, bushes, and obstacles.
- Prefer the most open walkable region.
- If the safe direction is unclear, choose stop.
- Give one short instruction.

Choose exactly one:
- continue straight
- move slightly left
- move slightly right
- stop

Return only JSON:
{
  "direction": "...",
  "reason": "...",
  "spoken_instruction": "..."
}
"""


def analyze_image(image_path):
    model = get_model()
    processor = get_processor()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(Path(image_path))},
                {"type": "text", "text": build_prompt()},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=96,
    )

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]

    output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return extract_json(output)
