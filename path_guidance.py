from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import re
import subprocess
import torch


def extract_json(text):
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        text = fenced_match.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")

    return json.loads(text[start:end + 1])


def speak(text):
    subprocess.run(["say", text], check=False)

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(model_id)

image_path = "frame.jpg"

detected_objects = """
Detected objects:
- person: large obstacle in the left/center-left foreground
- bushes: blocking the left side of the sidewalk
- pole: thin obstacle on the far right ahead
- open sidewalk: center-right area appears most walkable
"""

prompt = f"""
You are assisting a visually impaired user walking forward on a sidewalk.

Here are detected objects:
{detected_objects}

Use the image and detected objects to choose the safest immediate direction.

Rules:
- Image left means user's left.
- Image right means user's right.
- Only sidewalk is walkable.
- Avoid people and obstacles.
- Prefer the most open sidewalk region.
- Give one short instruction.

Choose exactly one:
- continue straight
- move slightly left
- move slightly right
- stop

Return only JSON:
{{
  "direction": "...",
  "reason": "...",
  "spoken_instruction": "..."
}}
"""

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt}
        ]
    }
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to(model.device)

generated_ids = model.generate(
    **inputs,
    max_new_tokens=128
)

generated_ids_trimmed = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
]

output = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]

print(output)

try:
    guidance = extract_json(output)
    spoken_instruction = guidance.get("spoken_instruction")
    if spoken_instruction:
        speak(spoken_instruction)
except (json.JSONDecodeError, ValueError) as error:
    print(f"Could not speak instruction: {error}")
