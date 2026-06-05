import json
import re
import subprocess


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


sample_model_output = """```json
{
  "direction": "move slightly right",
  "reason": "The sidewalk is clearer on the right.",
  "spoken_instruction": "Move slightly to the right."
}
```"""

guidance = extract_json(sample_model_output)
spoken_instruction = guidance["spoken_instruction"]

print(sample_model_output)
print(f"Speaking: {spoken_instruction}")
speak(spoken_instruction)
