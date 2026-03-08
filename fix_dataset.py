import json
import re

INPUT_FILE  = "dataset.jsonl"
OUTPUT_FILE = "dataset_fixed.jsonl"


def fix_text(text):
    """Convert escaped \\n and \\t to real newlines and tabs."""
    if not isinstance(text, str):
        return text
    return text.replace("\\n", "\n").replace("\\t", "\t")


def extract_json_objects(text):
    """Extract all valid JSON objects from messy text."""
    objects = []
    depth = 0
    start = -1

    for i, char in enumerate(text):
        if char == '{':
            if depth == 0:
                start = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start != -1:
                candidate = text[start:i+1]
                try:
                    obj = json.loads(candidate)
                    objects.append(obj)
                except json.JSONDecodeError:
                    # Try fixing common issues
                    try:
                        # Fix unescaped newlines inside strings
                        fixed = re.sub(r'(?<!\\)\n', '\\\\n', candidate)
                        obj = json.loads(fixed)
                        objects.append(obj)
                    except json.JSONDecodeError:
                        pass
                start = -1

    return objects


# Read entire file as text
print(f" Reading {INPUT_FILE}...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Remove markdown code blocks
raw_text = re.sub(r'```(?:json|jsonl|python)?\s*', '', raw_text)
raw_text = re.sub(r'```', '', raw_text)

#  Extract all JSON objects 
print("🔍 Extracting JSON objects...")
all_objects = extract_json_objects(raw_text)

#  Process and fix each object
fixed = []
skipped = 0

for obj in all_objects:
    # Must have at minimum instruction and output
    if "instruction" not in obj or "output" not in obj:
        skipped += 1
        continue

    fixed_item = {
        "instruction": fix_text(obj.get("instruction", "")),
        "output":      fix_text(obj.get("output", "")),
    }

    # Optional fields
    if "explanation" in obj:
        fixed_item["explanation"] = fix_text(obj["explanation"])
    if "examples" in obj:
        fixed_item["examples"] = fix_text(obj["examples"])

    fixed.append(fixed_item)


# Remove duplicates by instruction 
seen = set()
unique = []
for item in fixed:
    key = item["instruction"].strip().lower()
    if key not in seen:
        seen.add(key)
        unique.append(item)

duplicates_removed = len(fixed) - len(unique)


#  Save fixed dataset 
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in unique:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


#  Summary 
has_explanation = sum(1 for item in unique if "explanation" in item)
has_examples    = sum(1 for item in unique if "examples" in item)

print(f"\n Results:")
print(f"   Total extracted  : {len(all_objects)}")
print(f"   Valid samples    : {len(fixed)}")
print(f"   Duplicates removed: {duplicates_removed}")
print(f"   Skipped (no instruction/output): {skipped}")
print(f"   Final unique samples: {len(unique)}")
print(f"   Saved to: {OUTPUT_FILE}")

print(f"\n Field summary:")
print(f"   instruction : {len(unique)}")
print(f"   output      : {len(unique)}")
print(f"   explanation : {has_explanation}")
print(f"   examples    : {has_examples}")

# Show sample 
if unique:
    print(f"\n Sample (first item):")
    print(f"INSTRUCTION : {unique[0]['instruction']}")
    print(f"OUTPUT      :\n{unique[0]['output'][:200]}...")
    if "explanation" in unique[0]:
        print(f"EXPLANATION : {unique[0]['explanation'][:100]}...")
    if "examples" in unique[0]:
        print(f"EXAMPLES    : {unique[0]['examples'][:100]}...")
