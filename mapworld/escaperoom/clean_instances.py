import json
import os

with open(os.path.join("escaperoom", "in", "instances.json"), 'r') as f:
    json_data = json.load(f)

with open(os.path.join("escaperoom", "in", "instances.json"), 'w') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)