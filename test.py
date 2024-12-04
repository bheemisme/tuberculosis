import json

with open('tuberculosis-443609-18c56fe4746a.json', 'r') as f:
    data = json.load(f)

json_string = json.dumps(data)
print(json_string)