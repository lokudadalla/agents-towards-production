import requests
import json

res = requests.post(
    "http://localhost:8000/research/stream",
    json={"query": "AI and education"},
    stream=True,
)

for line in res.iter_lines():
    if line and line.startswith(b"data: "):
        data = json.loads(line[6:])
        try:
            print(
                f"{data.get('token')}",
                end="",
                flush=True,
            )
        except json.JSONDecodeError:
            print("\n Warning: Could not decode server response.")
