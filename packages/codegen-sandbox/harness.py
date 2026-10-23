import json
import sys

sys.path.insert(0, ".")

with open("request.json", encoding="utf-8") as f:
    request = json.load(f)

import generated  # noqa: E402

result = getattr(generated, request["entrypoint"])(*request["args"])
sys.stdout.write("\n__RESULT__" + json.dumps(result))
