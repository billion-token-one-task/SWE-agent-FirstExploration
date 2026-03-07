import json
import re
import subprocess
from pathlib import Path
from datasets import load_dataset

imgs = subprocess.check_output(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"], text=True).splitlines()
pat = re.compile(r"^swebench/sweb\.eval\.x86_64\.(.+):latest$")
local_ids = {pat.match(i).group(1).replace("_1776_", "__") for i in imgs if pat.match(i)}

lite_test = load_dataset("princeton-nlp/SWE-Bench_Lite", split="test")
sel = []
for row in lite_test:
    iid = row["instance_id"]
    if iid not in local_ids:
        continue
    image = f"docker.io/swebench/sweb.eval.x86_64.{iid.replace('__', '_1776_')}:latest".lower()
    sel.append(
        {
            "instance_id": iid,
            "image_name": image,
            "repo_name": "testbed",
            "base_commit": row.get("base_commit", "HEAD"),
            "problem_statement": row["problem_statement"],
        }
    )

out = Path('/data/liora/SWE-agent-src/lite_test_local83.json')
out.write_text(json.dumps(sel, ensure_ascii=False, indent=2))
print('written', out)
print('count', len(sel))
print('first10', [x['instance_id'] for x in sel[:10]])
