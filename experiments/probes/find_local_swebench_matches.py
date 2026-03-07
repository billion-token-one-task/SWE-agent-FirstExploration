import re
import subprocess
from datasets import load_dataset

imgs = subprocess.check_output(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"], text=True).splitlines()
pat = re.compile(r"swebench/sweb\.eval\.x86_64\.(.+):latest$")
local_ids = {pat.search(i).group(1).replace("_1776_", "__") for i in imgs if pat.search(i)}
print("local", len(local_ids))

cands = [
    ("princeton-nlp/SWE-Bench", "dev"),
    ("princeton-nlp/SWE-Bench", "test"),
    ("princeton-nlp/SWE-Bench_Verified", "dev"),
    ("princeton-nlp/SWE-Bench_Verified", "test"),
    ("princeton-nlp/SWE-Bench_Lite", "dev"),
    ("princeton-nlp/SWE-Bench_Lite", "test"),
]

for ds_name, split in cands:
    try:
        ds = load_dataset(ds_name, split=split)
    except Exception as e:
        print(ds_name, split, "ERR", type(e).__name__, str(e)[:120])
        continue
    ids = [r["instance_id"] for r in ds]
    inter = [i for i in ids if i in local_ids]
    print(ds_name, split, "total", len(ids), "match", len(inter), "sample", inter[:10])
