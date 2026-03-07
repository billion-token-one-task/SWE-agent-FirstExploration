# Probe Assets

这里收集了本次实验用到的探针相关脚本与补丁，方便单独复现实验前的模型/参数探测。

## 文件列表

- `litellm_*_probe.py`: LiteLLM 接口能力探测（responses/stream/toolcall/tabcode 等）
- `probe_temp_none.py`: `temperature=None` 探测
- `probe_top_p_none.py`: `top_p=None` 探测
- `build_lite83.py`: 构建本地 `lite83` 子集的辅助脚本
- `find_local_swebench_matches.py`: 本地实例匹配辅助脚本
- `thermo_probe.patch` / `thermo_probe_clean.patch`: 探针相关补丁
- `config/litellm_model_registry_gpt54.json`: 探针中使用的模型注册配置

## 说明

- 这些脚本来自原工作区 `/data/liora/tmp` 与 `SWE-agent-src` 根目录。
- 为了便于上传 GitHub，已统一归档到本目录，不再依赖 `tmp/`。
