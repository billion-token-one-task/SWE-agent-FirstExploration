# SWE-agent Experiment Release

本目录已经整理为可直接上传到 GitHub 的发布结构，包含三部分：

- `code/`: 干净的 `SWE-agent` 代码快照（已排除 `.venv`、缓存、运行期产物）
- `experiments/`: 实验配置、运行脚本、数据集定义、评测汇总 JSON
- `results/`: 原始实验结果（轨迹、评测历史、运行日志）和自动汇总结果

## 目录说明

- `code/SWE-agent-src/`
  - 仓库代码快照（可作为后续开发基线）
- `experiments/config/`
  - 实验配置文件（来自 `SWE-agent-src/config`）
- `experiments/run_scripts/`
  - 运行脚本与轨迹分析脚本
- `experiments/eval_summaries/`
  - 多次评测的总览 JSON（统一命名为 `summary__<run_setting>__<stage>.json`）
- `experiments/dataset/`
  - 本地实验用数据集配置（如 `lite_test_local83.json`）
- `experiments/probes/`
  - 探针脚本、探针补丁、LiteLLM 探针模型配置（原先在 `tmp/` 与 `SWE-agent-src` 根目录）
- `results/trajectories/`
  - 全部轨迹与实例级运行日志（`.traj` / `.pred` / `*.debug.log` 等）
- `results/evaluation_history/`
  - 评测产物（`report.json`、`test_output.txt`、`patch.diff`、`run_instance.log`）
- `results/runner_logs/`
  - 批量运行过程日志
- `results/summary/`
  - 自动生成的汇总文件：
    - `RESULTS_SUMMARY.md`
    - `eval_summaries_overview.csv`
    - `report_run_aggregates.csv`
    - `instance_reports.csv`
    - `trajectory_runs_overview.csv`
- `experiments/notes/rename_map.tsv`
  - 最近一次重命名操作的增量记录（便于追踪）

## 当前整理结果

- 总体体积约 `635MB`（`code` 约 18MB，`results` 约 617MB）
- 评测汇总 JSON：15 份
- 扫描到实例级 `report.json`：916 条实例记录
- 轨迹 run 目录：32 个；`.traj` 文件：1186 个

详细统计请看：
- `results/summary/RESULTS_SUMMARY.md`

## 重新生成汇总

在本目录执行：

```bash
./scripts/generate_results_summary.py
```

会刷新 `results/summary/` 下所有统计文件。

## 上传 GitHub

```bash
cd /data/liora/swe-agent-lite83-probe-study
git init
git add .
git commit -m "Add SWE-agent experiment code and results snapshot"
git branch -M main
git remote add origin <你的GitHub仓库URL>
git push -u origin main
```

## 注意事项

- 该目录不包含 `.env`（避免密钥泄露）。
- 当前没有单文件超过 GitHub 100MB 限制；如后续新增大文件，建议使用 Git LFS。
- `summary__lite83__gpt53-codex__tok25k_mt8_probev3_closed_window_rerun1__t1_pnone_c0__partial.json` 是部分评测快照（`total_instances=0`），属于未完整收敛的中间结果。
