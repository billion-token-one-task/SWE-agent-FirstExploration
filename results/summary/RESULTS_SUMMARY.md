# SWE-agent Results Summary

Generated: 2026-03-07 14:19:06 UTC

## Overview

- Eval summary JSON files: 15
- Report files scanned: 916
- Instance-level records: 916
- Resolved true/false/unknown: 505/411/0
- Trajectory run directories: 32
- Total `.traj` files: 1186

## Top Eval Runs (by resolved_instances)

| run_id | resolved | completed | submitted | total | resolved/total % | file |
| --- | --- | --- | --- | --- | --- | --- |
| lite83__gpt54__tok200k_mt8_probev3_gpt54_rerun3__t1_pnone_c0 | 61 | 82 | 83 | 300 | 20.33 | summary__lite83__gpt54__tok200k_mt8_probev3_gpt54_rerun3__t1_pnone_c0__main.json |
| lite83__gpt53-codex__tok100k_mt8_probev3_rerun1__t1_pnone_c0 | 49 | 79 | 83 | 300 | 16.33 | summary__lite83__gpt53-codex__tok100k_mt8_probev3_rerun1__t1_pnone_c0__main.json |
| lite83__gpt53-codex__local_cached__t1_pnone_c0 | 48 | 78 | 81 | 300 | 16.0 | summary__lite83__gpt53-codex__local_cached__t1_pnone_c0__submitted.json |
| lite83__gpt54__tok100k_mt8_probev3_gpt54_rerun1__t1_pnone_c0 | 48 | 74 | 83 | 300 | 16.0 | summary__lite83__gpt54__tok100k_mt8_probev3_gpt54_rerun1__t1_pnone_c0__main.json |
| lite83__gpt53-codex__nobudget_mt8_probev3_rerun1__t1_pnone_c0 | 47 | 74 | 83 | 300 | 15.67 | summary__lite83__gpt53-codex__nobudget_mt8_probev3_rerun1__t1_pnone_c0__main.json |
| lite83__gpt53-codex__tok100k__t1_pnone_c0 | 46 | 76 | 79 | 300 | 15.33 | summary__lite83__gpt53-codex__tok100k__t1_pnone_c0__localds_retryfull2.json |
| lite83__gpt53-codex__tok50k_mt4_probev2__t1_pnone_c0 | 44 | 82 | 83 | 300 | 14.67 | summary__lite83__gpt53-codex__tok50k_mt4_probev2__t1_pnone_c0__main.json |
| lite83__gpt53-codex__tok30k_mt4_probev3__t1_pnone_c0 | 43 | 73 | 83 | 300 | 14.33 | summary__lite83__gpt53-codex__tok30k_mt4_probev3__t1_pnone_c0__main.json |
| lite83__gpt53-codex__tok25k_mt8_probev3_rerun2__t1_pnone_c0 | 37 | 79 | 83 | 300 | 12.33 | summary__lite83__gpt53-codex__tok25k_mt8_probev3_rerun2__t1_pnone_c0__main.json |
| lite83__gpt53-codex__tok25k_mt8_probev3_closed_window_rerun1__t1_pnone_c0 | 32 | 72 | 83 | 0 | 0.0 | summary__lite83__gpt53-codex__tok25k_mt8_probev3_closed_window_rerun1__t1_pnone_c0__partial.json |

## Top Report Aggregates (by resolved_rate_pct)

| source | eval_session | run_id | instances | resolved_true | resolved_rate % |
| --- | --- | --- | --- | --- | --- |
| from_swe_agent_src | session__lite83__gpt53-codex__local_cached__t1_pnone_c0__smoke2 | lite83__gpt53-codex__local_cached__t1_pnone_c0 | 2 | 2 | 100.0 |
| from_swe_agent_src | session__lite83__gpt54__tok200k_mt8_probev3_gpt54_rerun3__t1_pnone_c0__main | lite83__gpt54__tok200k_mt8_probev3_gpt54_rerun3__t1_pnone_c0 | 82 | 61 | 74.39 |
| from_swe_agent_src | session__lite83__gpt54__tok100k_mt8_probev3_gpt54_rerun1__t1_pnone_c0__main | lite83__gpt54__tok100k_mt8_probev3_gpt54_rerun1__t1_pnone_c0 | 74 | 48 | 64.86 |
| from_swe_agent_src | session__lite83__gpt53-codex__nobudget_mt8_probev3_rerun1__t1_pnone_c0__main | lite83__gpt53-codex__nobudget_mt8_probev3_rerun1__t1_pnone_c0 | 74 | 47 | 63.51 |
| from_swe_agent_src | session__lite83__gpt53-codex__tok100k_mt8_probev3_rerun1__t1_pnone_c0__main | lite83__gpt53-codex__tok100k_mt8_probev3_rerun1__t1_pnone_c0 | 79 | 49 | 62.03 |
| from_swe_agent_src | session__lite83__gpt53-codex__local_cached__t1_pnone_c0__submitted | lite83__gpt53-codex__local_cached__t1_pnone_c0 | 78 | 48 | 61.54 |
| from_swe_agent_src | session__lite83__gpt53-codex__tok100k__t1_pnone_c0__localds_retryfull2 | lite83__gpt53-codex__tok100k__t1_pnone_c0 | 76 | 46 | 60.53 |
| from_swe_agent_src | session__lite83__gpt53-codex__tok30k_mt4_probev3__t1_pnone_c0__main | lite83__gpt53-codex__tok30k_mt4_probev3__t1_pnone_c0 | 73 | 43 | 58.9 |
| from_swe_agent_src | session__lite83__gpt53-codex__tok50k_mt4_probev2__t1_pnone_c0__main | lite83__gpt53-codex__tok50k_mt4_probev2__t1_pnone_c0 | 82 | 44 | 53.66 |
| from_swe_agent_src | session__lite83__gpt53-codex__tok25k_mt8_probev3_rerun2__t1_pnone_c0__main | lite83__gpt53-codex__tok25k_mt8_probev3_rerun2__t1_pnone_c0 | 79 | 37 | 46.84 |
