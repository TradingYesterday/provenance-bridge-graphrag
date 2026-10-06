# AutoDL

本批不租机器，也不跑付费 API。下面只固定以后迁移时的入口。

LLM 使用运行机器上的 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`。不要把密钥写进配置或仓库。

检索与重排计划使用 `sentence-transformers/all-MiniLM-L6-v2` 和 `BAAI/bge-reranker-v2-m3`。本批不下载它们。CPU 内核不能依赖 CUDA。

数据目录由 `BRIDGE_DATA_ROOT` 指向仓库外的磁盘。输出写到可保留的数据盘，按题追加。配置和数据 hash 一致时才续跑。

建议顺序：环境自检、输入审计、无 API 诊断、10 题 API 冒烟、再考虑矩阵先导。10 题冒烟测到每方法的调用、token、重排时间和延迟之前，不用 “每题一次生成” 估算费用。

`python scripts/run_pilot.py` 当前会拒绝大规模运行。
