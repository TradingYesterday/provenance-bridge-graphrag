# provenance-bridge-graphrag

首轮只检验一件事：查询计划和入口正确、某条非末跳关系在图里缺失、支持它的文本还在、恢复出的桥实体仍能在当前图里寻址时，带来源校验的关系绑定能不能把系统重新送回图检索，并取到后继证据。

当前包含任务 1–6 的可运行代码：数据契约、CS-RAG 只读适配、无 API 执行内核、mock/HTTP 模型接口、同一执行器上的 Core 与三个对照，以及评价端指标。上游 CS-RAG 进程不执行。合成诊断上的差值不是论文结论。

## 运行

```bash
pip install -e ".[dev]"
python -m pytest
python scripts/run_diagnostics.py --out outputs/diagnostics
python scripts/summarize_results.py outputs/diagnostics/summary.json
python scripts/validate_inputs.py --config configs/2wiki_pilot.yaml
```

诊断集是合成缺边案例，不是 2Wiki 分数。`scripts/run_pilot.py` 会拒绝启动 100–200 题 API 实验。

第二批在同一执行器上提供 Core 00/01/10/11、`simple_bind`、`cog_controlled` 和 `iterative_text`。模型调用走 OpenAI 兼容接口；测试使用 mock 和本地 HTTP，没有密钥时直接报错。配对指标在 `bridge_rag.evaluation`。上游 CS-RAG 进程仍然不执行。

## 边界

- 模型走环境变量里的 OpenAI 兼容接口。没有密钥时直接报错，不会编造答案。
- 诊断默认不加载检索模型。`scripts/smoke_retrieval.py` 在显卡空闲时才会加载 MiniLM 和 bge 重排；已有计算进程占用 GPU 时直接退出。
- 显存实验结束后再跑 GPU 冒烟。
- 不复制 CS-RAG 源码。适配器只读 JSON / JSONL。固定提交是 `myz12138/CS-RAG@c84a081c728dd45c284a7165031c853eef4e8f24`。
- 数据、KG、权重和 `.env` 不入库。

更细的约定见 `docs/`。
