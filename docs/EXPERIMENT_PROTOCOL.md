# 实验协议

## 本批验收

同一缺边案例在 `allow_successor_reentry=false` 时不取得新的后继图证据；oracle 文本绑定在重入开启时可以取回后继边。冲突、不可链接、撤销和预算耗尽有各自的状态。图文件不被永久加边。

诊断目录由 `python scripts/run_diagnostics.py` 重放。期望看的是证明种类、来源和状态，不是只看最终城市名。

## 第二批对照

`execution/methods.py` 在同一执行器的拷贝上运行 Core 00 / 01 / 10 / 11、simple slot、受控 CoG notebook 和 iterative text。CoG 这里只保留事实、线索、判断、分析和后续查询，不使用 B 的关系提交门槛，也不等于原论文的全球 KG 设定。

上游 CS-RAG 进程仍然不执行。合成诊断上的配对差值不是论文结论。

## 指标

`bridge_rag.evaluation` 计算后继证据、完整支持片段、证明精度、答案 EM/F1 和 trace 里的 token。空证明的精度记为不可计算。gold 只在评价端，不进执行器。

## 进入 100–200 题之前

- 真实 Phase 1 / Phase 2 / KG / 2Wiki 路径显式存在，并且 20 题定位没有 ID 或来源错位
- 诊断集里的正确绑定、冲突拒绝、不可链接、循环和预算耗尽全部符合预期
- oracle bridge 能稳定触发图重入
- 相同输入的消融共享候选缓存，不同绑定的缓存键分开
- prompt、模型版本、temperature 和重试已经冻结
- 有总预算；到达额度就停

`scripts/run_pilot.py` 在这些条件满足前返回非零。

