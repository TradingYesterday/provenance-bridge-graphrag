# 实验协议

## 本批验收

同一缺边案例在 `allow_successor_reentry=false` 时不取得新的后继图证据；oracle 文本绑定在重入开启时可以取回后继边。冲突、不可链接、撤销和预算耗尽有各自的状态。图文件不被永久加边。

诊断目录由 `python scripts/run_diagnostics.py` 重放。期望看的是证明种类、来源和状态，不是只看最终城市名。

## 还不能做的比较

下面的方法在本批标记为不可用：

- CS-RAG upstream 的真实运行
- Core 00 / 01 / 10 / 11 的完整模型矩阵
- Simple text bind
- CoG controlled
- Iterative text

配置里预留了开关，但没有模型调用时不能把诊断通过写成机制成立。

## 进入 100–200 题之前

- 真实 Phase 1 / Phase 2 / KG / 2Wiki 路径显式存在，并且 20 题定位没有 ID 或来源错位
- 诊断集里的正确绑定、冲突拒绝、不可链接、循环和预算耗尽全部符合预期
- oracle bridge 能稳定触发图重入
- 相同输入的消融共享候选缓存，不同绑定的缓存键分开
- prompt、模型版本、temperature 和重试已经冻结
- 有总预算；到达额度就停

`scripts/run_pilot.py` 在这些条件满足前返回非零。

## 指标

证据级、证明级、答案级和成本级指标留在下一批。空证明的精度不能记成满分。gold 不能由正在测试的抽取器或验证器生成。
