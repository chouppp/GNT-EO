> 命令中的项目路径相对于项目根目录；先激活已安装所需依赖的 Python 环境。

# 当前窗口前35%观测模式

> 已撤回仿真入口接入：按照用户要求，`test/nfv_realtime.py` 已恢复到修改前备份，LSTM 保持独立实验模块。下文关于 `GNT_PREFIX_RUN` 和启动仿真采集的步骤是已撤回方案的历史记录，不再适用，请勿执行。`prefix35` 数据接口仍保留在 lstm 内，但当前缺少真实前35%观测，不能直接用旧日志训练此模式。此次恢复没有改动独立的 `lstm_predict.py`。

采集功能已接入 `../test/nfv_realtime.py`，其余新代码位于 lstm。历史窗口预测模式仍可使用。原入口备份在 `work/backup_before_prefix_collection/nfv_realtime.py`。

## 数据口径

目标标签继续直接读取 `../result_storage/EO/<run>/resource_usage.yaml`：逐域截断 T 到 ±10000，空 T 按0，同时间重复记录取平均。不修改真实曲线。

新模式每100单位时间一个窗口，输入窗口起点+5、10、15、20、25、30、35的五域 F/H/T，预测该窗口日志 GNT。比如目标500，输入405至435。训练/验证/测试默认仍为实验1–5/6/7–8，标准化只拟合训练组。

## 旧数据限制（新模式需重新采集）

现有 `vnr_flow_history.yaml` 把多次状态统一标为窗口结束时间，缺少真实观测时间。`test/nfv_realtime.py` 还先处理整个窗口的离开事件，再处理到达事件；窗口后半段事件可能已经影响前半段状态。映射日志中的 event_time 不能单独解决这个因果问题。

`traffic_generator.py` 从窗口末态生成的20个人工子步不能用于这个模式。没有将它们改名或平移为真实前35%观测，也没有声称预测精度已经改善。

## 检查与运行

在 lstm 目录执行：

```powershell
& python -X utf8 -B gnt_predict.py --task prefix35 --check-data-only
& python -X utf8 -B gnt_predict.py --task prefix35
```

缺数据会在加载 TensorFlow、训练和输出结果之前报错。新结果使用 `GNT/direct_gnt/prefix35_holdout`，不会覆盖历史模式输出。

## 采集接口及接入边界

`gnt_prefix.record_observation(run, window_end, sample_time, latest_processed_event_time, domains)` 将真实快照写入 `GNT/observations/<run>/prefix35.jsonl`。

domains 必须是5项字典，字段 domain_id（0到4）、F、H、T。T须沿用仿真的计算定义；接口不推进仿真、不重算温度、不写数据库。latest_processed_event_time 必须是真实的已处理事件时间上界，不能人为填成 sample_time 来绕过检查。格式和时间校验不能代替采集来源的真实性保证。

原入口通过环境变量 `GNT_PREFIX_RUN` 开启采集。开启后按时间顺序合并到达、离开与采样事件，同时间先处理事件再采样。7个采样点会实际调用流量仿真并刷新数据库，随后读取 F/H/T；不会根据窗口末态人工生成输入。T 沿用温度计算器定义，以最近一次到达决策维护的 prev_tms 为参考；采集读取不更新该决策参考。窗口结束刷新流量后重新计算目标温度，避免沿用最后一次到达前的旧温度。没有到达事件的窗口也可采集。

这会改变随机流量推进次数及事件执行顺序，属于新的实验协议，不保证复现旧曲线，也不保证精度提高。应在实验方法中披露这一点。所有新观测、流量日志和目标标签写入 `GNT/collected/<run>`；已有同名目录会在连接数据库之前被拒绝，避免混合实验。映射算法及其原有其他输出保持原实现。未设置环境变量时仍按原协议运行。

在原来运行仿真的环境及工作目录（通常为 test）中执行：

```powershell
$env:GNT_PREFIX_RUN = '1'
& python -X utf8 -B ./nfv_realtime.py
Remove-Item Env:GNT_PREFIX_RUN
```

这条命令会像原仿真一样写数据库。此次代码修改没有执行它。每次实验仍需按项目原方式初始化数据库、随机种子及事件输入；将运行编号依次改为1至8，不要把同一次运行复制成多个实验。若运行失败，保留失败目录排查，另选新编号重跑；不要向不完整数据追加实验。

采集完8次后，在 lstm 中执行：

```powershell
& python -X utf8 -B gnt_predict.py --task prefix35 --raw-base-dir ./GNT/collected --observation-dir ./GNT/collected --check-data-only
& python -X utf8 -B gnt_predict.py --task prefix35 --raw-base-dir ./GNT/collected --observation-dir ./GNT/collected
```

新仿真轨迹不能与旧8次实验标签拼接。旧曲线可保留作旧实验参考，但不能要求新实验产生完全相同的真实值。

后续需要实际运行采集以生成匹配的观测/标签，才能训练和报告真实的测试误差。增加轮数或复用训练集作为验证集不能补齐这些缺失数据。
