> 命令中的项目路径相对于项目根目录；先激活已安装所需依赖的 Python 环境。

# A3C-GCN 节点选择 + EO 有向链路映射

本程序独立于 `NFV/test/nfv_realtime.py`，不修改 EO 原程序。
Actor 只选节点，使用训练时的11维特征、物理节点排序、掩码及虚拟节点顺序。
链路从数据库提供的候选中按跳数选择第一条满足剩余带宽的路径，并累计同一请求的带宽占用。
只在整个请求映射成功后扣资源。离开时释放当时实际占用的资源。

## 运行

从项目根目录使用模块入口（不是 pytest，也不是 train.py）：

```powershell
# Run from the project root
$env:EO_DB_PASSWORD = "你的数据库密码"
python -m NFV.a3c_gcn.a3c_gcn_realtime --config NFV\a3c_gcn\mapping_config.example.yaml
```

默认只读数据库，在内存中完成事件重放、映射、释放及资源统计；不计算温度/流量。

如需像原 `nfv_realtime` 一样在数据库中扣减、释放映射资源：

```powershell
python -m NFV.a3c_gcn.a3c_gcn_realtime --config NFV\a3c_gcn\mapping_config.example.yaml --write-db
```

`--write-db` 只修改 `node_single_table.cpu`、`link_single_table.bw_resource`，
不运行流量或温度仿真，不读取/写入 `current_flow`。
请使用独立实验数据库，禁止和其他实验/真实业务同时运行。
数据库资源必须处于满容量，程序会检查但不会自动重置。写入模式要求资源表使用 InnoDB，
节点与链路的资源扣减在同一事务中完成；路径表只读，可行性始终按当前单跳剩余带宽判断。
正常结束、Python 异常或 Ctrl+C 时会尝试释放本次剩余请求，检查汇总中的 cleanup_errors。
强制结束进程、断电或数据库失联不能保证清理成功。

## 配置与实验口径

- `checkpoint` 指向 actor_best 的纯 Actor state_dict；自动读取嵌入维度。
- `event_file` 是实验事件，`vnet_dir` 中必须存在对应 VNR；正式实验请换成独立测试数据。
- `max_paths_per_pair: 100` 与现有训练一致；`null` 读取全部数据库路径，与原 EO
  全量候选范围一致，可能消耗大量时间和内存。不同范围的结果必须分别标注。
- `distance_limit: 10.0` 与现有训练一致；不支持同一 VNR 重复使用物理节点。
- `time_window: 100` 模仿 EO：窗口内先处理离开、再处理到达。它不同于训练逐事件
  按时间执行的口径；同一窗口内先到后离的短请求也会受到此策略影响。
- 本实验不调用 MultiVNRFlowSimulator 或 TrafficMatrixCalculator，无流量、温度模拟。
  为保留 EO 资源记录的顶层字段，写入 `domain_temperatures: []`，不伪造温度值。
- 模型初始化不计入逐请求 mapping_seconds；该时间包括选点、链路映射和资源提交，
  不包括输出 YAML/JSON 文件的时间。跨算法比较耗时时应统一计时边界。

默认输出到 `NFV/result_storage/A3C_GCN/`，每次自动新建下一个数字目录
（`1/`、`2/`、……），不覆盖之前结果，不修改 `result_storage/EO/` 中的样例。
可通过 `output_dir` 改变算法结果根目录。两种模式都输出相同格式：

- `run_config.yaml`：实际配置（不保存数据库密码）。
- `mapping_results.jsonl`：逐请求节点/链路映射、成功状态、失败原因和耗时。
- `vne_mapping.yaml`：与 EO 样例相同的成功请求列表；自增 `id` 从1开始，
  `vnr_ind` 为请求ID，`event_id` 保留事件文件原ID，`event_time` 为原到达时间；
  `links` 中只含 `source`、`target`、`bw`、`linkpath`。失败请求不写入该文件。
- `resource_usage.yaml`：顶层 `records`；每窗口一条 `time`、`bandwidth`、`cpu`、
  `acceptance_rate`、`domain_temperatures`。资源项的百分比保留4位小数，接受率为
  截至该窗口累计成功数/累计到达数。温度列表为空。
- `experiment_summary.json`：最终接受率、失败计数、路径范围、事件策略和清理状态。
- 不再生成 `vnr_flow_history.yaml`；此前版本产生的历史文件不会删除。

## 离线测试

```powershell
python -m pytest NFV/a3c_gcn/test_selector.py -q
```

测试不连接数据库，不修改数据库，也不重新训练权重。
