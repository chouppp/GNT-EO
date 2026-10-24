> 命令中的项目路径相对于项目根目录；先激活已安装所需依赖的 Python 环境。

# 原始 GNT 直接预测

## 2026-09-11 当前默认：多组生成训练集，当前样本独立验证

默认入口根据现有8次实验的窗口基准流量，用不同随机种子生成8组新训练数据。
每组数据都先对8次生成轨迹逐窗口、逐小段、逐链路取平均，再以前7段（35%）
汇总出的五域流量与熵序列作为输入。训练目标直接读取8次
`resource_usage.yaml` 的平均 recorded GNT，不再预测后13段流量后重算另一种窗口GNT。
磁盘中原有8次平均流量样本不参与拟合，只用于早停验证和最终结果报告。输出写入
`GNT/window_generated_validation`，生成训练集保存在其 `generated_training_sets` 子目录。

训练集与验证集使用相同的窗口基准流量，但小段扰动随机种子不同；生成输入还复用了
同一条 recorded GNT 标签曲线。因此这是当前场景的抗扰动拟合验证，不代表对全新
网络负载或全新GNT标签的外推验证。

```powershell
Set-Location 'NFV/lstm'
& 'python' -X utf8 -B ./lstm_predict.py
& 'python' -X utf8 -B ./picture.py
```

`picture.py` 默认真实曲线读取 `../result_storage/EO/1..8/resource_usage.yaml`，
与原论文图的 recorded GNT 保持一致。若只想诊断窗口模型自身的重算目标，可显式使用
`./picture.py --actual-source window`；该曲线不能称为原始记录的真实GNT。

可用 `--generated-training-sets 16` 增加生成组数。旧的同样本训练拟合仍可通过
`--window-evaluation in_sample` 运行。

## 2026-09-10 旧模式：8次实验平均序列

> 当前最新默认入口会先对8次实验的20段窗口流量逐窗口、逐小段、逐链路求平均，形成一份平均窗口数据；然后以前7段（35%）为输入、后13段为训练目标，并在同一份平均数据上输出训练集拟合。结果写入 `GNT/window_mean_in_sample`。本节其余平均GNT模式仍可通过 `--target-mode mean` 运行。

`lstm_predict.py` 现在先对8次实验的同一时刻、同一域取平均，再以过去3个平均状态拟合平均GNT。按照当前实验要求，默认将500–2000全部16个点同时作为训练、验证和预测数据。这是训练集拟合，不是未见数据预测。

```powershell
Set-Location 'NFV/lstm'
& 'python' -X utf8 -B ./lstm_predict.py
& 'python' -X utf8 -B ./picture.py
```

结果位于 `GNT/mean_lstm`。默认使用64个隐藏单元、最多800轮训练，在同一训练集上输出LSTM拟合结果。若要恢复时间外推评估，使用 `--mean-evaluation chronological`，此时才采用500–1400训练、1500–1700验证、1800–2000测试和基线选择。

如需运行旧模式：`--target-mode recorded` 为逐实验留一模式，`--target-mode window` 为合成窗口流量模式。`picture.py` 分别使用 `--model-source recorded` 或 `--model-source window` 查看。

## 训练集与验证集相同的诊断选项

新增 `--validation-on-train`。默认独立验证流程不变。
开启后，训练组1–5也用于验证和检查点选择；原验证组6不使用，测试组7–8仍然独立。
该模式的val_loss是训练样本误差，不是独立验证误差。

```powershell
Set-Location 'NFV/lstm'
& 'python' -X utf8 -B gnt_predict.py --validation-on-train
```

结果单独写到 `GNT/direct_gnt/holdout_train_validation_diagnostic/`。
该目录根目录的图、指标是独立测试7–8组；
`models/holdout/training_fit/` 的图、指标才是训练组1–5的拟合情况，图题明确标为
`Training fit (NOT held out)`，不可当作未见数据上的预测效果。
两类图包含的实验不同，平均真实曲线因此不同；每个实验的真实标签都没有修改。

本次相同100 epochs上限下的实际结果：训练拟合MAE为555.42，独立测试MAE为3003.89。
默认独立验证模式的测试MAE为1576.06。训练拟合变好并未改善泛化，反而出现明显过拟合。
诊断模式确实跑满100 epochs，未依据测试集挑选检查点。
新增3项诊断划分测试通过，连同原测试共27项。

截图中若图例仍为 `Predicted GNT (window model)`，那是旧 `picture.py` 的流量模型图。
新模型由 `gnt_predict.py` 训练，结果由 `gnt_evaluate.py` 重新绘制，图例为 `Direct GNT LSTM`。

这套新增流程只读原始日志，预测原始日志中的五域温度，不经过“预测窗口流量再计算温度”的旧流程。
原 `lstm_predict.py`、`picture.py`、`traffic_generator.py` 和已保存的原始实验结果均保留。

## 任务和真实标签

- 输入过去 3 个完整窗口的域级信息，预测下一个窗口。例如 200、300、400 → 500。
- 每个输入时间步有 20 个特征：5 域的 log1p(F)、H、截断后 T/10000、T 是否有效。
- 输出 5 域的 T/10000，使用 tanh 限制在 [-1,1]，还原后平均得到预测 GNT。
- 真实标签使用 `../result_storage/EO/<run>/resource_usage.yaml`。每条记录先逐域截断到 ±10000，空值按原规则取 0，再对同 time 重复记录平均，最后平均 5 个域。
- 默认目标时间 500–2000，共每组 16 个样本。原始日志后续更晚的数据不加入本次训练。
- 这是滚动一步预测：预测 600 时可以使用已经观测到的 500 状态。不是在 400 时一次性预测到 2000，也不是“当前窗口前35%观测”的任务。
- 不使用目标窗口的 F/H/T 或基于其末态合成的小段作为输入。改变目标窗口状态不会改变该样本输入，已有测试验证。

## 文件

| 文件 | 用途 |
|---|---|
| gnt_dataset.py | 加载权威日志、历史窗口对齐、仅训练组拟合标准化参数、实验划分 |
| gnt_predict.py | 五输出小型LSTM和零预测、末值、历史均值、岭回归基线 |
| gnt_evaluate.py | 重新评估已保存预测、逐域/逐实验指标、单次与平均曲线 |
| test_direct_gnt.py | 新流程的12项回归测试 |

## 运行

先激活已安装项目依赖的 Python 环境。
全部命令在 `NFV/lstm` 运行。输出只能写入 lstm 内部。

```powershell
Set-Location 'NFV/lstm'
& 'python' -X utf8 -B gnt_predict.py
```

默认拟合组 1–5、验证组 6、测试组 7–8。训练组 80 个样本，验证组 16 个样本，测试组 32 个样本。
验证组选择早停检查点，测试组不参与标准化和检查点选择。默认最多100 epochs，patience=15，LSTM隐藏单元16，batch size=16。
损失是五域归一化温度MSE与平均归一化GNT误差平方之和。原温度定义不改变。

输出目录 `GNT/direct_gnt/holdout/`。该目录的平均图只含测试组7、8，所以不同于原来的8组平均图，这是实验集合不同，不是真实值发生了改变。

要生成全部8组的逐次留一验证预测及原8组真实平均曲线：

```powershell
& 'python' -X utf8 -B gnt_predict.py --mode loro
```

输出 `GNT/direct_gnt/loro/`。每折1组测试、另1组验证、其余6组拟合。
8组平均图只作辅助，跨折均值可能抵消误差，不能用曲线贴合程度替代逐实验指标。

仅重新画图和评估保存的预测，不重训：

```powershell
& 'python' -X utf8 -B gnt_evaluate.py --result-dir GNT/direct_gnt/loro
```

不要用旧 `picture.py` 读取这套新输出，它读取的是旧流量模型结果。
两套结果的预测任务不同，不宜将误差差异直接解释为同一任务下的模型改进。

可通过 `--train-runs ... --validation-runs ... --test-runs ...` 在holdout模式下显式设置互不重叠的划分；三者必须覆盖 `--runs`。
可以使用 `--output-dir` 在 lstm 下保存不同配置，默认重复运行会更新同一配置目录。
调整参数应只根据验证集，不要反复查看测试表现后选择配置。

## 输出内容

- `metrics.json`：任务边界、参数、输入文件哈希、实验划分、训练历史、各方法逐次/逐域/逐实验及平均曲线误差。
- `predictions.npz`、`predictions.csv`：原始单位的五域真实值和各方法预测。
- `gnt_<run>.png/.pdf`：每次测试实验的真实与预测曲线。
- `gnt_mean.png/.pdf`：明确标注实验集合的平均曲线。
- `models/<fold>/gnt_model.keras`：该折模型。
- `models/<fold>/preprocessing.npz`：训练组的均值、标准差和岭回归系数。
- `models/<fold>/training.json`：验证损失、选定epoch、训练组信息。

输入特征按 `gnt_dataset.FEATURE_NAMES` 排列。加载保存的模型进行推理时使用
`tf.keras.models.load_model(path, compile=False)`，并使用同折保存的均值、标准差调用
`gnt_dataset.transform()`；模型输出需乘10000。已验证保存再加载能够复现测试预测。

初始零输出也作为一个候选检查点，只用验证集决定是否保留；若训练未带来验证改进，
`selected_initial_zero_checkpoint` 会为 true，而不是强制使用更差的训练结果。

## 本次实际验证结果（2026-09-10）

已完成默认holdout及全部8折loro训练，没有根据测试分数调整超参数。

| 方法 | holdout 7–8组 MAE（32点） | 8折逐次MAE（128点） |
|---|---:|---:|
| 直接GNT LSTM | 1576.06 | 1587.35 |
| 岭回归 | 2136.96 | 1826.79 |
| 零预测 | 1560.09 | 1586.92 |
| 上一窗口温度保持 | 2161.68 | 2369.44 |
| 过去3窗口温度均值 | 1725.58 | 1850.62 |

目前LSTM没有超过零预测基线。留一验证中6折选择了初始零输出检查点，另2折选择第1个epoch，
因此预测接近零不是画图错位，而是验证选择没有支持更复杂的拟合。
这说明本次设定下的历史输入尚未证明能有效预测原始GNT，不证明所有可用特征和方法都无法预测。
本次代码解决的是目标和时序对齐，没有证明精度问题已解决。

24项新旧回归测试通过，128个真实GNT值均与原日志逐点匹配；模型保存/加载和独立重新评估通过。

```powershell
& 'python' -X utf8 -B -m unittest test_direct_gnt test_lstm_pipeline -v
```
## 当前推荐入口

现在直接运行 `lstm_predict.py`，默认预测原始 `resource_usage.yaml` 中的五域 GNT：

```powershell
& python -X utf8 -B ./lstm_predict.py
& python -X utf8 -B ./picture.py
```

结果保存在 `GNT/recorded_lstm`。每次以1组实验作为测试，其余实验提供同一时刻的训练均值；LSTM只学习相对该均值的残差。独立验证误差至少改善2%才采用残差，否则保留同期均值。本批数据中8折都没有达到门槛，因此最终预测采用同期均值。

平均曲线会与8次实验的总体平均曲线重合，这是留一同期均值的代数性质。它适合展示重复实验下的平均时间规律，但不能解释成对单次实验或未来新时间的零误差。应同时报告 `metrics.json` 的 `per_run_time_gnt` 和 `recorded_gnt_run_*.png`。当前逐实验MAE为1647.47。

旧的合成窗口流量实验仍可显式运行：

```powershell
& python -X utf8 -B ./lstm_predict.py --target-mode window
& python -X utf8 -B ./picture.py --model-source window
```
