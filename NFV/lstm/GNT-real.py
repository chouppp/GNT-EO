import yaml
import numpy as np
import matplotlib.pyplot as plt


# Edit parameters here.
YAML_FILE = "resource_usage.yaml"
OUTPUT_FIGURE = "average_network_temperature_0_1500.png"

START_TIME = 500
END_TIME = 2000
X_TICK_INTERVAL = 100

# Clip the final mean to [-10000, 10000] so extreme GNT values do not obscure the curve.
# Set to None to plot the unclipped raw mean GNT.
CLIP_LIMIT = 10000.0
# =============================================================


with open(YAML_FILE, "r", encoding="utf-8") as file:
    data = yaml.safe_load(file) or {}

records = data.get("records", [])

times = []
raw_average_gnt = []
plot_average_gnt = []

for record in records:
    time_value = float(record.get("time", 0))

    if not (START_TIME < time_value <= END_TIME):
        continue

    domain_temperatures = record.get("domain_temperatures", [])

    # Always average over five domains; treat null T or missing domains as zero.
    temperature_by_domain = {
        int(item["domain_id"]): (
            0.0 if item.get("T") is None else float(item["T"])
        )
        for item in domain_temperatures
    }
    temperatures = [
        temperature_by_domain.get(domain_id, 0.0)
        for domain_id in range(5)
    ]

    average_temperature = float(np.mean(temperatures))

    if CLIP_LIMIT is None:
        plotted_temperature = average_temperature
    else:
        plotted_temperature = float(
            np.clip(
                average_temperature,
                -CLIP_LIMIT,
                CLIP_LIMIT,
            )
        )

    times.append(time_value)
    raw_average_gnt.append(average_temperature)
    plot_average_gnt.append(plotted_temperature)


if not times:
    raise ValueError(
        f"在时间范围({START_TIME}, {END_TIME}]内没有找到温度记录。"
    )


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["font.size"] = 12

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    times,
    plot_average_gnt,
    color="#1f77b4",
    linewidth=1.8,
    marker="o",
    markersize=4.5,
    markerfacecolor="white",
    markeredgewidth=1.2,
    label="Average GNT",
)

ax.axhline(
    y=0,
    color="gray",
    linewidth=0.8,
    linestyle="--",
    alpha=0.7,
)

ax.set_xlabel("Time")
ax.set_ylabel("Average GNT")
ax.set_xlim(START_TIME, END_TIME)
ax.set_xticks(
    np.arange(
        START_TIME,
        END_TIME + 1,
        X_TICK_INTERVAL,
    )
)

if CLIP_LIMIT is not None:
    ax.set_ylim(-CLIP_LIMIT * 1.05, CLIP_LIMIT * 1.05)

ax.grid(
    True,
    which="major",
    linestyle="--",
    linewidth=0.6,
    alpha=0.45,
)
ax.legend(frameon=False, loc="best")

fig.tight_layout()
fig.savefig(
    OUTPUT_FIGURE,
    dpi=600,
    bbox_inches="tight",
)
plt.close(fig)


print(f"共读取{len(times)}个时间点。")
print(f"时间范围：{times[0]:g}--{times[-1]:g}")
print(f"原始平均GNT最小值：{min(raw_average_gnt):.6f}")
print(f"原始平均GNT最大值：{max(raw_average_gnt):.6f}")
if CLIP_LIMIT is not None:
    clipped_count = sum(
        abs(value) > CLIP_LIMIT for value in raw_average_gnt
    )
    print(f"图中被限制到±{CLIP_LIMIT:g}的点数：{clipped_count}")
print(f"图片已保存：{OUTPUT_FIGURE}")