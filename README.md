# Fitness Video Coach Demo

这是一个用于运动健身动作识别与复盘分析的完整工程。

工程包含两条主要链路：

1. **实时链路**：YOLO Pose + 规则引擎 + 教练提示策略。
2. **课后链路**：可选的 VLM 复盘总结。

当前支持动作：

- `squat` 深蹲
- `jumping_jack` 开合跳
- `curl` 杠铃弯举
- `pushup` 俯卧撑
- `bench_press` 卧推
- `crunch` 卷腹

其中：

- 实时链路默认采用规则方案，不启用实时 VLM。
- 课后 VLM 默认读取 YOLO 结构化结果作为主要输入。
- 用户表达风格默认使用 `simple`。

---

## 1. 安装

```bash
pip install -r requirements.txt
```

---

## 2. YOLO 权重

工程会自动创建 `weights/` 目录。

- Windows 默认会尝试从：`D:\16_运动指导\data\weights\YOLO` 拷贝权重（例如 `yolov8n-pose.pt`）到 `weights/`。
- 也可以通过环境变量覆盖权重目录：

```powershell
$env:YOLO_WEIGHTS_SRC_DIR="D:\path\to\weights"
```

---

## 3. 配置文件

默认配置文件：`config.default.json`

如果希望查看带解释的版本，可以阅读：

- `config.default.explained.jsonc`
- `PARAMETER_GUIDE.md`

重点配置包括：

- `global`：全局运行参数，如关键点置信度、平滑系数、是否按原视频速度播放等。
- `coach`：实时提示策略，如提示模式、分组大小、提示停留时间。
- `squat / jumping_jack / curl / pushup / bench_press / crunch`：各动作的规则阈值。
- `vlm`：课后 VLM 开关、接口配置和输入选择策略。

---

## 4. 运行方式

### 4.1 实时规则链路

```bash
python run_demo.py --video "YOUR_VIDEO.mp4" --exercise squat --goal 20 --show --save_video
```

### 4.2 启用课后 VLM 复盘

先在 `config.default.json` 中补充：

```json
"vlm": {
 "posthoc": true,
 "base_url": "https://your-openai-compatible-endpoint/v1",
 "api_key": "YOUR_KEY",
 "model_name": "YOUR_VLM_MODEL"
}
```

然后运行：

```bash
python run_demo.py --video "YOUR_VIDEO.mp4" --exercise squat --show --save_video
```

如果不想改默认文件，也可以复制一份自定义配置，例如 `config.local.json`，然后：

```bash
python run_demo.py --config config.local.json --video "YOUR_VIDEO.mp4" --exercise squat --show
```

### 4.3 命令行临时覆盖课后 VLM 开关

```bash
python run_demo.py --config config.local.json --video "YOUR_VIDEO.mp4" --exercise squat --vlm_posthoc
```

或

```bash
python run_demo.py --config config.local.json --video "YOUR_VIDEO.mp4" --exercise squat --no_vlm_posthoc
```

---

## 5. 输出结果

输出目录：`outputs/<video_name>_<exercise>_<config_name>_<timestamp>/`

典型输出包括：

- `*_out.mp4`：带骨架与提示的视频（开启 `--save_video` 时）
- `*_report.md`：课后复盘 markdown
- `*_report.json`：课后复盘结构化结果

说明：

- 当 `vlm.posthoc=false` 时，报告主要来自规则引擎统计。
- 当 `vlm.posthoc=true` 时，报告会额外写入 VLM 生成的复盘内容。

---

## 6. 代码与配置阅读顺序

建议按照下面顺序阅读：

1. `README.md`
2. `config.default.explained.jsonc`
3. `PARAMETER_GUIDE.md`
4. `run_demo.py`
5. `coach/demo.py`
6. `coach/engines/*.py`
7. `coach/vlm_inputs.py`
8. `coach/report.py`

---

## 7. 目录结构

```text
coach/
 demo.py        # 主流程：视频读取、YOLO 推理、规则引擎、可选 posthoc VLM
 yolo_pose.py     # YOLO Pose 前端
 vlm_api.py      # OpenAI-compatible VLM 调用
 vlm_inputs.py     # VLM 输入构造
 coach_policy.py    # 实时提示策略
 report.py       # 规则统计与报告生成
 engines/       # 六个动作的规则引擎
run_demo.py       # 统一启动入口
config.default.json   # 默认配置
README.md        # 使用说明
```

---

## 8. 单独测试 VLM 连通性

```bash
python tools/test_vlm.py --config config.default.json
```

---

## 9. 跑 Benchmark

如果想稳定复跑部署/时延/资源占用测试，可以使用：

```bash
python benchmark.py --video "YOUR_VIDEO.mp4" --exercise squat
```

常见用法：

```bash
python benchmark.py --video "YOUR_VIDEO.mp4" --exercise squat --runs 3 --warmup-runs 1
```

如果只想测实时链路，保持默认 `--posthoc-vlm off` 即可。

如果想测课后链路，但先不接真实 VLM，可以用本地 mock：

```bash
python benchmark.py --video "YOUR_VIDEO.mp4" --exercise squat --posthoc-vlm mock --mock-vlm-delay-s 1.0
```

如果想测真实课后 VLM：

```bash
python benchmark.py --video "YOUR_VIDEO.mp4" --exercise squat --posthoc-vlm real --config config.local.json
```

benchmark 会输出一个独立 JSON，总结：

- 每次 run 的端到端耗时、逐帧耗时、吞吐
- 进程 CPU / 内存采样
- `run_demo` 报告里的实时链路 / 课后链路性能字段
- 课后 VLM 的调用次数、时延和 token 统计（如果启用）

默认输出到：`outputs/benchmarks/`
