# 参数说明

这份文档用于集中说明工程里的主要参数及其作用。

## 一、整体说明

当前工程包含两条主要链路：

1. 实时链路：YOLO Pose + 规则判断 + 提示策略。
2. 课后链路：可选的 VLM 复盘总结。

默认行为如下：

1. 实时链路使用规则方案，不启用实时 VLM。
2. 课后链路可以启用 VLM，且主要输入为 YOLO 结构化结果。
3. 默认提示风格使用 `simple`。

---

## 二、配置文件结构

默认配置文件为 `config.default.json`，主要分成以下几块：

- `global`：全局参数
- `coach`：提示策略参数
- `squat`：深蹲规则参数
- `jumping_jack`：开合跳规则参数
- `curl`：弯举规则参数
- `pushup`：俯卧撑规则参数
- `bench_press`：卧推规则参数
- `crunch`：卷腹规则参数
- `vlm`：课后 VLM 参数

---

## 三、最常用的参数

### 1. `global.min_kpt_conf`

含义：YOLO 关键点最小置信度。

作用：
- 置信度低于这个值的关键点会被视为不可靠。
- 值太低，容易引入抖动和误判。
- 值太高，容易丢掉本来可用的关键点。

### 2. `global.smooth_alpha`

含义：关键点平滑系数。

作用：
- 用于减少关键点随帧抖动。
- 值越小，越平滑，但响应越慢。
- 值越大，越灵敏，但画面更容易跳动。

### 3. `coach.mode`

含义：提示触发模式。

常见取值：
- `event`：按事件触发
- `group`：按动作组触发
- `time`：按固定时间间隔触发

### 4. `coach.group_n`

含义：每多少次动作触发一次组级反馈。

例如：
- `1` 表示每做完 1 次动作就可以给一次反馈。
- `3` 表示每做完 3 次动作再汇总给一次反馈。

### 5. `coach.tip_hold_s`

含义：一条提示在画面上保留多久。

作用：
- 值太短，用户来不及看清。
- 值太长，容易遮挡画面。

### 6. `vlm.posthoc`

含义：是否启用课后 VLM 总结。

说明：
- `false`：只输出规则统计结果。
- `true`：在规则统计结果基础上，再调用 VLM 生成总结内容。

### 7. `vlm.base_url / vlm.api_key / vlm.model_name`

含义：VLM 服务地址、鉴权密钥和模型名。

用途：
- 当启用 `vlm.posthoc=true` 时，这三项必须正确配置。

### 8. `vlm.inputs.posthoc.yolo_output`

含义：是否把 YOLO 结构化结果作为课后 VLM 的输入。

说明：
- 这是课后 VLM 的主要输入来源。
- 如果关闭这一项，VLM 可用的信息会明显变少。

### 9. `vlm.inputs.posthoc.raw_frames`

含义：是否把原始帧图像直接送给 VLM。

说明：
- 打开后，VLM 可以直接看图。
- 关闭后，VLM 主要依赖结构化结果进行总结。

---

## 四、如果要调识别效果，通常先看哪里

### 1. 深蹲不容易计数
优先看：
- `squat.knee_start_max_deg`
- `squat.knee_bottom_hint_deg`
- `squat.knee_top_ready_deg`
- `squat.hip_drop_start_ratio`
- `squat.hip_drop_bottom_ratio`

### 2. 开合跳开合判断不稳定
优先看：
- `jumping_jack.detect_open_hand_ratio`
- `jumping_jack.detect_open_feet_ratio_sw`
- `jumping_jack.detect_close_hand_ratio`
- `jumping_jack.detect_close_feet_ratio_sw`

### 3. 弯举容易误判借力
优先看：
- `curl.elbow_drift_warn_ratio`
- `curl.trunk_swing_warn_deg`
- `curl.grade_perfect_max_elbow_deg`
- `curl.grade_good_max_elbow_deg`

### 4. 关键点抖动太明显
优先看：
- `global.min_kpt_conf`
- `global.smooth_alpha`

---

## 五、建议的调参顺序

建议流程：

1. 先确认视频视角、人物大小和光照是否合适。
2. 再调 `global.min_kpt_conf` 和 `global.smooth_alpha`，保证关键点尽量稳定。
3. 然后针对具体动作调对应规则阈值。
4. 最后再微调 `coach` 相关参数，让提示节奏更合适。


### 5. 卧推底部轨迹和锁定判断
优先看：
- `bench_press.lower_entry_elbow_deg`
- `bench_press.bottom_hint_elbow_deg`
- `bench_press.press_finish_elbow_deg`
- `bench_press.bottom_forearm_vertical_warn_ratio`
- `bench_press.hip_bridge_warn_ratio`

### 6. 卷腹卷起高度和代偿判断
优先看：
- `crunch.curl_entry_torso_deg`
- `crunch.top_torso_deg`
- `crunch.return_ready_torso_deg`
- `crunch.hip_drift_warn_ratio`
- `crunch.knee_drift_warn_ratio`
- `crunch.momentum_warn_deg_s`
