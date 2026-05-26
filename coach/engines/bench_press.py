"""卧推规则引擎。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import coco17
from ..geometry import angle_deg
from .base import ExerciseEngine, Feedback, RepRecord


@dataclass
class BenchPressParams:
  """卧推规则阈值集合。

  说明：
  1. 当前规则假设机位以侧面或偏侧面为主；
  2. 由于当前工程只使用人体关键点，不直接检测杠铃，因此所有“触胸、锁定、桥起”等判断
     都是通过上肢角度、腕肘相对位置、髋部抬升幅度等人体运动学代理特征来完成；
  3. 这些阈值更适合常规平板卧推的普通训练视频，若机位、身材比例、握距差异较大，
     需要结合实际数据再做微调。
  """

  # ---------- rep 起始 / 结束判定 ----------
  # 顶部准备位：肘关节接近伸直。
  start_ready_elbow_deg: float = 150.0
  # 开始下放：肘角明显变小，说明已经离开顶部锁定区。
  lower_entry_elbow_deg: float = 138.0
  # 底部提示区：当肘角下降到该阈值附近，通常已经进入下胸/胸前的工作区。
  bottom_hint_elbow_deg: float = 100.0
  # 回到顶部完成一次推起时的肘角阈值。
  press_finish_elbow_deg: float = 156.0

  # ---------- 深度 / 锁定质量 ----------
  # “优”对应的底部肘角区间：既要下得足够深，又避免过度塌肩。
  grade_perfect_bottom_low_deg: float = 78.0
  grade_perfect_bottom_high_deg: float = 98.0
  # “良”对应的底部肘角区间。
  grade_good_bottom_low_deg: float = 68.0
  grade_good_bottom_high_deg: float = 110.0
  # “优”对应的顶部锁定下限。
  grade_perfect_lockout_min_deg: float = 166.0
  # “良”对应的顶部锁定下限。
  grade_good_lockout_min_deg: float = 156.0

  # ---------- 轨迹 / 稳定性 ----------
  # 底部腕肘水平偏移比例阈值。
  # 在侧面机位下，理想状态应接近“腕在肘正上方，前臂近似垂直地面”。
  bottom_forearm_vertical_warn_ratio: float = 0.12
  # 髋部抬升（桥起）比例阈值。以躯干长度归一化。
  hip_bridge_warn_ratio: float = 0.09
  # 单次 rep 过快阈值：若过快，通常意味着离心控制不足。
  tempo_fast_warn_s: float = 0.75

  # ---------- 辅助阈值 ----------
  # 腕相对肩的下放比例；可作为“是否明显下到胸前工作区”的辅助判据。
  wrist_drop_bottom_ratio: float = 0.20
  # 顶部回位时，腕相对肩的高度应重新接近顶部基准。
  wrist_drop_return_ratio: float = 0.10
  # 最小关键点置信度。
  min_conf: float = 0.25


class BenchPressEngine(ExerciseEngine):
  """卧推计数与打分引擎。

  核心思路：
  1. 选择当前画面中更清晰的一侧上肢；
  2. 用肘角驱动主状态机，用腕-肩下放比例辅助确认底部工作段；
  3. 用“底部深度、顶部锁定、底部前臂垂直度、髋部桥起、单次节奏”构成质量判断；
  4. 输出 rep 级别的结构化指标，供报告与课后复盘使用。
  """

  def __init__(self, params: BenchPressParams, cooldown_s: float = 0.8):
    """初始化卧推状态。"""
    super().__init__(cooldown_s=cooldown_s)
    self.p = params

    # 状态定义：
    # TOP_READY：顶部准备位
    # LOWERING：下放阶段
    # BOTTOM: 底部工作区
    # PRESSING：向上推起阶段
    self.state = 'TOP_READY'

    self._cur_start_t: Optional[float] = None
    self._cur_issues: set[str] = set()

    # rep 内统计量
    self._elbow_min = 180.0
    self._elbow_top_max = 0.0
    self._bottom_elbow_vals: deque[float] = deque(maxlen=30)
    self._bottom_forearm_offset_ratio_max = 0.0
    self._wrist_drop_ratio_max = 0.0
    self._hip_lift_ratio_max = 0.0

    # 参考位：在 rep 开始时记录，后续用于判断髋部是否桥起。
    self._hip_ref_y: Optional[float] = None

    self._prev_ea: Optional[float] = None
    self._prev_drop: Optional[float] = None
    self._bottom_seen = False

  def _valid(self, k, idx: int) -> bool:
    """判断指定关键点是否有效。"""
    return k is not None and float(k[idx, 2]) >= self.p.min_conf

  def _choose_arm(self, k):
    """选择当前更清晰的一侧手臂。

    侧面机位下，常常只有一侧上肢更完整可见，因此这里优先选择可见点更多的一侧。
    """
    left_ok = sum([self._valid(k, coco17.L_SHOULDER), self._valid(k, coco17.L_ELBOW), self._valid(k, coco17.L_WRIST), self._valid(k, coco17.L_HIP)])
    right_ok = sum([self._valid(k, coco17.R_SHOULDER), self._valid(k, coco17.R_ELBOW), self._valid(k, coco17.R_WRIST), self._valid(k, coco17.R_HIP)])
    if right_ok > left_ok:
      return coco17.R_SHOULDER, coco17.R_ELBOW, coco17.R_WRIST, coco17.R_HIP
    return coco17.L_SHOULDER, coco17.L_ELBOW, coco17.L_WRIST, coco17.L_HIP

  def _torso_len(self, k) -> Optional[float]:
    """估算当前侧躯干长度，用作尺度归一化。"""
    s, _e, _w, h = self._choose_arm(k)
    if not (self._valid(k, s) and self._valid(k, h)):
      return None
    return float(max(1.0, np.linalg.norm(k[s, 0:2] - k[h, 0:2])))

  def _elbow_angle(self, k) -> Optional[float]:
    """计算当前主要工作侧的肘角。"""
    s, e, w, _h = self._choose_arm(k)
    if not (self._valid(k, s) and self._valid(k, e) and self._valid(k, w)):
      return None
    ang = angle_deg(k[s, 0:2], k[e, 0:2], k[w, 0:2])
    if not np.isfinite(ang):
      return None
    return float(ang)

  def _wrist_drop_ratio(self, k) -> Optional[float]:
    """计算腕相对肩的下放比例。

    在侧面卧推里，顶部时腕与肩高度接近；随着下放，腕部会逐渐低于肩部。
    该指标越大，通常表示越接近胸前底部工作段。
    """
    s, _e, w, _h = self._choose_arm(k)
    torso_len = self._torso_len(k)
    if torso_len is None or not (self._valid(k, s) and self._valid(k, w)):
      return None
    return max(0.0, float((k[w, 1] - k[s, 1]) / torso_len))

  def _bottom_forearm_offset_ratio(self, k) -> Optional[float]:
    """计算腕肘水平偏移比例。

    侧面卧推底部的一个关键技术点是“腕尽量叠在肘上方，前臂近似垂直地面”。
    若腕肘水平偏差太大，通常说明下放位置偏前/偏后，受力线路不够理想。
    """
    _s, e, w, _h = self._choose_arm(k)
    torso_len = self._torso_len(k)
    if torso_len is None or not (self._valid(k, e) and self._valid(k, w)):
      return None
    return float(abs(float(k[w, 0]) - float(k[e, 0])) / torso_len)

  def _hip_lift_ratio(self, k) -> Optional[float]:
    """计算髋部上抬（桥起）比例。"""
    _s, _e, _w, h = self._choose_arm(k)
    torso_len = self._torso_len(k)
    if torso_len is None or not self._valid(k, h):
      return None
    hip_y = float(k[h, 1])
    if self._hip_ref_y is None:
      self._hip_ref_y = hip_y
    # 图像坐标系里 y 越小表示越向上，因此 ref_y - hip_y > 0 代表髋部抬起。
    return max(0.0, float((self._hip_ref_y - hip_y) / torso_len))

  def _reset_rep_stats(self, t: float, k):
    """在一次新 rep 开始时重置统计量。"""
    self._cur_start_t = t
    self._cur_issues = set()
    self._elbow_min = 180.0
    self._elbow_top_max = 0.0
    self._bottom_elbow_vals.clear()
    self._bottom_forearm_offset_ratio_max = 0.0
    self._wrist_drop_ratio_max = 0.0
    self._hip_lift_ratio_max = 0.0
    _s, _e, _w, h = self._choose_arm(k)
    self._hip_ref_y = float(k[h, 1]) if self._valid(k, h) else None
    self._prev_ea = self._elbow_angle(k)
    self._prev_drop = self._wrist_drop_ratio(k)
    self._bottom_seen = False

  def _grade(self, bottom_elbow_deg: float, top_lockout_deg: float, issues: set[str]) -> str:
    """根据卧推底部深度、顶部锁定和主要技术问题给出等级。"""
    if bottom_elbow_deg < self.p.grade_perfect_bottom_low_deg or bottom_elbow_deg > self.p.grade_perfect_bottom_high_deg:
      perfect_by_depth = False
    else:
      perfect_by_depth = True
    good_by_depth = self.p.grade_good_bottom_low_deg <= bottom_elbow_deg <= self.p.grade_good_bottom_high_deg

    perfect_by_lockout = top_lockout_deg >= self.p.grade_perfect_lockout_min_deg
    good_by_lockout = top_lockout_deg >= self.p.grade_good_lockout_min_deg

    major_issue = any(it in issues for it in ('hip_bridge', 'forearm_not_vertical'))

    if perfect_by_depth and perfect_by_lockout and not major_issue:
      return 'perfect'
    if good_by_depth and good_by_lockout:
      return 'good'
    return 'poor'

  def update(self, kpts, t: float, fps: float) -> Feedback:
    """消费当前帧关键点，推进卧推状态机并返回反馈。"""
    self.frame_issues = []
    self.rep_completed = False
    self.last_rep = None

    # ---------- 1) 提取当前帧关键特征 ----------
    ea = self._elbow_angle(kpts)
    if ea is None:
      return Feedback('', '')
    wrist_drop = self._wrist_drop_ratio(kpts)
    forearm_offset = self._bottom_forearm_offset_ratio(kpts)
    hip_lift = self._hip_lift_ratio(kpts)

    moving_down = self._prev_ea is not None and float(ea) < float(self._prev_ea) - 0.8
    moving_up = self._prev_ea is not None and float(ea) > float(self._prev_ea) + 0.8

    # ---------- 2) 只要已经离开顶部准备位，就持续累积 rep 级统计量 ----------
    if self.state != 'TOP_READY':
      self._elbow_min = min(self._elbow_min, float(ea))
      self._elbow_top_max = max(self._elbow_top_max, float(ea))
      if wrist_drop is not None:
        self._wrist_drop_ratio_max = max(self._wrist_drop_ratio_max, float(wrist_drop))
      if hip_lift is not None:
        self._hip_lift_ratio_max = max(self._hip_lift_ratio_max, float(hip_lift))

      # 底部窗口统计：只在明显进入下胸工作区后，才把这些帧收进底部角度窗口。
      if (wrist_drop is not None and wrist_drop >= self.p.wrist_drop_bottom_ratio) or float(ea) <= self.p.bottom_hint_elbow_deg:
        self._bottom_elbow_vals.append(float(ea))
        self._bottom_seen = True
        if forearm_offset is not None:
          self._bottom_forearm_offset_ratio_max = max(self._bottom_forearm_offset_ratio_max, float(forearm_offset))

    # ---------- 3) 卧推状态机 ----------
    if self.state == 'TOP_READY':
      # 从顶部进入下放：肘角开始明显减小，或腕部开始明显下沉。
      if (float(ea) <= self.p.lower_entry_elbow_deg and moving_down) or (wrist_drop is not None and wrist_drop >= self.p.wrist_drop_return_ratio and moving_down):
        self.state = 'LOWERING'
        self._reset_rep_stats(t, kpts)
        self._elbow_min = float(ea)
        self._elbow_top_max = float(ea)
    elif self.state == 'LOWERING':
      if self._bottom_seen:
        self.state = 'BOTTOM'
    elif self.state == 'BOTTOM':
      # 一旦确认到底部，再检测是否开始向上推起。
      if moving_up:
        self.state = 'PRESSING'
    elif self.state == 'PRESSING':
      # 底部技术问题主要在推起前后判定最稳定，因此在 PRESSING 阶段集中给出。
      if self._elbow_min > self.p.grade_good_bottom_high_deg:
        self._cur_issues.add('rom_insufficient')
        self.frame_issues.append('rom_insufficient')
      if self._bottom_forearm_offset_ratio_max > self.p.bottom_forearm_vertical_warn_ratio:
        self._cur_issues.add('forearm_not_vertical')
        self.frame_issues.append('forearm_not_vertical')
      if self._hip_lift_ratio_max > self.p.hip_bridge_warn_ratio:
        self._cur_issues.add('hip_bridge')
        self.frame_issues.append('hip_bridge')

      # 回到顶部并基本重新锁定，认为一次完整 rep 结束。
      ready_by_elbow = float(ea) >= self.p.press_finish_elbow_deg
      ready_by_drop = wrist_drop is not None and wrist_drop <= self.p.wrist_drop_return_ratio
      if self._bottom_seen and ready_by_elbow and ready_by_drop:
        self.state = 'TOP_READY'
        self.rep_count += 1
        st = float(self._cur_start_t) if self._cur_start_t is not None else max(0.0, t - 0.6)
        dur = float(max(0.0, t - st))
        if dur < self.p.tempo_fast_warn_s:
          self._cur_issues.add('tempo_too_fast')
        if self._elbow_top_max < self.p.grade_good_lockout_min_deg:
          self._cur_issues.add('lockout_incomplete')

        bottom_elbow_med = float(np.median(list(self._bottom_elbow_vals))) if self._bottom_elbow_vals else float(self._elbow_min)
        grade = self._grade(bottom_elbow_med, float(self._elbow_top_max), self._cur_issues)

        rr = RepRecord(
          rep_idx=self.rep_count,
          start_t=st,
          end_t=float(t),
          issues=sorted(self._cur_issues),
          metrics={
            'elbow_bottom_med_deg': bottom_elbow_med,
            'elbow_min_deg': float(self._elbow_min),
            'elbow_top_max_deg': float(self._elbow_top_max),
            'wrist_drop_ratio_max': float(self._wrist_drop_ratio_max),
            'bottom_forearm_offset_ratio_max': float(self._bottom_forearm_offset_ratio_max),
            'hip_lift_ratio_max': float(self._hip_lift_ratio_max),
            'tempo_s': dur,
          },
          grade=grade,
        )
        self._reps.append(rr)
        self.rep_completed = True
        self.last_rep = rr
        self._cur_start_t = None
        self._bottom_seen = False

    # ---------- 4) 实时提示优先处理最关键、最可即时修正的问题 ----------
    if self.state in ('LOWERING', 'BOTTOM', 'PRESSING'):
      if 'hip_bridge' in self.frame_issues:
        return self._throttled_feedback(t, '臀部别顶桥，肩臀保持稳定', 'hip_bridge')
      if 'forearm_not_vertical' in self.frame_issues:
        return self._throttled_feedback(t, '底部让腕叠在肘上方，前臂尽量竖直', 'forearm_not_vertical')
      if 'rom_insufficient' in self.frame_issues:
        return self._throttled_feedback(t, '再下到下胸附近，再稳稳推起', 'rom_insufficient')

    self._prev_ea = float(ea)
    self._prev_drop = float(wrist_drop) if wrist_drop is not None else self._prev_drop
    return Feedback('', '')

  def _motivation_line(self) -> str:
    """给卧推提供更贴近动作的鼓励文案。"""
    if self.rep_count == 0:
      return '先把下放控制住，再把发力做扎实。'
    return '推得不错，继续保持下放控制和顶部锁定！'
