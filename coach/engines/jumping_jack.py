"""开合跳规则引擎。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import coco17
from .base import ExerciseEngine, Feedback, RepRecord


@dataclass
class JumpingJackParams:
  """开合跳规则阈值集合。"""
  detect_open_hand_ratio: float = 0.55
  detect_open_feet_ratio_sw: float = 0.95
  detect_open_entry_hand_ratio: float = 0.35
  detect_open_entry_feet_ratio_sw: float = 0.70
  detect_close_hand_ratio: float = 0.30
  detect_close_feet_ratio_sw: float = 0.80

  grade_open_hand_perfect_low: float = 0.85
  grade_open_hand_perfect_high: float = 1.35
  grade_open_feet_perfect_min: float = 1.40
  grade_close_hand_perfect_max: float = 0.15
  grade_close_feet_perfect_max: float = 0.40

  grade_open_hand_good_low: float = 0.70
  grade_open_hand_good_high: float = 1.40
  grade_open_feet_good_min: float = 1.05
  grade_close_hand_good_max: float = 0.25
  grade_close_feet_good_max: float = 0.75

  min_conf: float = 0.25


class JumpingJackEngine(ExerciseEngine):
  """开合跳计数与打分引擎。"""

  def __init__(self, params: JumpingJackParams, cooldown_s: float = 0.6):
    """初始化开合跳引擎状态。"""
    super().__init__(cooldown_s=cooldown_s)
    self.p = params
    self.state = 'CLOSED'
    self._cur_start_t: Optional[float] = None
    self._cur_issues: set[str] = set()
    self._open_seen = False
    self._opening_started = False

    self._hand_ratio_max = 0.0
    self._feet_ratio_max = 0.0
    self._hand_ratio_min = 1e9
    self._feet_ratio_min = 1e9
    self._recent_hand: deque[float] = deque(maxlen=5)
    self._recent_feet: deque[float] = deque(maxlen=5)

  def _valid(self, k, idx: int) -> bool:
    """判断指定关键点是否有效。"""
    return k is not None and float(k[idx, 2]) >= self.p.min_conf

  def _shoulder_width(self, k) -> Optional[float]:
    """内部辅助函数。"""
    if not (self._valid(k, coco17.L_SHOULDER) and self._valid(k, coco17.R_SHOULDER)):
      return None
    return float(abs(k[coco17.L_SHOULDER, 0] - k[coco17.R_SHOULDER, 0]))

  def _head_y(self, k) -> Optional[float]:
    """内部辅助函数。"""
    for idx in (coco17.NOSE, coco17.L_EYE, coco17.R_EYE):
      if self._valid(k, idx):
        return float(k[idx, 1])
    return None

  def _hip_mid(self, k) -> Optional[np.ndarray]:
    """内部辅助函数。"""
    if not (self._valid(k, coco17.L_HIP) and self._valid(k, coco17.R_HIP)):
      return None
    return (k[coco17.L_HIP, 0:2] + k[coco17.R_HIP, 0:2]) / 2.0

  def _wrist_min_y(self, k) -> Optional[float]:
    """内部辅助函数。"""
    ys = [float(k[idx, 1]) for idx in (coco17.L_WRIST, coco17.R_WRIST) if self._valid(k, idx)]
    return float(min(ys)) if ys else None

  def _ankle_dist_x(self, k) -> Optional[float]:
    """内部辅助函数。"""
    if not (self._valid(k, coco17.L_ANKLE) and self._valid(k, coco17.R_ANKLE)):
      return None
    return float(abs(k[coco17.L_ANKLE, 0] - k[coco17.R_ANKLE, 0]))

  def _ratios(self, k) -> Optional[tuple[float, float]]:
    """计算归一化比例特征。"""
    sw = self._shoulder_width(k)
    head_y = self._head_y(k)
    hip = self._hip_mid(k)
    wrist_y = self._wrist_min_y(k)
    ankle_dx = self._ankle_dist_x(k)
    if sw is None or head_y is None or hip is None or wrist_y is None or ankle_dx is None:
      return None
    upper_len = float(abs(float(hip[1]) - head_y))
    if sw < 1e-3 or upper_len < 1e-3:
      return None
    hand_ratio = float((float(hip[1]) - wrist_y) / upper_len)
    feet_ratio = float(ankle_dx / sw)
    return hand_ratio, feet_ratio

  def _reset_rep_stats(self, t: float):
    """内部辅助函数。"""
    self._cur_start_t = t
    self._cur_issues = set()
    self._open_seen = False
    self._opening_started = False
    self._hand_ratio_max = 0.0
    self._feet_ratio_max = 0.0
    self._hand_ratio_min = 1e9
    self._feet_ratio_min = 1e9
    self._recent_hand.clear()
    self._recent_feet.clear()

  def _grade(self, open_hand: float, open_feet: float, close_hand: float, close_feet: float) -> str:
    # 对开合跳来说，手举得更高通常不应被惩罚。
    # 因此这里把上界更多看成软限制，只作为安全兜底；
    # 只要超过下界，就仍可以认为打开幅度是合格的。
    """内部辅助函数。"""
    open_perfect = open_hand >= self.p.grade_open_hand_perfect_low and open_feet >= self.p.grade_open_feet_perfect_min
    close_perfect = close_hand <= self.p.grade_close_hand_perfect_max and close_feet <= self.p.grade_close_feet_perfect_max
    if open_perfect and close_perfect:
      return 'perfect'
    open_good = open_hand >= self.p.grade_open_hand_good_low and open_feet >= self.p.grade_open_feet_good_min
    close_good = close_hand <= self.p.grade_close_hand_good_max and close_feet <= self.p.grade_close_feet_good_max
    if open_good and close_good:
      return 'good'
    return 'poor'

  def update(self, kpts, t: float, fps: float) -> Feedback:
    """消费当前帧关键点，推进开合跳状态机并返回反馈。"""
    self.frame_issues = []
    self.rep_completed = False
    self.last_rep = None

    # 1) 先把手部打开幅度、脚部打开幅度转成归一化比例。
    ratios = self._ratios(kpts)
    if ratios is None:
      return Feedback('', '')
    hand_ratio, feet_ratio = ratios
    self._recent_hand.append(float(hand_ratio))
    self._recent_feet.append(float(feet_ratio))
    hand_med = float(np.median(list(self._recent_hand)))
    feet_med = float(np.median(list(self._recent_feet)))

    # 2) 用平滑后的中位数比例，判断是否进入打开/闭合等关键阶段。
    opening_entry = hand_med >= self.p.detect_open_entry_hand_ratio or feet_med >= self.p.detect_open_entry_feet_ratio_sw
    open_cond = hand_med >= self.p.detect_open_hand_ratio or feet_med >= self.p.detect_open_feet_ratio_sw
    close_cond = hand_med <= self.p.detect_close_hand_ratio and feet_med <= self.p.detect_close_feet_ratio_sw

    # 3) 状态机：CLOSED -> OPENING -> OPEN -> CLOSING -> CLOSED。
    if self.state == 'CLOSED':
      if opening_entry:
        self.state = 'OPENING'
        self._reset_rep_stats(t)
        self._opening_started = True
    elif self.state == 'OPENING':
      self._hand_ratio_max = max(self._hand_ratio_max, float(hand_med))
      self._feet_ratio_max = max(self._feet_ratio_max, float(feet_med))
      self._hand_ratio_min = min(self._hand_ratio_min, float(hand_med))
      self._feet_ratio_min = min(self._feet_ratio_min, float(feet_med))
      if open_cond:
        self.state = 'OPEN'
        self._open_seen = True
    elif self.state == 'OPEN':
      self._hand_ratio_max = max(self._hand_ratio_max, float(hand_med))
      self._feet_ratio_max = max(self._feet_ratio_max, float(feet_med))
      self._hand_ratio_min = min(self._hand_ratio_min, float(hand_med))
      self._feet_ratio_min = min(self._feet_ratio_min, float(feet_med))
      if hand_med < self.p.grade_open_hand_good_low:
        self.frame_issues.append('hands_low')
      if feet_med < self.p.grade_open_feet_good_min:
        self.frame_issues.append('feet_narrow')
      if hand_med < self.p.detect_open_entry_hand_ratio and feet_med < self.p.detect_open_entry_feet_ratio_sw:
        self.state = 'CLOSING'
    elif self.state == 'CLOSING':
      self._hand_ratio_max = max(self._hand_ratio_max, float(hand_med))
      self._feet_ratio_max = max(self._feet_ratio_max, float(feet_med))
      self._hand_ratio_min = min(self._hand_ratio_min, float(hand_med))
      self._feet_ratio_min = min(self._feet_ratio_min, float(feet_med))
      if close_cond and self._open_seen:
        close_h = float(self._hand_ratio_min)
        close_f = float(self._feet_ratio_min)
        # rep 级别的问题判定，应基于这次动作全过程里最能代表开/合状态的证据，
        # 而不是仅凭中间某一帧的瞬时状态。
        if float(self._hand_ratio_max) < self.p.grade_open_hand_good_low:
          self._cur_issues.add('hands_low')
        if float(self._feet_ratio_max) < self.p.grade_open_feet_good_min:
          self._cur_issues.add('feet_narrow')
        if close_h > self.p.grade_close_hand_good_max:
          self._cur_issues.add('hands_not_closed')
        if close_f > self.p.grade_close_feet_good_max:
          self._cur_issues.add('feet_not_closed')
        grade = self._grade(
          open_hand=float(self._hand_ratio_max),
          open_feet=float(self._feet_ratio_max),
          close_hand=close_h,
          close_feet=close_f,
        )
        self.state = 'CLOSED'
        self.rep_count += 1
        st = float(self._cur_start_t) if self._cur_start_t is not None else max(0.0, t - 0.5)
        # 4) 一次完整开合完成后，记录最大打开幅度与最小闭合幅度。
        rr = RepRecord(
          rep_idx=self.rep_count,
          start_t=st,
          end_t=float(t),
          issues=sorted(self._cur_issues),
          metrics={
            'open_hand_ratio_max': float(self._hand_ratio_max),
            'open_feet_ratio_max': float(self._feet_ratio_max),
            'close_hand_ratio_min': close_h,
            'close_feet_ratio_min': close_f,
          },
          grade=grade,
        )
        self._reps.append(rr)
        self.rep_completed = True
        self.last_rep = rr
        self._cur_start_t = None
        self._opening_started = False
      elif hand_med > self.p.detect_open_hand_ratio or feet_med > self.p.detect_open_feet_ratio_sw:
        self.state = 'OPEN'

    # 5) 实时提示优先针对“手不够高”“脚打开不够”这类最容易边做边纠正的问题。
    if self.state in ('OPEN', 'CLOSING'):
      if 'hands_low' in self.frame_issues:
        return self._throttled_feedback(t, '手再抬高一点，尽量过头顶', 'hands_low')
      if 'feet_narrow' in self.frame_issues:
        return self._throttled_feedback(t, '脚再打开一点，落地稳一点', 'feet_narrow')

    return Feedback('', '')

  def _motivation_line(self) -> str:
    """内部辅助函数。"""
    if self.rep_count == 0:
      return '先跟上节奏，幅度自然做出来！'
    return '节奏不错，继续保持开合幅度！'
