"""深蹲规则引擎。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import coco17
from ..geometry import angle_deg
from .base import ExerciseEngine, Feedback, RepRecord


@dataclass
class SquatParams:
  """深蹲规则阈值集合。"""
  # 计数阈值相对宽松：计数任务优先避免漏掉本来有效的 rep。
  knee_start_max_deg: float = 155.0
  knee_bottom_hint_deg: float = 128.0
  knee_top_ready_deg: float = 148.0
  hip_drop_start_ratio: float = 0.10
  hip_drop_bottom_ratio: float = 0.20
  hip_drop_return_ratio: float = 0.08

  torso_lean_warn_deg: float = 28.0

  # 打分窗口使用“底部阶段的中位数”，而不是某一帧的极端值。
  grade_perfect_low_deg: float = 82.0
  grade_perfect_high_deg: float = 102.0
  grade_good_low_deg: float = 72.0
  grade_good_high_deg: float = 112.0

  bottom_window_deg: float = 135.0
  min_conf: float = 0.25


class SquatEngine(ExerciseEngine):
  """深蹲计数与打分引擎。"""

  def __init__(self, params: SquatParams, cooldown_s: float = 0.9):
    """初始化深蹲引擎状态。"""
    super().__init__(cooldown_s=cooldown_s)
    self.p = params
    self.state = 'TOP_READY'
    self._cur_start_t: Optional[float] = None
    self._cur_issues: set[str] = set()

    self._prev_knee: Optional[float] = None
    self._prev_drop: Optional[float] = None
    self._top_hip_y_ref: Optional[float] = None

    self._bottom_seen = False
    self._knee_bottom_vals: deque[float] = deque(maxlen=25)
    self._trunk_bottom_vals: deque[float] = deque(maxlen=25)
    self._knee_min = 180.0
    self._drop_max = 0.0
    self._trunk_lean_max = 0.0

  def _valid(self, k, idx: int) -> bool:
    """判断指定关键点是否有效。"""
    return k is not None and float(k[idx, 2]) >= self.p.min_conf

  def _mid(self, k, a: int, b: int) -> Optional[np.ndarray]:
    """内部辅助函数。"""
    if not (self._valid(k, a) and self._valid(k, b)):
      return None
    return (k[a, 0:2] + k[b, 0:2]) / 2.0

  def _hip_mid(self, k) -> Optional[np.ndarray]:
    """内部辅助函数。"""
    return self._mid(k, coco17.L_HIP, coco17.R_HIP)

  def _shoulder_mid(self, k) -> Optional[np.ndarray]:
    """内部辅助函数。"""
    return self._mid(k, coco17.L_SHOULDER, coco17.R_SHOULDER)

  def _torso_len(self, k) -> Optional[float]:
    """内部辅助函数。"""
    sh = self._shoulder_mid(k)
    hp = self._hip_mid(k)
    if sh is None or hp is None:
      return None
    return float(max(1.0, np.linalg.norm(sh - hp)))

  def _knee_angle(self, k) -> Optional[float]:
    """计算当前姿态对应的角度特征。"""
    vals = []
    if self._valid(k, coco17.L_HIP) and self._valid(k, coco17.L_KNEE) and self._valid(k, coco17.L_ANKLE):
      vals.append(angle_deg(k[coco17.L_HIP, 0:2], k[coco17.L_KNEE, 0:2], k[coco17.L_ANKLE, 0:2]))
    if self._valid(k, coco17.R_HIP) and self._valid(k, coco17.R_KNEE) and self._valid(k, coco17.R_ANKLE):
      vals.append(angle_deg(k[coco17.R_HIP, 0:2], k[coco17.R_KNEE, 0:2], k[coco17.R_ANKLE, 0:2]))
    vals = [float(v) for v in vals if np.isfinite(v)]
    return float(min(vals)) if vals else None

  def _trunk_lean(self, k) -> Optional[float]:
    """内部辅助函数。"""
    sh = self._shoulder_mid(k)
    hp = self._hip_mid(k)
    if sh is None or hp is None:
      return None
    v = sh - hp
    nv = float(np.linalg.norm(v))
    if nv < 1e-6:
      return None
    vert = np.array([0.0, -1.0], dtype=np.float32)
    cosang = float(np.clip(np.dot(v / nv, vert), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))

  def _hip_drop_ratio(self, k) -> Optional[float]:
    """计算归一化比例特征。"""
    hp = self._hip_mid(k)
    torso_len = self._torso_len(k)
    if hp is None or torso_len is None:
      return None
    hip_y = float(hp[1])
    if self._top_hip_y_ref is None:
      self._top_hip_y_ref = hip_y
    return max(0.0, float((hip_y - self._top_hip_y_ref) / torso_len))

  def _reset_rep_stats(self, t: float):
    """内部辅助函数。"""
    self._cur_start_t = t
    self._cur_issues = set()
    self._bottom_seen = False
    self._knee_bottom_vals.clear()
    self._trunk_bottom_vals.clear()
    self._knee_min = 180.0
    self._drop_max = 0.0
    self._trunk_lean_max = 0.0

  def _grade_from_bottom(self, knee_bottom_med: float) -> str:
    """内部辅助函数。"""
    if self.p.grade_perfect_low_deg <= knee_bottom_med <= self.p.grade_perfect_high_deg:
      return 'perfect'
    if self.p.grade_good_low_deg <= knee_bottom_med <= self.p.grade_good_high_deg:
      return 'good'
    return 'poor'

  def update(self, kpts, t: float, fps: float) -> Feedback:
    """消费当前帧关键点，推进深蹲状态机并返回反馈。"""
    self.frame_issues = []
    self.rep_completed = False
    self.last_rep = None

    # 1) 深蹲核心特征：膝角、髋部下降比例、躯干前倾角。
    knee = self._knee_angle(kpts)
    drop = self._hip_drop_ratio(kpts)
    if knee is None or drop is None:
      return Feedback('', '')
    trunk = self._trunk_lean(kpts)

    # 当用户明显处于站直状态时，持续刷新顶部参考位。
    if knee > max(self.p.knee_top_ready_deg, self.p.knee_start_max_deg) and drop < self.p.hip_drop_return_ratio * 1.5:
      hp = self._hip_mid(kpts)
      if hp is not None:
        hip_y = float(hp[1])
        if self._top_hip_y_ref is None:
          self._top_hip_y_ref = hip_y
        else:
          self._top_hip_y_ref = 0.9 * float(self._top_hip_y_ref) + 0.1 * hip_y

    # 在一次类似 rep 的运动过程中，持续更新运行时统计量。
    if self.state != 'TOP_READY':
      self._knee_min = min(self._knee_min, float(knee))
      self._drop_max = max(self._drop_max, float(drop))
      if trunk is not None:
        self._trunk_lean_max = max(self._trunk_lean_max, float(trunk))
      if drop >= self.p.hip_drop_bottom_ratio or knee <= self.p.bottom_window_deg:
        self._knee_bottom_vals.append(float(knee))
        if trunk is not None:
          self._trunk_bottom_vals.append(float(trunk))

    if self.state == 'TOP_READY':
      moving_down = self._prev_drop is not None and drop > float(self._prev_drop) + 0.01
      if (drop >= self.p.hip_drop_start_ratio and knee <= self.p.knee_start_max_deg) or (moving_down and knee <= self.p.knee_start_max_deg - 5.0):
        self.state = 'DESCENDING'
        self._reset_rep_stats(t)
        self._knee_min = float(knee)
        self._drop_max = float(drop)
        if trunk is not None:
          self._trunk_lean_max = float(trunk)
    elif self.state == 'DESCENDING':
      if drop >= self.p.hip_drop_bottom_ratio or knee <= self.p.knee_bottom_hint_deg:
        self._bottom_seen = True
        self.state = 'BOTTOM_REACHED'
    elif self.state == 'BOTTOM_REACHED':
      if self._prev_knee is not None and float(knee) > float(self._prev_knee) + 1.0:
        self.state = 'ASCENDING'
    elif self.state == 'ASCENDING':
      if self._bottom_seen and drop <= self.p.hip_drop_return_ratio and knee >= self.p.knee_top_ready_deg:
        self.state = 'TOP_READY'
        self.rep_count += 1
        st = float(self._cur_start_t) if self._cur_start_t is not None else max(0.0, t - 0.6)
        knee_bottom_med = float(np.median(list(self._knee_bottom_vals))) if self._knee_bottom_vals else float(self._knee_min)
        trunk_bottom_med = float(np.median(list(self._trunk_bottom_vals))) if self._trunk_bottom_vals else float(self._trunk_lean_max)

        # 问题标签与总等级独立计算，不完全由等级直接决定。
        if knee_bottom_med > self.p.grade_good_high_deg:
          self._cur_issues.add('depth_insufficient')
        elif knee_bottom_med < self.p.grade_good_low_deg:
          self._cur_issues.add('depth_too_deep')
        if trunk_bottom_med > self.p.torso_lean_warn_deg:
          self._cur_issues.add('torso_lean')

        # 3) 当一次完整深蹲返回顶部后，写入 RepRecord。
        rr = RepRecord(
          rep_idx=self.rep_count,
          start_t=st,
          end_t=float(t),
          issues=sorted(self._cur_issues),
          metrics={
            'knee_bottom_med_deg': knee_bottom_med,
            'knee_min_deg': float(self._knee_min),
            'hip_drop_max_ratio': float(self._drop_max),
            'trunk_lean_bottom_med_deg': trunk_bottom_med,
            'trunk_lean_max_deg': float(self._trunk_lean_max),
          },
          grade=self._grade_from_bottom(knee_bottom_med),
        )
        self._reps.append(rr)
        self.rep_completed = True
        self.last_rep = rr
        self._cur_start_t = None
        self._bottom_seen = False

    # 只有当用户明显处于动作工作区间时，才给逐帧提示，避免噪声干扰。
    if self.state in ('DESCENDING', 'BOTTOM_REACHED', 'ASCENDING'):
      if drop >= self.p.hip_drop_bottom_ratio:
        knee_ref = float(np.median(list(self._knee_bottom_vals))) if self._knee_bottom_vals else float(knee)
        if knee_ref > self.p.grade_good_high_deg:
          self.frame_issues.append('depth_insufficient')
        elif knee_ref < self.p.grade_good_low_deg:
          self.frame_issues.append('depth_too_deep')
      if trunk is not None and float(trunk) > self.p.torso_lean_warn_deg and drop >= self.p.hip_drop_start_ratio:
        self.frame_issues.append('torso_lean')

    self._prev_knee = float(knee)
    self._prev_drop = float(drop)

    if self.state in ('DESCENDING', 'BOTTOM_REACHED', 'ASCENDING'):
      if 'depth_insufficient' in self.frame_issues:
        return self._throttled_feedback(t, '再下去一点，先把完整行程做出来', 'depth_insufficient')
      if 'depth_too_deep' in self.frame_issues:
        return self._throttled_feedback(t, '先别追求太深，蹲到稳稳能控制的位置就行', 'depth_too_deep')
      if 'torso_lean' in self.frame_issues:
        return self._throttled_feedback(t, '胸抬起来一点，核心收紧，慢慢起', 'torso_lean')

    return Feedback('', '')

  def _motivation_line(self) -> str:
    """内部辅助函数。"""
    if self.rep_count == 0:
      return '先把节奏找稳，我们开始！'
    return '不错，继续保持完整行程！'
