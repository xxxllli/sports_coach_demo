"""杠铃弯举规则引擎。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import coco17
from ..geometry import angle_deg
from .base import ExerciseEngine, Feedback, RepRecord


@dataclass
class CurlParams:
  """杠铃弯举规则阈值集合。"""
  lift_entry_elbow_deg: float = 138.0
  top_hint_elbow_deg: float = 112.0
  return_ready_elbow_deg: float = 138.0
  top_wrist_lift_ratio: float = 0.12
  return_wrist_lift_ratio: float = 0.05

  grade_perfect_max_elbow_deg: float = 35.0
  grade_good_max_elbow_deg: float = 55.0
  elbow_drift_warn_ratio: float = 0.12
  trunk_swing_warn_deg: float = 14.0
  min_conf: float = 0.25


class CurlEngine(ExerciseEngine):
  """杠铃弯举计数与打分引擎。"""

  def __init__(self, params: CurlParams, cooldown_s: float = 0.8):
    """初始化杠铃弯举引擎状态。"""
    super().__init__(cooldown_s=cooldown_s)
    self.p = params
    self.state = 'DOWN_READY'
    self._cur_start_t: Optional[float] = None
    self._cur_issues: set[str] = set()

    self._elbow_ref = None
    self._wrist_ref_y: Optional[float] = None
    self._elbow_drift_max_px = 0.0
    self._elbow_drift_ratio_max = 0.0
    self._trunk_angle_min = 180.0
    self._trunk_angle_max = 0.0
    self._top_vals: deque[float] = deque(maxlen=25)
    self._elbow_min = 180.0
    self._prev_ea: Optional[float] = None
    self._prev_lift: Optional[float] = None
    self._top_seen = False

  def _valid(self, k, idx: int) -> bool:
    """判断指定关键点是否有效。"""
    return k is not None and float(k[idx, 2]) >= self.p.min_conf

  def _choose_arm(self, k):
    """内部辅助函数。"""
    left_ok = sum([self._valid(k, coco17.L_SHOULDER), self._valid(k, coco17.L_ELBOW), self._valid(k, coco17.L_WRIST)])
    right_ok = sum([self._valid(k, coco17.R_SHOULDER), self._valid(k, coco17.R_ELBOW), self._valid(k, coco17.R_WRIST)])
    if right_ok > left_ok:
      return coco17.R_SHOULDER, coco17.R_ELBOW, coco17.R_WRIST
    return coco17.L_SHOULDER, coco17.L_ELBOW, coco17.L_WRIST

  def _elbow_angle(self, k) -> Optional[float]:
    """计算当前姿态对应的角度特征。"""
    s, e, w = self._choose_arm(k)
    if not (self._valid(k, s) and self._valid(k, e) and self._valid(k, w)):
      return None
    return float(angle_deg(k[s, 0:2], k[e, 0:2], k[w, 0:2]))

  def _elbow_point(self, k) -> Optional[np.ndarray]:
    """内部辅助函数。"""
    _s, e, _w = self._choose_arm(k)
    if not self._valid(k, e):
      return None
    return k[e, 0:2].copy()

  def _wrist_y(self, k) -> Optional[float]:
    """内部辅助函数。"""
    _s, _e, w = self._choose_arm(k)
    if not self._valid(k, w):
      return None
    return float(k[w, 1])

  def _torso_len(self, k) -> Optional[float]:
    """内部辅助函数。"""
    if not (self._valid(k, coco17.L_SHOULDER) and self._valid(k, coco17.R_SHOULDER) and self._valid(k, coco17.L_HIP) and self._valid(k, coco17.R_HIP)):
      return None
    sh = (k[coco17.L_SHOULDER, 0:2] + k[coco17.R_SHOULDER, 0:2]) / 2
    hp = (k[coco17.L_HIP, 0:2] + k[coco17.R_HIP, 0:2]) / 2
    return float(max(1.0, np.linalg.norm(sh - hp)))

  def _trunk_angle(self, k) -> Optional[float]:
    """计算当前姿态对应的角度特征。"""
    if not (self._valid(k, coco17.L_SHOULDER) and self._valid(k, coco17.R_SHOULDER) and self._valid(k, coco17.L_HIP) and self._valid(k, coco17.R_HIP)):
      return None
    sh = (k[coco17.L_SHOULDER, 0:2] + k[coco17.R_SHOULDER, 0:2]) / 2
    hp = (k[coco17.L_HIP, 0:2] + k[coco17.R_HIP, 0:2]) / 2
    v = sh - hp
    nv = float(np.linalg.norm(v))
    if nv < 1e-6:
      return None
    vert = np.array([0.0, -1.0], dtype=np.float32)
    cosang = float(np.clip(np.dot(v / nv, vert), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))

  def _wrist_lift_ratio(self, k) -> Optional[float]:
    """计算归一化比例特征。"""
    wy = self._wrist_y(k)
    torso_len = self._torso_len(k)
    if wy is None or torso_len is None:
      return None
    if self._wrist_ref_y is None:
      self._wrist_ref_y = wy
    return max(0.0, float((self._wrist_ref_y - wy) / torso_len))

  def _reset_rep_stats(self, t: float, k):
    """内部辅助函数。"""
    self._cur_start_t = t
    self._cur_issues = set()
    self._elbow_ref = self._elbow_point(k)
    self._wrist_ref_y = self._wrist_y(k)
    self._elbow_drift_max_px = 0.0
    self._elbow_drift_ratio_max = 0.0
    ta = self._trunk_angle(k)
    self._trunk_angle_min = float(ta) if ta is not None else 180.0
    self._trunk_angle_max = float(ta) if ta is not None else 0.0
    self._top_vals.clear()
    self._elbow_min = 180.0
    self._prev_ea = self._elbow_angle(k)
    self._prev_lift = self._wrist_lift_ratio(k)
    self._top_seen = False

  def update(self, kpts, t: float, fps: float) -> Feedback:
    """消费当前帧关键点，推进弯举状态机并返回反馈。"""
    self.frame_issues = []
    self.rep_completed = False
    self.last_rep = None

    # 1) 先提取当前帧最关键的弯举特征：肘角、手腕抬起比例、躯干尺度。
    ea = self._elbow_angle(kpts)
    if ea is None:
      return Feedback('', '')
    lift = self._wrist_lift_ratio(kpts)
    torso_len = self._torso_len(kpts)

    if self.state != 'DOWN_READY':
      self._elbow_min = min(self._elbow_min, float(ea))
      if ea <= self.p.top_hint_elbow_deg + 15.0:
        self._top_vals.append(float(ea))
      ep = self._elbow_point(kpts)
      if ep is not None and self._elbow_ref is not None:
        drift = float(np.linalg.norm(ep - self._elbow_ref))
        self._elbow_drift_max_px = max(self._elbow_drift_max_px, drift)
        if torso_len is not None and torso_len > 1.0:
          self._elbow_drift_ratio_max = max(self._elbow_drift_ratio_max, drift / torso_len)
      ta = self._trunk_angle(kpts)
      if ta is not None:
        self._trunk_angle_min = min(self._trunk_angle_min, float(ta))
        self._trunk_angle_max = max(self._trunk_angle_max, float(ta))

    # 2) 用前后帧差分判断当前是在向上发力还是向下回落。
    moving_up = self._prev_ea is not None and float(ea) < float(self._prev_ea) - 0.8
    moving_down = self._prev_ea is not None and float(ea) > float(self._prev_ea) + 0.8
    lift_up = lift is not None and self._prev_lift is not None and float(lift) > float(self._prev_lift) + 0.005
    lift_down = lift is not None and self._prev_lift is not None and float(lift) < float(self._prev_lift) - 0.005

    # 3) 状态机：DOWN_READY -> LIFTING -> TOP_REACHED -> LOWERING -> DOWN_READY。
    if self.state == 'DOWN_READY':
      if (float(ea) <= self.p.lift_entry_elbow_deg and moving_up) or (lift is not None and lift >= self.p.return_wrist_lift_ratio and lift_up):
        self.state = 'LIFTING'
        self._reset_rep_stats(t, kpts)
        self._elbow_min = float(ea)
    elif self.state == 'LIFTING':
      if float(ea) <= self.p.top_hint_elbow_deg or (lift is not None and lift >= self.p.top_wrist_lift_ratio):
        self.state = 'TOP_REACHED'
        self._top_seen = True
    elif self.state == 'TOP_REACHED':
      if moving_down or lift_down:
        self.state = 'LOWERING'
    elif self.state == 'LOWERING':
      if self._elbow_drift_ratio_max > self.p.elbow_drift_warn_ratio:
        self._cur_issues.add('elbow_drift')
        self.frame_issues.append('elbow_drift')
      trunk_swing = max(0.0, self._trunk_angle_max - self._trunk_angle_min)
      if trunk_swing > self.p.trunk_swing_warn_deg:
        self._cur_issues.add('body_swing')
        self.frame_issues.append('body_swing')
      ready_by_angle = float(ea) >= self.p.return_ready_elbow_deg
      ready_by_lift = lift is not None and lift <= self.p.return_wrist_lift_ratio
      if self._top_seen and (ready_by_angle or ready_by_lift) and (moving_down or (lift is not None and lift_down)):
        self.state = 'DOWN_READY'
        self.rep_count += 1
        st = float(self._cur_start_t) if self._cur_start_t is not None else max(0.0, t - 0.5)
        elbow_top_med = float(np.median(list(self._top_vals))) if self._top_vals else float(self._elbow_min)
        if elbow_top_med > self.p.grade_good_max_elbow_deg:
          self._cur_issues.add('rom_insufficient')
        if elbow_top_med <= self.p.grade_perfect_max_elbow_deg:
          grade = 'perfect'
        elif elbow_top_med <= self.p.grade_good_max_elbow_deg:
          grade = 'good'
        else:
          grade = 'poor'
        # 4) 当一次完整弯举闭环结束后，生成一个 RepRecord 供课后总结使用。
        rr = RepRecord(
          rep_idx=self.rep_count,
          start_t=st,
          end_t=float(t),
          issues=sorted(self._cur_issues),
          metrics={
            'elbow_top_med_deg': elbow_top_med,
            'elbow_min_deg': float(self._elbow_min),
            'elbow_drift_px': float(self._elbow_drift_max_px),
            'elbow_drift_ratio': float(self._elbow_drift_ratio_max),
            'trunk_swing_deg': float(max(0.0, self._trunk_angle_max - self._trunk_angle_min)),
            'wrist_lift_ratio_max': float(max(self.p.return_wrist_lift_ratio, lift or 0.0)),
          },
          grade=grade,
        )
        self._reps.append(rr)
        self.rep_completed = True
        self.last_rep = rr
        self._cur_start_t = None
        self._top_seen = False
        self._wrist_ref_y = self._wrist_y(kpts)

    self._prev_ea = float(ea)
    self._prev_lift = float(lift) if lift is not None else self._prev_lift

    # 5) 逐帧提示主要发生在动作顶部和下放阶段，避免在静止阶段刷屏。
    if self.state in ('TOP_REACHED', 'LOWERING'):
      top_ref = float(np.median(list(self._top_vals))) if self._top_vals else float(self._elbow_min)
      if top_ref > self.p.grade_good_max_elbow_deg:
        self.frame_issues.append('rom_insufficient')
      if 'rom_insufficient' in self.frame_issues:
        return self._throttled_feedback(t, '再卷高一点，顶端稍微停一下', 'rom_insufficient')
      if 'elbow_drift' in self.frame_issues:
        return self._throttled_feedback(t, '肘固定在身体两侧，别往前跑', 'elbow_drift')
      if 'body_swing' in self.frame_issues:
        return self._throttled_feedback(t, '身体别借力，下降放慢一点', 'body_swing')

    return Feedback('', '')

  def _motivation_line(self) -> str:
    """内部辅助函数。"""
    if self.rep_count == 0:
      return '先把动作做稳，再慢慢加速！'
    return '很好，保持节奏，别借力！'
