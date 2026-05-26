"""俯卧撑规则引擎。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import coco17
from ..geometry import angle_deg
from .base import ExerciseEngine, Feedback, RepRecord


@dataclass
class PushupParams:
  """俯卧撑规则阈值集合。"""
  elbow_down_deg: float = 95.0
  elbow_up_deg: float = 160.0
  depth_warn_elbow_deg: float = 105.0
  body_bend_warn_deg: float = 18.0
  min_conf: float = 0.25


class PushupEngine(ExerciseEngine):
  """俯卧撑计数与打分引擎。"""

  def __init__(self, params: PushupParams, cooldown_s: float = 0.8):
    """初始化俯卧撑引擎状态。"""
    super().__init__(cooldown_s=cooldown_s)
    self.p = params
    self.state = 'UP'
    self._cur_start_t: Optional[float] = None
    self._cur_issues: set[str] = set()
    self._elbow_min = 180.0
    self._bend_max = 0.0

  def _valid(self, k, idx: int) -> bool:
    """判断指定关键点是否有效。"""
    return k is not None and float(k[idx, 2]) >= self.p.min_conf

  def _elbow_angle(self, k) -> Optional[float]:
    """计算当前姿态对应的角度特征。"""
    vals = []
    if self._valid(k, coco17.L_SHOULDER) and self._valid(k, coco17.L_ELBOW) and self._valid(k, coco17.L_WRIST):
      vals.append(angle_deg(k[coco17.L_SHOULDER,0:2], k[coco17.L_ELBOW,0:2], k[coco17.L_WRIST,0:2]))
    if self._valid(k, coco17.R_SHOULDER) and self._valid(k, coco17.R_ELBOW) and self._valid(k, coco17.R_WRIST):
      vals.append(angle_deg(k[coco17.R_SHOULDER,0:2], k[coco17.R_ELBOW,0:2], k[coco17.R_WRIST,0:2]))
    vals = [float(v) for v in vals if np.isfinite(v)]
    if not vals:
      return None
    return float(min(vals))

  def _body_bend(self, k) -> Optional[float]:
    """内部辅助函数。"""
    if not (self._valid(k,coco17.L_SHOULDER) and self._valid(k,coco17.R_SHOULDER) and self._valid(k,coco17.L_HIP) and self._valid(k,coco17.R_HIP) and self._valid(k,coco17.L_ANKLE) and self._valid(k,coco17.R_ANKLE)):
      return None
    sh = (k[coco17.L_SHOULDER,0:2] + k[coco17.R_SHOULDER,0:2]) / 2
    hp = (k[coco17.L_HIP,0:2] + k[coco17.R_HIP,0:2]) / 2
    an = (k[coco17.L_ANKLE,0:2] + k[coco17.R_ANKLE,0:2]) / 2
    ang = angle_deg(sh, hp, an)
    if not np.isfinite(ang):
      return None
    return float(abs(180.0 - ang))

  def update(self, kpts, t: float, fps: float) -> Feedback:
    """消费当前帧关键点，推进俯卧撑状态机并返回反馈。"""
    self.frame_issues = []
    self.rep_completed = False
    self.last_rep = None

    # 1) 俯卧撑主要看两个特征：肘角够不够低、身体是否保持足够平直。
    ea = self._elbow_angle(kpts)
    if ea is None:
      return Feedback('', '')
    bend = self._body_bend(kpts)

    # 2) 状态机非常简单：UP（顶部）和 DOWN（底部工作段）两态切换。
    if self.state == 'UP':
      if ea < self.p.elbow_down_deg:
        self.state = 'DOWN'
        self._cur_start_t = t
        self._cur_issues = set()
        self._elbow_min = ea
        self._bend_max = float(bend) if bend is not None else 0.0
    else:
      self._elbow_min = min(self._elbow_min, ea)
      if bend is not None:
        self._bend_max = max(self._bend_max, float(bend))

      if ea > self.p.depth_warn_elbow_deg:
        self._cur_issues.add('depth_insufficient')
        self.frame_issues.append('depth_insufficient')
      if bend is not None and float(bend) > self.p.body_bend_warn_deg:
        self._cur_issues.add('body_not_straight')
        self.frame_issues.append('body_not_straight')

      if ea > self.p.elbow_up_deg:
        self.state = 'UP'
        self.rep_count += 1
        st = float(self._cur_start_t) if self._cur_start_t is not None else max(0.0, t - 0.5)
        # 3) 当重新回到顶部时，认为一次完整俯卧撑结束并落盘。
        rr = RepRecord(rep_idx=self.rep_count, start_t=st, end_t=float(t), issues=sorted(list(self._cur_issues)),
               metrics={'elbow_min_deg': float(self._elbow_min), 'body_bend_max_deg': float(self._bend_max)})
        self._reps.append(rr)
        self.rep_completed = True
        self.last_rep = rr
        self._cur_start_t = None

    # 4) 逐帧提示只在下去的工作段触发，这时最容易及时纠正动作。
    if self.state == 'DOWN':
      if 'body_not_straight' in self.frame_issues:
        return self._throttled_feedback(t, '核心收紧，身体像一块板', 'body_not_straight')
      if 'depth_insufficient' in self.frame_issues:
        return self._throttled_feedback(t, '再下去一点，胸更靠近地面', 'depth_insufficient')

    return Feedback('', '')
