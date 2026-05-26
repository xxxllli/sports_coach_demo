"""卷腹规则引擎。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import coco17
from .base import ExerciseEngine, Feedback, RepRecord


@dataclass
class CrunchParams:
  """卷腹规则阈值集合。

  说明：
  1. 当前规则更适合侧面或偏侧面机位；
  2. 由于只有人体关键点，没有地面、垫子和骨盆精细姿态信息，因此“卷起高度、是否借腿、
     是否靠惯性甩起”等判断都是基于肩-髋躯干角、肩部上抬比例、髋膝漂移和角速度等代理特征；
  3. 规则更贴近传统地面卷腹，而不是 V-up、仰卧起坐或俄式转体。
  """

  # ---------- rep 起始 / 结束判定 ----------
  # 起始平躺准备位：躯干抬起角较小。
  start_ready_torso_deg: float = 22.0
  # 开始卷起：躯干抬起角明显进入发力区。
  curl_entry_torso_deg: float = 28.0
  # 顶部工作区：达到该角度后，说明已经完成了有效卷起。
  top_torso_deg: float = 40.0
  # 回到底部准备下一次动作时的躯干角阈值。
  return_ready_torso_deg: float = 24.0

  # ---------- 质量评分 ----------
  grade_perfect_top_low_deg: float = 38.0
  grade_perfect_top_high_deg: float = 65.0
  grade_good_top_low_deg: float = 30.0
  grade_good_top_high_deg: float = 75.0

  # ---------- 稳定性 / 代偿 ----------
  # 髋部漂移阈值：卷腹应以躯干卷曲为主，而不是明显抬髋或前移骨盆。
  hip_drift_warn_ratio: float = 0.10
  # 膝部漂移阈值：若膝盖/下肢明显移动，常见于借腿、拉腿或整体坐起。
  knee_drift_warn_ratio: float = 0.12
  # 角速度过快阈值：用于识别“甩上去”的惯性型卷腹。
  momentum_warn_deg_s: float = 125.0
  # 单次过快阈值：从体验上与角速度一起构成“控制不足”的判断。
  tempo_fast_warn_s: float = 0.65

  min_conf: float = 0.25


class CrunchEngine(ExerciseEngine):
  """卷腹计数与打分引擎。"""

  def __init__(self, params: CrunchParams, cooldown_s: float = 0.8):
    """初始化卷腹状态。"""
    super().__init__(cooldown_s=cooldown_s)
    self.p = params

    # 状态定义：
    # DOWN_READY：底部准备位
    # CURLING：向上卷起
    # TOP_REACHED：到达顶部工作区
    # LOWERING：向下回程
    self.state = 'DOWN_READY'

    self._cur_start_t: Optional[float] = None
    self._cur_issues: set[str] = set()

    self._torso_angle_max = 0.0
    self._hip_drift_ratio_max = 0.0
    self._knee_drift_ratio_max = 0.0
    self._ang_speed_max = 0.0
    self._return_torso_angle = 180.0

    self._hip_ref = None
    self._knee_ref = None
    self._prev_torso_angle: Optional[float] = None
    self._top_seen = False

  def _valid(self, k, idx: int) -> bool:
    """判断指定关键点是否有效。"""
    return k is not None and float(k[idx, 2]) >= self.p.min_conf

  def _choose_side(self, k):
    """选择当前更清晰的一侧躯干链。

    卷腹通常需要肩-髋-膝这一整条链比较稳定，因此这里优先选择可见点更多的一侧。
    """
    left_ok = sum([self._valid(k, coco17.L_SHOULDER), self._valid(k, coco17.L_HIP), self._valid(k, coco17.L_KNEE)])
    right_ok = sum([self._valid(k, coco17.R_SHOULDER), self._valid(k, coco17.R_HIP), self._valid(k, coco17.R_KNEE)])
    if right_ok > left_ok:
      return coco17.R_SHOULDER, coco17.R_HIP, coco17.R_KNEE
    return coco17.L_SHOULDER, coco17.L_HIP, coco17.L_KNEE

  def _torso_len(self, k) -> Optional[float]:
    """计算肩髋距离，作为卷腹尺度归一化长度。"""
    s, h, _kn = self._choose_side(k)
    if not (self._valid(k, s) and self._valid(k, h)):
      return None
    return float(max(1.0, np.linalg.norm(k[s, 0:2] - k[h, 0:2])))

  def _torso_angle_to_horizontal(self, k) -> Optional[float]:
    """计算肩髋连线相对水平线的夹角。

    对地面卷腹来说：
    - 平躺准备位时，躯干相对水平线的角度较小；
    - 卷起时，肩部抬离地面，肩髋连线角度明显变大。
    """
    s, h, _kn = self._choose_side(k)
    if not (self._valid(k, s) and self._valid(k, h)):
      return None
    v = k[s, 0:2] - k[h, 0:2]
    ang = abs(float(np.degrees(np.arctan2(float(v[1]), float(v[0])))))
    if ang > 90.0:
      ang = 180.0 - ang
    return float(ang)

  def _hip_point(self, k):
    """返回主要工作侧髋点。"""
    _s, h, _kn = self._choose_side(k)
    if not self._valid(k, h):
      return None
    return k[h, 0:2].copy()

  def _knee_point(self, k):
    """返回主要工作侧膝点。"""
    _s, _h, kn = self._choose_side(k)
    if not self._valid(k, kn):
      return None
    return k[kn, 0:2].copy()

  def _hip_drift_ratio(self, k) -> Optional[float]:
    """计算髋部漂移比例。"""
    torso_len = self._torso_len(k)
    hip = self._hip_point(k)
    if torso_len is None or hip is None:
      return None
    if self._hip_ref is None:
      self._hip_ref = hip.copy()
    return float(np.linalg.norm(hip - self._hip_ref) / torso_len)

  def _knee_drift_ratio(self, k) -> Optional[float]:
    """计算膝部漂移比例。"""
    torso_len = self._torso_len(k)
    knee = self._knee_point(k)
    if torso_len is None or knee is None:
      return None
    if self._knee_ref is None:
      self._knee_ref = knee.copy()
    return float(np.linalg.norm(knee - self._knee_ref) / torso_len)

  def _reset_rep_stats(self, t: float, k):
    """在一次新卷腹开始时重置统计量。"""
    self._cur_start_t = t
    self._cur_issues = set()
    self._torso_angle_max = 0.0
    self._hip_drift_ratio_max = 0.0
    self._knee_drift_ratio_max = 0.0
    self._ang_speed_max = 0.0
    self._return_torso_angle = 180.0
    self._hip_ref = self._hip_point(k)
    self._knee_ref = self._knee_point(k)
    self._prev_torso_angle = self._torso_angle_to_horizontal(k)
    self._top_seen = False

  def _grade(self, top_torso_deg: float, issues: set[str]) -> str:
    """根据卷起高度和主要代偿问题给出等级。"""
    major_issue = any(it in issues for it in ('hip_flexion_compensation', 'momentum_excessive'))
    perfect_by_rom = self.p.grade_perfect_top_low_deg <= top_torso_deg <= self.p.grade_perfect_top_high_deg
    good_by_rom = self.p.grade_good_top_low_deg <= top_torso_deg <= self.p.grade_good_top_high_deg
    if perfect_by_rom and not major_issue:
      return 'perfect'
    if good_by_rom:
      return 'good'
    return 'poor'

  def update(self, kpts, t: float, fps: float) -> Feedback:
    """消费当前帧关键点，推进卷腹状态机并返回反馈。"""
    self.frame_issues = []
    self.rep_completed = False
    self.last_rep = None

    # ---------- 1) 提取当前卷腹关键特征 ----------
    torso_angle = self._torso_angle_to_horizontal(kpts)
    if torso_angle is None:
      return Feedback('', '')
    hip_drift = self._hip_drift_ratio(kpts)
    knee_drift = self._knee_drift_ratio(kpts)

    if self._prev_torso_angle is None:
      self._prev_torso_angle = float(torso_angle)
    ang_speed = abs(float(torso_angle) - float(self._prev_torso_angle)) * max(1.0, float(fps))
    moving_up = float(torso_angle) > float(self._prev_torso_angle) + 0.6
    moving_down = float(torso_angle) < float(self._prev_torso_angle) - 0.6

    if self.state != 'DOWN_READY':
      self._torso_angle_max = max(self._torso_angle_max, float(torso_angle))
      self._ang_speed_max = max(self._ang_speed_max, float(ang_speed))
      if hip_drift is not None:
        self._hip_drift_ratio_max = max(self._hip_drift_ratio_max, float(hip_drift))
      if knee_drift is not None:
        self._knee_drift_ratio_max = max(self._knee_drift_ratio_max, float(knee_drift))

    # ---------- 2) 卷腹状态机 ----------
    if self.state == 'DOWN_READY':
      # 从平躺准备位进入卷起阶段。
      if float(torso_angle) >= self.p.curl_entry_torso_deg and moving_up:
        self.state = 'CURLING'
        self._reset_rep_stats(t, kpts)
        self._torso_angle_max = float(torso_angle)
        self._ang_speed_max = float(ang_speed)
    elif self.state == 'CURLING':
      if float(torso_angle) >= self.p.top_torso_deg:
        self.state = 'TOP_REACHED'
        self._top_seen = True
        if self._torso_angle_max < self.p.grade_good_top_low_deg:
          self.frame_issues.append('rom_insufficient')
    elif self.state == 'TOP_REACHED':
      # 到达顶部后，检测是否开始受控回程。
      if hip_drift is not None and hip_drift > self.p.hip_drift_warn_ratio:
        self._cur_issues.add('hip_flexion_compensation')
        self.frame_issues.append('hip_flexion_compensation')
      if knee_drift is not None and knee_drift > self.p.knee_drift_warn_ratio:
        self._cur_issues.add('hip_flexion_compensation')
        self.frame_issues.append('hip_flexion_compensation')
      if float(self._ang_speed_max) > self.p.momentum_warn_deg_s:
        self._cur_issues.add('momentum_excessive')
        self.frame_issues.append('momentum_excessive')
      if moving_down:
        self.state = 'LOWERING'
    elif self.state == 'LOWERING':
      self._return_torso_angle = min(self._return_torso_angle, float(torso_angle))
      if hip_drift is not None and hip_drift > self.p.hip_drift_warn_ratio:
        self._cur_issues.add('hip_flexion_compensation')
        self.frame_issues.append('hip_flexion_compensation')
      if knee_drift is not None and knee_drift > self.p.knee_drift_warn_ratio:
        self._cur_issues.add('hip_flexion_compensation')
        self.frame_issues.append('hip_flexion_compensation')
      if float(self._ang_speed_max) > self.p.momentum_warn_deg_s:
        self._cur_issues.add('momentum_excessive')
        self.frame_issues.append('momentum_excessive')
      # 卷腹回到底部准备位后，记为一次完整 rep。
      if self._top_seen and float(torso_angle) <= self.p.return_ready_torso_deg:
        self.state = 'DOWN_READY'
        self.rep_count += 1
        st = float(self._cur_start_t) if self._cur_start_t is not None else max(0.0, t - 0.5)
        dur = float(max(0.0, t - st))
        if dur < self.p.tempo_fast_warn_s:
          self._cur_issues.add('momentum_excessive')
        # 如果回程没有充分回到底部，也记为问题。
        if self._return_torso_angle > self.p.start_ready_torso_deg:
          self._cur_issues.add('return_incomplete')

        grade = self._grade(float(self._torso_angle_max), self._cur_issues)
        rr = RepRecord(
          rep_idx=self.rep_count,
          start_t=st,
          end_t=float(t),
          issues=sorted(self._cur_issues),
          metrics={
            'torso_angle_top_max_deg': float(self._torso_angle_max),
            'hip_drift_ratio_max': float(self._hip_drift_ratio_max),
            'knee_drift_ratio_max': float(self._knee_drift_ratio_max),
            'angular_speed_max_deg_s': float(self._ang_speed_max),
            'return_torso_angle_min_deg': float(self._return_torso_angle),
            'tempo_s': dur,
          },
          grade=grade,
        )
        self._reps.append(rr)
        self.rep_completed = True
        self.last_rep = rr
        self._cur_start_t = None
        self._top_seen = False

    # ---------- 3) 实时提示：优先提醒幅度、借腿和惯性 ----------
    if self.state in ('CURLING', 'TOP_REACHED', 'LOWERING'):
      if self.state in ('CURLING', 'TOP_REACHED') and self._torso_angle_max < self.p.grade_good_top_low_deg:
        self.frame_issues.append('rom_insufficient')
      if 'hip_flexion_compensation' in self.frame_issues:
        return self._throttled_feedback(t, '别用腿带，卷腹时保持骨盆和下肢更稳', 'hip_flexion_compensation')
      if 'momentum_excessive' in self.frame_issues:
        return self._throttled_feedback(t, '别甩起来，卷起和回程都放慢一点', 'momentum_excessive')
      if 'rom_insufficient' in self.frame_issues:
        return self._throttled_feedback(t, '再卷高一点，想象肋骨向骨盆靠近', 'rom_insufficient')

    self._prev_torso_angle = float(torso_angle)
    return Feedback('', '')

  def _motivation_line(self) -> str:
    """给卷腹提供更贴近动作的鼓励文案。"""
    if self.rep_count == 0:
      return '先把卷起高度做出来，再追求节奏。'
    return '卷得不错，继续保持控制，不要借腿！'
