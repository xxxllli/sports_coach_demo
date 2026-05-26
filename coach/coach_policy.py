"""
实时教练提示策略。

负责决定在什么时机说什么提示，而不是负责动作识别本身。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .report_text import advice_for


@dataclass
class CoachOutput:
  """实时教练输出结构。"""
  text: str = ''
  src: str = 'auto' # 文案来源：rule / policy / group / time / vlm


def _top_issue(issue_counts: Dict[str, int]) -> Optional[str]:
  """从问题计数字典中找出当前最主要的问题标签。"""
  if not issue_counts:
    return None
  return sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[0][0]


def _goal_push_line(remaining: int) -> str:
  """根据剩余目标次数生成一句鼓励文案。"""
  return f'再做{remaining}个，我们就达成今天的目标了！'


def _issue_line(issue: str, mode: str = 'simple') -> str:
  """把问题标签转换成一句短提示。"""
  if issue == 'depth_insufficient':
    return '深度不够，再下去一点' if mode == 'simple' else '深度不足：最低点再下去一点'
  if issue == 'depth_too_deep':
    return '有点太深了，稳住膝盖' if mode == 'simple' else '下蹲过深：控制到更舒服的深度'
  if issue == 'torso_lean':
    return '胸抬起来，背更直一些'
  if issue == 'hands_low':
    return '手抬高一点，尽量过头顶'
  if issue == 'feet_narrow':
    return '脚再打开一点，落地稳'
  if issue == 'hands_not_closed':
    return '手回到身体两侧，再合得干净一点'
  if issue == 'feet_not_closed':
    return '脚合回一点，动作更利索'
  if issue == 'rom_insufficient':
    return '顶端再卷一点' if mode == 'simple' else '弯举顶端收紧一点'
  if issue == 'elbow_drift':
    return '肘固定在身体两侧别跑'
  if issue == 'body_swing':
    return '身体别借力，慢一点'
  if issue == 'body_not_straight':
    return '核心收紧，身体像一块板'
  if issue == 'lockout_incomplete':
    return '顶部再推直一点，完成锁定'
  if issue == 'forearm_not_vertical':
    return '底部让腕叠在肘上方，前臂更竖直'
  if issue == 'hip_bridge':
    return '臀部别顶桥，肩臀保持稳定'
  if issue == 'tempo_too_fast':
    return '节奏太快了，下放慢一点'
  if issue == 'hip_flexion_compensation':
    return '别用腿带，重点把腹部卷起来'
  if issue == 'momentum_excessive':
    return '别甩起来，卷起和回程都放慢'
  if issue == 'return_incomplete':
    return '回程再放低一点，再开始下一次'
  return '动作再稳一点'


class CoachPolicy:
  """
  实时教练提示策略。
  
  输入来自规则引擎的逐帧信号，输出当前帧应该显示/播报的文本。
  """

  def __init__(self, mode: str = 'event', group_n: int = 3, interval_s: float = 4.0, goal_reps: int = 20, user_mode: str = 'simple'):
    """初始化策略对象，并记录模式、组大小、时间间隔等参数。"""
    self.mode = mode
    self.group_n = max(1, int(group_n))
    self.interval_s = max(0.5, float(interval_s))
    self.goal_reps = int(goal_reps)
    self.user_mode = str(user_mode)

    self._seen_goal_milestones: set[int] = set()
    self._group_issue_counts: Dict[str, int] = {}
    self._group_rep_start = 1
    self._time_issue_counts: Dict[str, int] = {}
    self._next_time_t: float = 0.0

  def _add_issues(self, counter: Dict[str, int], issues: List[str]):
    """把当前帧或当前次 rep 的问题累加到计数字典中。"""
    for it in issues:
      counter[it] = counter.get(it, 0) + 1

  def _maybe_goal(self, rep_count: int) -> str:
    """在适当时机加入目标进度鼓励文案。"""
    remaining = max(0, self.goal_reps - rep_count)
    if remaining in (10, 5, 3, 1) and remaining not in self._seen_goal_milestones and remaining > 0:
      self._seen_goal_milestones.add(remaining)
      return _goal_push_line(remaining)
    if remaining == 0 and 0 not in self._seen_goal_milestones:
      self._seen_goal_milestones.add(0)
      return '目标达成！太棒了！'
    return ''

  def step(self, t: float, rep_count: int, frame_issues: List[str], rep_completed: bool, rep_issues: List[str] | None, rep_grade: str | None, rule_text: str) -> CoachOutput:
    """消费当前帧信号，决定这一帧是否要输出教练提示。"""
    if frame_issues:
      self._add_issues(self._time_issue_counts, frame_issues)
    if rep_completed and rep_issues is not None:
      self._add_issues(self._group_issue_counts, rep_issues)

    if self.mode == 'event':
      if frame_issues:
        top = _top_issue({it: 1 for it in frame_issues})
        if top:
          return CoachOutput(text=_issue_line(top, mode=self.user_mode), src='rule')

      if rep_completed and rep_count > 0 and (rep_count % self.group_n == 0):
        top = _top_issue(self._group_issue_counts)
        if top is None:
          msg = '这一组做得很稳！继续保持。' if self.user_mode == 'simple' else '该组整体稳定，动作质量不错。'
        else:
          msg = f'{_issue_line(top, mode=self.user_mode)}。{advice_for(top, self.user_mode)}'
        self._group_issue_counts = {}
        self._group_rep_start = rep_count + 1
        g = self._maybe_goal(rep_count)
        if g:
          msg = f'{msg} {g}'
        return CoachOutput(text=msg, src='group')

      if rep_completed:
        if rep_grade == 'poor':
          return CoachOutput(text='这一下有点走样，放慢一点再来。', src='policy')
        g = self._maybe_goal(rep_count)
        if g:
          return CoachOutput(text=g, src='policy')
        return CoachOutput(text='不错，继续保持！', src='policy')

      if rule_text:
        return CoachOutput(text=rule_text, src='rule')
      return CoachOutput('', 'auto')

    if self.mode == 'group':
      if rep_completed and rep_count > 0 and (rep_count % self.group_n == 0):
        top = _top_issue(self._group_issue_counts)
        if top is None:
          msg = '这组做得很稳！继续保持这个节奏。'
        else:
          msg = f'{_issue_line(top, mode=self.user_mode)}。{advice_for(top, self.user_mode)}'
        self._group_issue_counts = {}
        self._group_rep_start = rep_count + 1
        g = self._maybe_goal(rep_count)
        if g:
          msg = f'{msg} {g}'
        return CoachOutput(text=msg, src='group')
      return CoachOutput('', 'auto')

    if self.mode == 'time':
      if self._next_time_t <= 0.0:
        self._next_time_t = t + self.interval_s
      if t >= self._next_time_t:
        top = _top_issue(self._time_issue_counts)
        if top is None:
          msg = '这段做得不错，节奏很稳！'
        else:
          msg = f'{_issue_line(top, mode=self.user_mode)}。{advice_for(top, self.user_mode)}'
        self._time_issue_counts = {}
        self._next_time_t = t + self.interval_s
        g = self._maybe_goal(rep_count)
        if g:
          msg = f'{msg} {g}'
        return CoachOutput(text=msg, src='time')
      return CoachOutput('', 'auto')

    return CoachOutput('', 'auto')
