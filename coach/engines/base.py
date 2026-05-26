"""规则动作引擎的基础抽象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Feedback:
  """规则引擎逐帧反馈。"""
  text: str = ''
  tag: str = ''


@dataclass
class RepRecord:
  """一次完整 rep 的记录。"""

  rep_idx: int
  start_t: float
  end_t: float
  issues: List[str] = field(default_factory=list)
  metrics: Dict[str, float] = field(default_factory=dict)
  # 这次 rep 的质量等级：perfect / good / poor。
  grade: str = 'good'
  # 预留字段：将来如果需要保留不完整 rep，可使用该字段。
  incomplete: bool = False


class ExerciseEngine:
  """所有规则动作引擎的基类。"""

  def __init__(self, cooldown_s: float = 0.8):
    # cooldown_s 用来防止同一个问题在每一帧都重复提示。
    """初始化基类状态，例如 cooldown、rep_count、历史 rep 列表等。"""
    self.cooldown_s = float(cooldown_s)
    self.last_feedback_t = -1e9

    # rep_count 是展示给用户看的实时累计次数。
    self.rep_count = 0
    # _reps 用于保存所有完整 rep，供课后总结与报告使用。
    self._reps: List[RepRecord] = []

    # 这些是每一帧都会被 CoachPolicy 消费的实时信号。
    self.frame_issues: List[str] = []
    self.rep_completed: bool = False
    self.last_rep: Optional[RepRecord] = None

  def _throttled_feedback(self, t: float, text: str, tag: str = '') -> Feedback:
    """只有在冷却时间已过时，才返回一条新的提示。"""
    if not text:
      return Feedback('', '')
    if (t - self.last_feedback_t) < self.cooldown_s:
      return Feedback('', '')
    self.last_feedback_t = t
    return Feedback(text=text, tag=tag)

  def update(self, kpts, t: float, fps: float) -> Feedback:
    """消费当前帧关键点。子类必须实现自己的状态机逻辑。"""
    raise NotImplementedError

  def finalize(self, total_t: float) -> List[RepRecord]:
    """在视频结束后返回最终 rep 列表。"""
    return list(self._reps)

  def _motivation_line(self) -> str:
    """当没有明确问题时使用的默认鼓励文案。"""
    return '继续保持！'
