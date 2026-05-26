"""
关键点平滑工具。

主要使用 EMA（指数滑动平均）来降低关键点抖动。
"""

from __future__ import annotations

import numpy as np


class EMASmoother:
  """
  关键点 EMA 平滑器。
  
  用于降低 YOLO Pose 关键点逐帧抖动对规则引擎的影响。
  """

  def __init__(self, alpha: float = 0.35):
    # alpha 越接近 1.0，输出越跟随当前帧。
    # alpha 越接近 0.0，结果越平滑，但响应也会更慢。
    """
    初始化平滑器。
    
    alpha 越大越跟随当前帧，alpha 越小越平滑但也越滞后。
    """
    self.alpha = float(alpha)
    self._prev = None

  def reset(self):
    """清空历史状态。"""
    self._prev = None

  def update(self, kpts: np.ndarray | None) -> np.ndarray | None:
    """输入当前帧关键点，并返回平滑后的关键点结果。"""
    if kpts is None:
      return None
    kpts = np.asarray(kpts, dtype=np.float32)
    if self._prev is None:
      self._prev = kpts.copy()
      return kpts

    out = kpts.copy()
    # 只有当前帧和历史帧都存在该点时，才对该点做平滑。
    mask = (kpts[:, 2] > 0) & (self._prev[:, 2] > 0)
    out[mask, 0:2] = self.alpha * kpts[mask, 0:2] + (1 - self.alpha) * self._prev[mask, 0:2]
    self._prev = out
    return out
