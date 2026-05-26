"""规则引擎用到的几何计算辅助函数。"""

from __future__ import annotations

import numpy as np


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
  """计算 ∠ABC 的角度值（单位：度）。"""
  a = np.asarray(a, dtype=np.float32)
  b = np.asarray(b, dtype=np.float32)
  c = np.asarray(c, dtype=np.float32)

  # 先把三个点转换成在 b 点相交的两条向量。
  ba = a - b
  bc = c - b
  nba = float(np.linalg.norm(ba))
  nbc = float(np.linalg.norm(bc))
  if nba < 1e-6 or nbc < 1e-6:
    return float('nan')

  cosang = float(np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0))
  return float(np.degrees(np.arccos(cosang)))
