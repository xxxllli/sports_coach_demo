"""
非常轻量的机位视角启发式判断。

这些规则不参与核心动作计数，只用于补充报告里的机位提示。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from . import coco17


def _valid(kpts, idx: int, min_conf: float) -> bool:
  """判断指定关键点是否存在且置信度达标。"""
  return kpts is not None and idx < int(kpts.shape[0]) and float(kpts[idx, 2]) >= float(min_conf)


def _pt(kpts, idx: int, min_conf: float):
  """安全读取某个关键点的二维坐标。"""
  if _valid(kpts, idx, min_conf):
    return kpts[idx, 0:2].astype(np.float32)
  return None


def _pair_width(kpts, a: int, b: int, min_conf: float) -> Optional[float]:
  """计算两个关键点之间的水平距离。"""
  pa = _pt(kpts, a, min_conf)
  pb = _pt(kpts, b, min_conf)
  if pa is None or pb is None:
    return None
  return float(abs(pa[0] - pb[0]))


def _body_height(kpts, min_conf: float) -> Optional[float]:
  """估算人体在当前帧中的有效高度。"""
  xs, ys = [], []
  if kpts is None:
    return None
  for i in range(int(kpts.shape[0])):
    if float(kpts[i, 2]) >= float(min_conf):
      xs.append(float(kpts[i, 0]))
      ys.append(float(kpts[i, 1]))
  if len(ys) < 4:
    return None
  return max(1.0, max(ys) - min(ys))


def infer_view(exercise: str, kpts, min_conf: float = 0.25) -> Dict[str, Any]:
  """粗略判断机位更像正面、侧面还是不确定。"""
  bh = _body_height(kpts, min_conf)
  if bh is None:
    return {'type': 'uncertain', 'confidence': 0.0, 'valid_for_exercise': False, 'reason': '关键点太少，无法判断机位', 'features': {}}

  shoulder_w = _pair_width(kpts, coco17.L_SHOULDER, coco17.R_SHOULDER, min_conf)
  hip_w = _pair_width(kpts, coco17.L_HIP, coco17.R_HIP, min_conf)
  knee_w = _pair_width(kpts, coco17.L_KNEE, coco17.R_KNEE, min_conf)
  ankle_w = _pair_width(kpts, coco17.L_ANKLE, coco17.R_ANKLE, min_conf)

  visible_pairs = []
  for a, b in [
    (coco17.L_SHOULDER, coco17.R_SHOULDER),
    (coco17.L_HIP, coco17.R_HIP),
    (coco17.L_KNEE, coco17.R_KNEE),
    (coco17.L_ANKLE, coco17.R_ANKLE),
  ]:
    visible_pairs.append(int(_valid(kpts, a, min_conf)) + int(_valid(kpts, b, min_conf)))
  bilateral_ratio = sum(1 for v in visible_pairs if v == 2) / max(1, len(visible_pairs))

  ratios = [x / bh for x in [shoulder_w, hip_w, knee_w, ankle_w] if x is not None]
  avg_ratio = float(np.mean(ratios)) if ratios else 0.0

  front_score = 0.0
  if avg_ratio > 0.18:
    front_score += min(1.0, (avg_ratio - 0.18) / 0.12)
  front_score += 0.5 * bilateral_ratio
  front_score = min(1.5, front_score)

  side_score = 0.0
  if avg_ratio < 0.15:
    side_score += min(1.0, (0.15 - avg_ratio) / 0.10)
  side_score += 0.5 * (1.0 - bilateral_ratio)
  side_score = min(1.5, side_score)

  if front_score >= side_score + 0.15 and front_score >= 0.55:
    view_type = 'front_like'
    conf = min(0.95, front_score / 1.5)
  elif side_score >= front_score + 0.15 and side_score >= 0.55:
    view_type = 'side_like'
    conf = min(0.95, side_score / 1.5)
  else:
    view_type = 'uncertain'
    conf = 0.35

  expected = 'front_like' if exercise == 'jumping_jack' else 'side_like'
  valid = (view_type == expected and conf >= 0.5)

  if view_type == 'uncertain':
    reason = '机位不够明确，识别结果可能受影响'
  elif valid:
    reason = '当前机位基本适合这个动作'
  else:
    reason = '当前机位可能不是这个动作的理想拍法'

  return {
    'type': view_type,
    'confidence': round(float(conf), 3),
    'valid_for_exercise': bool(valid),
    'expected': expected,
    'reason': reason,
    'features': {
      'body_height_px': round(float(bh), 1),
      'avg_pair_width_ratio': round(float(avg_ratio), 3),
      'bilateral_ratio': round(float(bilateral_ratio), 3),
      'shoulder_w_px': round(float(shoulder_w), 1) if shoulder_w is not None else None,
      'hip_w_px': round(float(hip_w), 1) if hip_w is not None else None,
      'knee_w_px': round(float(knee_w), 1) if knee_w is not None else None,
      'ankle_w_px': round(float(ankle_w), 1) if ankle_w is not None else None,
    },
  }
