"""用于粗略估算文本与图像 token 消耗的辅助函数。"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Optional

_CJK_RE = re.compile(r'[㐀-䶿一-鿿豈-﫿]')
_WORD_RE = re.compile(r'[A-Za-z0-9_]+')
_PUNCT_RE = re.compile(r'[^\w\s㐀-䶿一-鿿豈-﫿]')


def estimate_text_tokens(text: str) -> int:
  """粗略估算一段文本的 token 数。"""
  if not text:
    return 0
  cjk = len(_CJK_RE.findall(text))
  words = len(_WORD_RE.findall(text))
  punct = len(_PUNCT_RE.findall(text))
  est = cjk * 1.05 + words * 1.3 + punct * 0.35 + max(0, len(text) - cjk - punct) * 0.08
  return int(max(1, round(est)))


def estimate_image_tokens(width: int, height: int) -> int:
  """按图像尺寸粗略估算图像 token 数。"""
  if width <= 0 or height <= 0:
    return 0
  tiles = math.ceil(width / 512.0) * math.ceil(height / 512.0)
  return int(85 + 85 * tiles)


def empty_usage_stats() -> Dict[str, Any]:
  """创建一个空的 usage 统计字典。"""
  return {
    'calls': 0,
    'prompt_tokens_total': 0,
    'completion_tokens_total': 0,
    'total_tokens_total': 0,
    'parts_prompt_tokens_est_total': {},
    'parts_output_tokens_est_total': {},
    'parts_calls': {},
  }


def accumulate_usage(dst: Dict[str, Any], usage: Optional[Dict[str, Any]], part_prompt_est: Optional[Dict[str, int]] = None, part_output_est: Optional[Dict[str, int]] = None) -> None:
  """把一次调用的 usage 统计累加到目标字典中。"""
  if dst is None:
    return
  if 'calls' not in dst:
    dst.update(empty_usage_stats())
  dst['calls'] = int(dst.get('calls', 0)) + 1
  if usage:
    dst['prompt_tokens_total'] = int(dst.get('prompt_tokens_total', 0)) + int(usage.get('prompt_tokens') or 0)
    dst['completion_tokens_total'] = int(dst.get('completion_tokens_total', 0)) + int(usage.get('completion_tokens') or 0)
    dst['total_tokens_total'] = int(dst.get('total_tokens_total', 0)) + int(usage.get('total_tokens') or 0)
  if part_prompt_est:
    bag = dst.setdefault('parts_prompt_tokens_est_total', {})
    calls_bag = dst.setdefault('parts_calls', {})
    for k, v in part_prompt_est.items():
      bag[k] = int(bag.get(k, 0)) + int(v or 0)
      if int(v or 0) > 0:
        calls_bag[k] = int(calls_bag.get(k, 0)) + 1
  if part_output_est:
    bag2 = dst.setdefault('parts_output_tokens_est_total', {})
    for k, v in part_output_est.items():
      bag2[k] = int(bag2.get(k, 0)) + int(v or 0)


def finalize_usage(dst: Dict[str, Any]) -> Dict[str, Any]:
  """补充 usage 的均值字段，便于在报告中展示。"""
  if not dst:
    return {}
  calls = max(1, int(dst.get('calls', 0) or 0))
  out = dict(dst)
  out['prompt_tokens_avg'] = round(float(out.get('prompt_tokens_total', 0)) / calls, 2)
  out['completion_tokens_avg'] = round(float(out.get('completion_tokens_total', 0)) / calls, 2)
  out['total_tokens_avg'] = round(float(out.get('total_tokens_total', 0)) / calls, 2)
  part_prompt_avg = {}
  for k, total in (out.get('parts_prompt_tokens_est_total', {}) or {}).items():
    part_prompt_avg[k] = round(float(total) / calls, 2)
  out['parts_prompt_tokens_est_avg'] = part_prompt_avg
  part_output_avg = {}
  for k, total in (out.get('parts_output_tokens_est_total', {}) or {}).items():
    part_output_avg[k] = round(float(total) / calls, 2)
  out['parts_output_tokens_est_avg'] = part_output_avg
  return out
