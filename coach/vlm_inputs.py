"""构造实时链路与课后链路 VLM 输入内容的辅助函数。"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .token_stats import estimate_image_tokens, estimate_text_tokens
from .vlm_api import _img_to_data_url_jpeg


def sample_evenly(items: List[Any], n: int) -> List[Any]:
  """从序列中尽量均匀抽取 n 个样本。"""
  if not items:
    return []
  n = max(1, int(n))
  if len(items) <= n:
    return list(items)
  idxs = np.linspace(0, len(items) - 1, n).astype(int).tolist()
  return [items[i] for i in idxs]


def sample_tail(items: List[Any], n: int) -> List[Any]:
  """从序列尾部抽取 n 个样本。"""
  if not items:
    return []
  n = max(1, int(n))
  return list(items[-n:])


def sample_phase_points(items: List[Any], n: int) -> List[Any]:
  """按动作过程阶段尽量均匀地抽取代表性样本。"""
  return sample_evenly(items, n)


def make_montage(images: List[np.ndarray], target_h: int = 320) -> Optional[np.ndarray]:
  """把多张图按统一高度拼接成一张横向拼图。"""
  imgs = [im for im in images if im is not None]
  if not imgs:
    return None
  resized = []
  for im in imgs:
    h, w = im.shape[:2]
    if h <= 0 or w <= 0:
      continue
    new_w = max(1, int(w * target_h / h))
    resized.append(cv2.resize(im, (new_w, target_h)))
  if not resized:
    return None
  return cv2.hconcat(resized)


def _resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
  """把图像最长边限制在指定大小以内。"""
  if img is None:
    return img
  if max_side <= 0:
    return img
  h, w = img.shape[:2]
  m = max(h, w)
  if m <= max_side:
    return img
  scale = max_side / float(m)
  nw = max(1, int(round(w * scale)))
  nh = max(1, int(round(h * scale)))
  return cv2.resize(img, (nw, nh))


def normalize_box(box, frame_w: int, frame_h: int):
  """把检测框坐标归一化到 [0, 1] 范围。"""
  if box is None or frame_w <= 0 or frame_h <= 0:
    return None
  x1, y1, x2, y2 = [float(x) for x in box]
  return {
    'x1': round(x1 / frame_w, 4),
    'y1': round(y1 / frame_h, 4),
    'x2': round(x2 / frame_w, 4),
    'y2': round(y2 / frame_h, 4),
    'w': round((x2 - x1) / frame_w, 4),
    'h': round((y2 - y1) / frame_h, 4),
  }


def pose_summary(kpts, box, frame_w: int, frame_h: int, min_conf: float = 0.25, max_points: int = 17) -> Dict[str, Any]:
  """把关键点和框整理成更紧凑、更适合传给 VLM 的结构化摘要。"""
  out: Dict[str, Any] = {
    'detected': bool(kpts is not None),
    'box_norm': normalize_box(box, frame_w, frame_h),
    'num_points_above_conf': 0,
    'min_conf': float(min_conf),
    'points_norm': [],
  }
  if kpts is None:
    return out
  pts = []
  cnt = 0
  for i in range(min(int(max_points), int(kpts.shape[0]))):
    x, y, c = float(kpts[i, 0]), float(kpts[i, 1]), float(kpts[i, 2])
    if c >= min_conf:
      cnt += 1
    pts.append({
      'idx': int(i),
      'x': round(x / frame_w, 4) if frame_w > 0 else None,
      'y': round(y / frame_h, 4) if frame_h > 0 else None,
      'conf': round(c, 4),
    })
  out['num_points_above_conf'] = int(cnt)
  out['points_norm'] = pts
  return out


def as_json_block(title: str, obj: Any) -> str:
  """把对象格式化成带标题的 JSON 文本块。"""
  return f'[{title}]\n' + json.dumps(obj, ensure_ascii=False, indent=2)


def build_rt_review_json(exercise: str, rep_count: int, goal_reps: int, rule_tip: str, frame_issues: List[str], coach_src: str) -> Dict[str, Any]:
  """构造实时链路 review_json。"""
  return {
    'exercise': exercise,
    'rep_count': int(rep_count),
    'goal_reps': int(goal_reps),
    'rule_tip': str(rule_tip or ''),
    'frame_issues': list(frame_issues or []),
    'coach_src': str(coach_src or ''),
  }


def build_posthoc_review_json(summary: Dict[str, Any], engineering: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
  """构造课后链路 review_json。"""
  out = {
    'engine': summary.get('engine'),
    'rep_count': summary.get('rep_count'),
    'avg_score': summary.get('avg_score'),
    'issue_counts': summary.get('issue_counts'),
    'groups': summary.get('groups'),
    'worst_reps': summary.get('worst_reps'),
    'best_reps': summary.get('best_reps'),
    'metrics_summary': summary.get('metrics_summary'),
    'reps': summary.get('reps'),
    'user_mode': summary.get('user_mode'),
  }
  if engineering is not None:
    out['engineering'] = {
      'fps': engineering.get('fps'),
      'duration_s': engineering.get('duration_s'),
      'resolution': engineering.get('resolution'),
      'config_digest': engineering.get('config_digest'),
      'view_check': engineering.get('view_check'),
    }
  return out


def build_rule_conclusion_text_rt(exercise: str, rep_count: int, goal_reps: int, rule_tip: str, coach_src: str) -> str:
  """构造实时链路的规则结论文本。"""
  return (
    f'[规则判断结论-实时]\n'
    f'动作: {exercise}\n'
    f'当前次数: {rep_count} / 目标: {goal_reps}\n'
    f'来源: {coach_src}\n'
    f'规则原始提示: {rule_tip or "<empty>"}'
  )


def build_rule_conclusion_text_posthoc(summary: Dict[str, Any]) -> str:
  """构造课后链路的规则结论文本。"""
  issues = summary.get('issue_counts', {}) or {}
  groups = summary.get('groups', []) or []
  worst = summary.get('worst_reps', []) or []
  return (
    '[规则判断结论-课后]\n'
    f'动作: {summary.get("engine")}\n'
    f'总次数: {summary.get("rep_count")}\n'
    f'平均质量: {summary.get("avg_score")}\n'
    f'问题统计: {json.dumps(issues, ensure_ascii=False)}\n'
    f'分组结果: {json.dumps(groups, ensure_ascii=False)}\n'
    f'最需要关注的 reps: {json.dumps(worst, ensure_ascii=False)}'
  )


def content_selection(cfg, chain: str) -> Dict[str, Any]:
  """根据配置决定当前链路要向 VLM 暴露哪些证据。"""
  node = cfg.get('vlm', 'inputs', chain, default={}) or {}
  return {
    # task_context：非常紧凑的上下文摘要，例如总次数、平均分、分组编号。
    # 在工程里，通常会关闭它，以保证 VLM 输入更简单。
    'task_context': bool(node.get('task_context', True)),
    'rule_conclusion': bool(node.get('rule_conclusion', True)),
    'yolo_output': bool(node.get('yolo_output', True)),
    'review_json': bool(node.get('review_json', chain == 'posthoc')),
    'raw_frames': bool(node.get('raw_frames', chain == 'posthoc')),
    'raw_frame_count': int(node.get('raw_frame_count', 4 if chain == 'posthoc' else 1)),
    'raw_frame_strategy': str(node.get('raw_frame_strategy', 'phase_points' if chain == 'posthoc' else 'recent_even')),
    'image_delivery': str(node.get('image_delivery', 'both' if chain == 'posthoc' else 'individual')),
    'image_max_side': int(node.get('image_max_side', 1024 if chain == 'posthoc' else 960)),
    'image_jpeg_quality': int(node.get('image_jpeg_quality', 85)),
    'montage_height': int(node.get('montage_height', 320)),
  }


def _select_frames(frames: List[np.ndarray], strategy: str, count: int, current_frame=None) -> List[np.ndarray]:
  """根据抽帧策略选择要送入 VLM 的帧。"""
  seq = list(frames or [])
  if current_frame is not None:
    seq = seq + [current_frame]
  if not seq:
    return []
  if strategy == 'tail':
    return sample_tail(seq, count)
  if strategy == 'current_only':
    return [seq[-1]] if seq else []
  if strategy == 'phase_points':
    return sample_phase_points(seq, count)
  return sample_evenly(seq, count)


def _push_text(items: List[dict], stats: Dict[str, int], part: str, text: str):
  """向 content_items 追加一段文本内容，并记录 token 估计。"""
  if not text:
    return
  items.append({'type': 'text', 'text': text})
  stats[part] = stats.get(part, 0) + estimate_text_tokens(text)


def _push_image(items: List[dict], stats: Dict[str, int], part: str, img: np.ndarray, quality: int):
  """向 content_items 追加一张图片内容，并记录 token 估计。"""
  if img is None:
    return
  h, w = img.shape[:2]
  items.append({'type': 'image_url', 'image_url': {'url': _img_to_data_url_jpeg(img, quality=quality)}})
  stats[part] = stats.get(part, 0) + estimate_image_tokens(w, h)


def build_rt_content_items(
  cfg,
  exercise: str,
  rep_count: int,
  goal_reps: int,
  base_text: str,
  coach_src: str,
  current_frame,
  recent_frames: List[np.ndarray],
  current_kpts,
  current_box,
  frame_w: int,
  frame_h: int,
  min_conf: float,
) -> Tuple[List[dict], Dict[str, Any]]:
  """构造实时链路发给 VLM 的内容列表。"""
  sel = content_selection(cfg, 'realtime')
  items: List[dict] = []
  token_parts: Dict[str, int] = {}
  evidence = []
  if sel['rule_conclusion']:
    evidence.append('规则结论')
  if sel['yolo_output']:
    evidence.append('YOLO火柴人/框')
  if sel['review_json']:
    evidence.append('复盘JSON')
  if sel['raw_frames']:
    evidence.append('原始帧图像')
  if sel['image_delivery'] in ('stitched', 'both') and sel['raw_frames']:
    evidence.append('拼接图')
  task_text = (
    '[实时教练任务]\n'
    f'当前可用证据: {", ".join(evidence) if evidence else "仅系统指令"}\n'
    '请根据提供的证据，把提示改写成更自然的一句话或两句话；'
    '如果看不到足够证据，就优先保留原始规则含义，不要瞎猜。'
  )
  _push_text(items, token_parts, 'task_prompt', task_text)

  if sel['rule_conclusion']:
    _push_text(items, token_parts, 'rule_conclusion', build_rule_conclusion_text_rt(exercise, rep_count, goal_reps, base_text, coach_src))

  if sel['review_json']:
    rt_json = build_rt_review_json(exercise, rep_count, goal_reps, base_text, [], coach_src)
    _push_text(items, token_parts, 'review_json', as_json_block('当前状态JSON', rt_json))

  if sel['yolo_output']:
    yolo_obj = pose_summary(current_kpts, current_box, frame_w, frame_h, min_conf=min_conf)
    _push_text(items, token_parts, 'yolo_output', as_json_block('YOLO输出', yolo_obj))

  if sel['raw_frames']:
    imgs = _select_frames(recent_frames, sel['raw_frame_strategy'], sel['raw_frame_count'], current_frame=current_frame)
    imgs = [_resize_max_side(img, sel['image_max_side']) for img in imgs]
    if sel['image_delivery'] in ('individual', 'both'):
      for img in imgs:
        _push_image(items, token_parts, 'raw_frames', img, sel['image_jpeg_quality'])
    if sel['image_delivery'] in ('stitched', 'both'):
      montage = make_montage(imgs, target_h=sel['montage_height'])
      if montage is not None:
        _push_image(items, token_parts, 'stitched_image', montage, sel['image_jpeg_quality'])

  meta = {'inputs': sel, 'prompt_token_parts_est': token_parts}
  return items, meta


def build_posthoc_content_items(
  cfg,
  summary: Dict[str, Any],
  engineering: Dict[str, Any],
  raw_frames: List[np.ndarray],
  raw_pose_summaries: List[Dict[str, Any]],
  kb_context: str,
) -> Tuple[List[dict], Dict[str, Any]]:
  """
  构造课后链路发给 VLM 的内容列表。
  
  在工程中，默认只保留 yolo_output 这一路主要证据。
  """
  sel = content_selection(cfg, 'posthoc')
  items: List[dict] = []
  token_parts: Dict[str, int] = {}
  evidence = []
  if sel['rule_conclusion']:
    evidence.append('规则结论')
  if sel['review_json']:
    evidence.append('复盘JSON')
  if sel['yolo_output']:
    evidence.append('YOLO火柴人/框')
  if sel['raw_frames']:
    evidence.append('原始帧图像')
  if sel['image_delivery'] in ('stitched', 'both') and sel['raw_frames']:
    evidence.append('拼接图')
  if kb_context:
    evidence.append('RAG知识库')
  actual_groups = [str(g.get('group')) for g in (summary.get('groups') or []) if g.get('group')]
  actual_tags = [str(k) for k in (summary.get('issue_counts') or {}).keys()]
  task_text = (
    '[课后复盘任务]\n'
    f'当前可用证据: {", ".join(evidence) if evidence else "仅系统指令"}\n'
    '请尽量让报告中的主要文字模块由你来生成：总体评价、分组点评、关键问题与建议、下次目标。'
    '如果某类证据未提供，请不要假装看到了。\n'
    '禁止输出底层字段术语给普通用户，例如：帧123、207px、x=0.32、y=0.18、ratio=1.24。'
    '请改写成日常语言，例如：手抬得不够高、脚打开幅度偏小、后半段更稳定。'
    f'\n只允许引用这些分组: {actual_groups if actual_groups else []}'
    f'\n只允许引用这些问题标签: {actual_tags if actual_tags else []}'
  )
  _push_text(items, token_parts, 'task_prompt', task_text)

  if sel['task_context']:
    compact_context = {
      'engine': summary.get('engine'),
      'rep_count': summary.get('rep_count'),
      'groups': actual_groups,
      'issue_tags': actual_tags,
      'avg_score': summary.get('avg_score'),
    }
    _push_text(items, token_parts, 'task_context', as_json_block('任务上下文', compact_context))

  if sel['rule_conclusion']:
    _push_text(items, token_parts, 'rule_conclusion', build_rule_conclusion_text_posthoc(summary))

  if sel['review_json']:
    review_obj = build_posthoc_review_json(summary, engineering)
    _push_text(items, token_parts, 'review_json', as_json_block('复盘JSON', review_obj))

  if sel['yolo_output'] and raw_pose_summaries:
    yolo_block = {'sampled_frames': len(raw_pose_summaries), 'frames': raw_pose_summaries}
    _push_text(items, token_parts, 'yolo_output', as_json_block('YOLO输出', yolo_block))

  if kb_context:
    _push_text(items, token_parts, 'rag_context', '[专业参考知识库]\n' + kb_context)

  if sel['raw_frames']:
    imgs = _select_frames(raw_frames, sel['raw_frame_strategy'], sel['raw_frame_count'])
    imgs = [_resize_max_side(img, sel['image_max_side']) for img in imgs]
    if sel['image_delivery'] in ('individual', 'both'):
      for img in imgs:
        _push_image(items, token_parts, 'raw_frames', img, sel['image_jpeg_quality'])
    if sel['image_delivery'] in ('stitched', 'both'):
      montage = make_montage(imgs, target_h=sel['montage_height'])
      if montage is not None:
        _push_image(items, token_parts, 'stitched_image', montage, sel['image_jpeg_quality'])

  meta = {'inputs': sel, 'prompt_token_parts_est': token_parts}
  return items, meta
