"""Demo 主流程：读取视频、YOLO 推理、规则判断、可视化与报告输出。"""

from __future__ import annotations

import json
import re
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, Future

import cv2
import numpy as np

from .config_loader import DemoConfig
from .smoothing import EMASmoother
from .text_render import put_text_unicode
from .yolo_pose import YoloPoseFrontend
from .kb import build_vlm_kb_context
from .vlm_api import VLMConfig, safe_call_vlm_content
from .token_stats import accumulate_usage, empty_usage_stats, finalize_usage, estimate_text_tokens
from .view_heuristics import infer_view
from .vlm_inputs import (
  build_posthoc_content_items,
  build_rt_content_items,
  pose_summary,
)
from .report import summarize_reps, save_report
from .coach_policy import CoachPolicy

from .engines import (
  SquatEngine, SquatParams,
  JumpingJackEngine, JumpingJackParams,
  CurlEngine, CurlParams,
  PushupEngine, PushupParams,
  BenchPressEngine, BenchPressParams,
  CrunchEngine, CrunchParams,
)

SKELETON_EDGES = [
  (5, 7), (7, 9),
  (6, 8), (8, 10),
  (5, 6),
  (5, 11), (6, 12),
  (11, 12),
  (11, 13), (13, 15),
  (12, 14), (14, 16),
]


def _draw_skeleton(frame_bgr, kpts, min_conf: float):
  """在当前帧上绘制人体骨架。"""
  if kpts is None:
    return
  for a, b in SKELETON_EDGES:
    if float(kpts[a, 2]) < min_conf or float(kpts[b, 2]) < min_conf:
      continue
    ax, ay = int(kpts[a, 0]), int(kpts[a, 1])
    bx, by = int(kpts[b, 0]), int(kpts[b, 1])
    cv2.line(frame_bgr, (ax, ay), (bx, by), (0, 255, 0), 2)
  for i in range(kpts.shape[0]):
    if float(kpts[i, 2]) < min_conf:
      continue
    x, y = int(kpts[i, 0]), int(kpts[i, 1])
    cv2.circle(frame_bgr, (x, y), 3, (255, 0, 0), -1)


def _has_any(kpts, idxs, min_conf: float) -> bool:
  """判断一组关键点里是否至少有一个点有效。"""
  if kpts is None:
    return False
  for i in idxs:
    try:
      if float(kpts[i, 2]) >= min_conf:
        return True
    except Exception:
      continue
  return False


def _bbox_ratio(kpts, min_conf: float, frame_w: int, frame_h: int) -> Optional[float]:
  """估算人物框占画面的比例，用于初始化提示。"""
  if kpts is None:
    return None
  xs, ys = [], []
  for i in range(kpts.shape[0]):
    if float(kpts[i, 2]) >= min_conf:
      xs.append(float(kpts[i, 0]))
      ys.append(float(kpts[i, 1]))
  if not xs or not ys or frame_h <= 0:
    return None
  h = max(ys) - min(ys)
  return float(h / frame_h)


def _init_check_lines(exercise: str, kpts, min_conf: float, frame_w: int, frame_h: int, view_info: Optional[dict] = None):
  """生成视频开头若干秒的初始化检查提示。"""
  from . import coco17
  lines = []
  if exercise == 'squat':
    ok = _has_any(kpts, [coco17.L_HIP, coco17.R_HIP], min_conf) and _has_any(kpts, [coco17.L_KNEE, coco17.R_KNEE], min_conf) and _has_any(kpts, [coco17.L_ANKLE, coco17.R_ANKLE], min_conf)
    lines.append(f"InitCheck-下半身: {'OK' if ok else '缺点(髋/膝/踝)'}")
  elif exercise == 'curl':
    ok = _has_any(kpts, [coco17.L_SHOULDER, coco17.R_SHOULDER], min_conf) and _has_any(kpts, [coco17.L_ELBOW, coco17.R_ELBOW], min_conf) and _has_any(kpts, [coco17.L_WRIST, coco17.R_WRIST], min_conf)
    lines.append(f"InitCheck-上半身: {'OK' if ok else '缺点(肩/肘/腕)'}")
  elif exercise == 'jumping_jack':
    ok = _has_any(kpts, [coco17.NOSE, coco17.L_EYE, coco17.R_EYE], min_conf) and _has_any(kpts, [coco17.L_SHOULDER, coco17.R_SHOULDER], min_conf) and _has_any(kpts, [coco17.L_HIP, coco17.R_HIP], min_conf) and _has_any(kpts, [coco17.L_ANKLE, coco17.R_ANKLE], min_conf)
    lines.append(f"InitCheck-全身: {'OK' if ok else '缺点(头/肩/髋/踝)'}")
  elif exercise == 'pushup':
    ok = _has_any(kpts, [coco17.L_SHOULDER, coco17.R_SHOULDER], min_conf) and _has_any(kpts, [coco17.L_ELBOW, coco17.R_ELBOW], min_conf) and _has_any(kpts, [coco17.L_WRIST, coco17.R_WRIST], min_conf) and _has_any(kpts, [coco17.L_HIP, coco17.R_HIP], min_conf) and _has_any(kpts, [coco17.L_ANKLE, coco17.R_ANKLE], min_conf)
    lines.append(f"InitCheck-支撑链: {'OK' if ok else '缺点(肩/肘/腕/髋/踝)'}")
  elif exercise == 'bench_press':
    ok = _has_any(kpts, [coco17.L_SHOULDER, coco17.R_SHOULDER], min_conf) and _has_any(kpts, [coco17.L_ELBOW, coco17.R_ELBOW], min_conf) and _has_any(kpts, [coco17.L_WRIST, coco17.R_WRIST], min_conf) and _has_any(kpts, [coco17.L_HIP, coco17.R_HIP], min_conf)
    lines.append(f"InitCheck-卧推链: {'OK' if ok else '缺点(肩/肘/腕/髋)'}")
  elif exercise == 'crunch':
    ok = _has_any(kpts, [coco17.L_SHOULDER, coco17.R_SHOULDER], min_conf) and _has_any(kpts, [coco17.L_HIP, coco17.R_HIP], min_conf) and _has_any(kpts, [coco17.L_KNEE, coco17.R_KNEE], min_conf)
    lines.append(f"InitCheck-卷腹链: {'OK' if ok else '缺点(肩/髋/膝)'}")
  br = _bbox_ratio(kpts, min_conf, frame_w, frame_h)
  if br is not None:
    if br < 0.45:
      lines.append('距离: 有点远（建议靠近一点）')
    elif br > 0.95:
      lines.append('距离: 有点近（建议稍微退远）')
    else:
      lines.append('距离: OK')
  if view_info is not None:
    label = {'front_like': '偏正面', 'side_like': '偏侧面', 'uncertain': '不确定'}.get(str(view_info.get('type','uncertain')), '不确定')
    conf = float(view_info.get('confidence', 0.0))
    valid = bool(view_info.get('valid_for_exercise', False))
    lines.append(f"视角实验: {label} conf={conf:.2f} | {'适合当前动作' if valid else '可能不太适合'}")
  return lines


def _digest_cfg(cfg: DemoConfig, exercise: str) -> str:
  """把当前动作相关配置压缩成一行摘要，便于写入日志和报告。"""
  g = cfg.get('global', default={}) or {}
  c = cfg.get('coach', default={}) or {}
  e = cfg.get(exercise, default={}) or {}
  v = cfg.get('vlm', 'inputs', default={}) or {}
  return (
    f"min_conf={g.get('min_kpt_conf')} smooth_alpha={g.get('smooth_alpha')} coach_mode={c.get('mode')} "
    f"group_n={c.get('group_n')} interval_s={c.get('interval_s')} thresholds={e} view_exp={g.get('view_experiment_enabled', True)} posthoc_vlm_inputs={v.get('posthoc', {})}"
  )


def _ensure_out_dir(video_path: str, exercise: str, config_name: str) -> tuple[Path, str]:
  """创建当前视频对应的输出目录。"""
  out_root = Path('outputs')
  out_root.mkdir(parents=True, exist_ok=True)
  stem = Path(video_path).stem
  ts = datetime.now().strftime('%Y%m%d_%H%M%S')
  cfg_name = ''.join(ch if (ch.isalnum() or ch in ('-', '_')) else '_' for ch in (config_name or 'default'))
  out_dir = out_root / f"{stem}_{exercise}_{cfg_name}_{ts}"
  out_dir.mkdir(parents=True, exist_ok=True)
  return out_dir, ts





def _safe_tag(name: str) -> str:
  """把字符串清洗成更适合文件名/目录名使用的形式。"""
  s = ''.join(ch if (ch.isalnum() or ch in ('-', '_')) else '_' for ch in (name or '').strip())
  return s[:80] if s else 'unknown'


def _make_engine(exercise: str, cfg: DemoConfig):
  """根据动作类型和配置实例化对应规则引擎。"""
  min_conf = float(cfg.get('global', 'min_kpt_conf', default=0.25))
  if exercise == 'squat':
    p = SquatParams(
      knee_start_max_deg=float(cfg.get('squat', 'knee_start_max_deg', default=160.0)),
      knee_bottom_hint_deg=float(cfg.get('squat', 'knee_bottom_hint_deg', default=135.0)),
      knee_top_ready_deg=float(cfg.get('squat', 'knee_top_ready_deg', default=145.0)),
      hip_drop_start_ratio=float(cfg.get('squat', 'hip_drop_start_ratio', default=0.08)),
      hip_drop_bottom_ratio=float(cfg.get('squat', 'hip_drop_bottom_ratio', default=0.16)),
      hip_drop_return_ratio=float(cfg.get('squat', 'hip_drop_return_ratio', default=0.10)),
      torso_lean_warn_deg=float(cfg.get('squat', 'torso_lean_warn_deg', default=28.0)),
      grade_perfect_low_deg=float(cfg.get('squat', 'grade_perfect_low_deg', default=82.0)),
      grade_perfect_high_deg=float(cfg.get('squat', 'grade_perfect_high_deg', default=102.0)),
      grade_good_low_deg=float(cfg.get('squat', 'grade_good_low_deg', default=72.0)),
      grade_good_high_deg=float(cfg.get('squat', 'grade_good_high_deg', default=112.0)),
      bottom_window_deg=float(cfg.get('squat', 'bottom_window_deg', default=140.0)),
      min_conf=min_conf,
    )
    return SquatEngine(p, cooldown_s=float(cfg.get('squat', 'cooldown_s', default=0.9)))
  if exercise == 'jumping_jack':
    p = JumpingJackParams(
      detect_open_hand_ratio=float(cfg.get('jumping_jack', 'detect_open_hand_ratio', default=0.55)),
      detect_open_feet_ratio_sw=float(cfg.get('jumping_jack', 'detect_open_feet_ratio_sw', default=0.95)),
      detect_open_entry_hand_ratio=float(cfg.get('jumping_jack', 'detect_open_entry_hand_ratio', default=0.35)),
      detect_open_entry_feet_ratio_sw=float(cfg.get('jumping_jack', 'detect_open_entry_feet_ratio_sw', default=0.70)),
      detect_close_hand_ratio=float(cfg.get('jumping_jack', 'detect_close_hand_ratio', default=0.30)),
      detect_close_feet_ratio_sw=float(cfg.get('jumping_jack', 'detect_close_feet_ratio_sw', default=0.80)),
      grade_open_hand_perfect_low=float(cfg.get('jumping_jack', 'grade_open_hand_perfect_low', default=0.85)),
      grade_open_hand_perfect_high=float(cfg.get('jumping_jack', 'grade_open_hand_perfect_high', default=1.15)),
      grade_open_feet_perfect_min=float(cfg.get('jumping_jack', 'grade_open_feet_perfect_min', default=1.40)),
      grade_close_hand_perfect_max=float(cfg.get('jumping_jack', 'grade_close_hand_perfect_max', default=0.15)),
      grade_close_feet_perfect_max=float(cfg.get('jumping_jack', 'grade_close_feet_perfect_max', default=0.40)),
      grade_open_hand_good_low=float(cfg.get('jumping_jack', 'grade_open_hand_good_low', default=0.70)),
      grade_open_hand_good_high=float(cfg.get('jumping_jack', 'grade_open_hand_good_high', default=1.20)),
      grade_open_feet_good_min=float(cfg.get('jumping_jack', 'grade_open_feet_good_min', default=1.05)),
      grade_close_hand_good_max=float(cfg.get('jumping_jack', 'grade_close_hand_good_max', default=0.25)),
      grade_close_feet_good_max=float(cfg.get('jumping_jack', 'grade_close_feet_good_max', default=0.75)),
      min_conf=min_conf,
    )
    return JumpingJackEngine(p, cooldown_s=float(cfg.get('jumping_jack', 'cooldown_s', default=0.6)))
  if exercise == 'curl':
    p = CurlParams(
      lift_entry_elbow_deg=float(cfg.get('curl', 'lift_entry_elbow_deg', default=138.0)),
      top_hint_elbow_deg=float(cfg.get('curl', 'top_hint_elbow_deg', default=112.0)),
      return_ready_elbow_deg=float(cfg.get('curl', 'return_ready_elbow_deg', default=138.0)),
      top_wrist_lift_ratio=float(cfg.get('curl', 'top_wrist_lift_ratio', default=0.12)),
      return_wrist_lift_ratio=float(cfg.get('curl', 'return_wrist_lift_ratio', default=0.05)),
      grade_perfect_max_elbow_deg=float(cfg.get('curl', 'grade_perfect_max_elbow_deg', default=35.0)),
      grade_good_max_elbow_deg=float(cfg.get('curl', 'grade_good_max_elbow_deg', default=55.0)),
      elbow_drift_warn_ratio=float(cfg.get('curl', 'elbow_drift_warn_ratio', default=0.12)),
      trunk_swing_warn_deg=float(cfg.get('curl', 'trunk_swing_warn_deg', default=14.0)),
      min_conf=min_conf,
    )
    return CurlEngine(p, cooldown_s=float(cfg.get('curl', 'cooldown_s', default=0.8)))
  if exercise == 'pushup':
    p = PushupParams(
      elbow_down_deg=float(cfg.get('pushup', 'elbow_down_deg', default=95.0)),
      elbow_up_deg=float(cfg.get('pushup', 'elbow_up_deg', default=160.0)),
      depth_warn_elbow_deg=float(cfg.get('pushup', 'depth_warn_elbow_deg', default=105.0)),
      body_bend_warn_deg=float(cfg.get('pushup', 'body_bend_warn_deg', default=18.0)),
      min_conf=min_conf,
    )
    return PushupEngine(p, cooldown_s=float(cfg.get('pushup', 'cooldown_s', default=0.8)))
  if exercise == 'bench_press':
    p = BenchPressParams(
      start_ready_elbow_deg=float(cfg.get('bench_press', 'start_ready_elbow_deg', default=150.0)),
      lower_entry_elbow_deg=float(cfg.get('bench_press', 'lower_entry_elbow_deg', default=138.0)),
      bottom_hint_elbow_deg=float(cfg.get('bench_press', 'bottom_hint_elbow_deg', default=100.0)),
      press_finish_elbow_deg=float(cfg.get('bench_press', 'press_finish_elbow_deg', default=156.0)),
      grade_perfect_bottom_low_deg=float(cfg.get('bench_press', 'grade_perfect_bottom_low_deg', default=78.0)),
      grade_perfect_bottom_high_deg=float(cfg.get('bench_press', 'grade_perfect_bottom_high_deg', default=98.0)),
      grade_good_bottom_low_deg=float(cfg.get('bench_press', 'grade_good_bottom_low_deg', default=68.0)),
      grade_good_bottom_high_deg=float(cfg.get('bench_press', 'grade_good_bottom_high_deg', default=110.0)),
      grade_perfect_lockout_min_deg=float(cfg.get('bench_press', 'grade_perfect_lockout_min_deg', default=166.0)),
      grade_good_lockout_min_deg=float(cfg.get('bench_press', 'grade_good_lockout_min_deg', default=156.0)),
      bottom_forearm_vertical_warn_ratio=float(cfg.get('bench_press', 'bottom_forearm_vertical_warn_ratio', default=0.12)),
      hip_bridge_warn_ratio=float(cfg.get('bench_press', 'hip_bridge_warn_ratio', default=0.09)),
      tempo_fast_warn_s=float(cfg.get('bench_press', 'tempo_fast_warn_s', default=0.75)),
      wrist_drop_bottom_ratio=float(cfg.get('bench_press', 'wrist_drop_bottom_ratio', default=0.20)),
      wrist_drop_return_ratio=float(cfg.get('bench_press', 'wrist_drop_return_ratio', default=0.10)),
      min_conf=min_conf,
    )
    return BenchPressEngine(p, cooldown_s=float(cfg.get('bench_press', 'cooldown_s', default=0.8)))
  if exercise == 'crunch':
    p = CrunchParams(
      start_ready_torso_deg=float(cfg.get('crunch', 'start_ready_torso_deg', default=22.0)),
      curl_entry_torso_deg=float(cfg.get('crunch', 'curl_entry_torso_deg', default=28.0)),
      top_torso_deg=float(cfg.get('crunch', 'top_torso_deg', default=40.0)),
      return_ready_torso_deg=float(cfg.get('crunch', 'return_ready_torso_deg', default=24.0)),
      grade_perfect_top_low_deg=float(cfg.get('crunch', 'grade_perfect_top_low_deg', default=38.0)),
      grade_perfect_top_high_deg=float(cfg.get('crunch', 'grade_perfect_top_high_deg', default=65.0)),
      grade_good_top_low_deg=float(cfg.get('crunch', 'grade_good_top_low_deg', default=30.0)),
      grade_good_top_high_deg=float(cfg.get('crunch', 'grade_good_top_high_deg', default=75.0)),
      hip_drift_warn_ratio=float(cfg.get('crunch', 'hip_drift_warn_ratio', default=0.10)),
      knee_drift_warn_ratio=float(cfg.get('crunch', 'knee_drift_warn_ratio', default=0.12)),
      momentum_warn_deg_s=float(cfg.get('crunch', 'momentum_warn_deg_s', default=125.0)),
      tempo_fast_warn_s=float(cfg.get('crunch', 'tempo_fast_warn_s', default=0.65)),
      min_conf=min_conf,
    )
    return CrunchEngine(p, cooldown_s=float(cfg.get('crunch', 'cooldown_s', default=0.8)))
  raise ValueError(f'Unknown exercise: {exercise}')




def _extract_json_obj(text: str) -> Optional[dict]:
  """尽量从 VLM 返回文本中提取一个 JSON 对象。"""
  if not text:
    return None
  txt = text.strip()
  try:
    obj = json.loads(txt)
    if isinstance(obj, dict):
      return obj
  except Exception:
    pass
  m = re.search(r'\{.*\}', txt, re.S)
  if m:
    try:
      obj = json.loads(m.group(0))
      if isinstance(obj, dict):
        return obj
    except Exception:
      return None
  return None


def _sanitize_vlm_plain_text(text: str, valid_groups: Optional[List[str]] = None) -> str:
  """清洗 VLM 纯文本输出，避免格式异常和非法分组名。"""
  if not text:
    return text
  out = str(text)
  out = re.sub(r'帧\s*\d+', '某一段动作', out)
  out = re.sub(r'\b\d+(?:\.\d+)?\s*px\b', '较小的幅度', out)
  out = re.sub(r'\b[xy]\s*=\s*\d+(?:\.\d+)?\b', '', out)
  out = re.sub(r'\bratio\s*=\s*\d+(?:\.\d+)?\b', '', out)
  out = re.sub(r'\s{2,}', ' ', out).strip()

  if valid_groups:
    valid = {str(g).replace(' ', '') for g in valid_groups}

    def _repl_group(m):
      """把文本里的组号引用限制在当前报告真实存在的分组集合内。"""
      raw = m.group(1).replace(' ', '')
      return m.group(0) if raw in valid else '后面的几组'

    out = re.sub(r'第?\s*(\d+\s*-\s*\d+)\s*组', _repl_group, out)
  return out


def _sanitize_vlm_sections(obj: dict, summary: Dict[str, Any]) -> dict:
  """清洗 VLM 返回的结构化 sections。"""
  if not isinstance(obj, dict):
    return obj
  valid_groups = [str(g.get('group')) for g in (summary.get('groups') or []) if g.get('group')]
  valid_tags = set((summary.get('issue_counts') or {}).keys())

  out = dict(obj)
  for key in ('overall_comment', 'next_goal', 'coach_summary'):
    if isinstance(out.get(key), str):
      out[key] = _sanitize_vlm_plain_text(out[key], valid_groups=valid_groups)

  g_in = out.get('group_comments')
  if isinstance(g_in, dict):
    g_out = {}
    for g in valid_groups:
      txt = g_in.get(g)
      if isinstance(txt, str) and txt.strip():
        g_out[g] = _sanitize_vlm_plain_text(txt, valid_groups=valid_groups)
    out['group_comments'] = g_out

  issue_list = out.get('issue_analysis')
  if isinstance(issue_list, list):
    cleaned = []
    for item in issue_list:
      if not isinstance(item, dict):
        continue
      tag = str(item.get('tag') or item.get('issue') or '').strip()
      if tag and valid_tags and tag not in valid_tags:
        continue
      txt = str(item.get('text') or item.get('advice') or '').strip()
      txt = _sanitize_vlm_plain_text(txt, valid_groups=valid_groups)
      if txt:
        item2 = dict(item)
        item2['tag'] = tag or item2.get('tag') or item2.get('issue') or 'issue'
        item2['text'] = txt
        cleaned.append(item2)
    out['issue_analysis'] = cleaned
  return out


def run_demo(
  video_path: str,
  exercise: str,
  goal_reps: int,
  cfg: DemoConfig,
  show: bool = True,
  save_video: bool = False,
  yolo_model: str = 'yolov8n-pose.pt',
  out_dir: Optional[str] = None,
  realtime_sync: Optional[bool] = None,
  user_mode: Optional[str] = None,
  coach_mode: Optional[str] = None,
  coach_group_n: Optional[int] = None,
  coach_interval_s: Optional[float] = None,
  tip_hold_s: Optional[float] = None,
  enable_vlm_rt: bool = False,
  vlm_min_interval_s: float = 3.0,
  enable_vlm_posthoc: bool = False,
  vlm_rag: bool = True,
  vlm_debug: bool = False,
  terminal_verbose: Optional[bool] = None,
) -> Path:
  """
  运行完整 Demo 流程。
  
  参数说明：
  - video_path：输入视频路径
  - exercise：动作类型
  - cfg：全局配置
  - show / save_video：是否显示预览、是否保存可视化视频
  - enable_vlm_rt / enable_vlm_posthoc：是否开启实时/课后 VLM
  
  主要阶段：
  1. 逐帧 YOLO Pose 推理
  2. 关键点平滑
  3. 规则引擎计数与打分
  4. 实时提示与可视化
  5. 课后汇总与可选 VLM 复盘
  """
  video_path = str(video_path)
  if out_dir is None:
    outp, ts_tag = _ensure_out_dir(video_path, exercise, cfg.name)
  else:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    ts_tag = datetime.now().strftime('%Y%m%d_%H%M%S')

  if realtime_sync is None:
    realtime_sync = bool(cfg.get('global', 'realtime_sync', default=False))
  if user_mode is None:
    # 当前默认使用 simple，但这里仍保留兜底逻辑，
    # 这样这个内部函数仍然可以被其他调用方复用。
    user_mode = str(cfg.get('global', 'user_mode', default='simple'))
  if coach_mode is None:
    coach_mode = str(cfg.get('coach', 'mode', default='event'))
  if coach_group_n is None:
    coach_group_n = int(cfg.get('coach', 'group_n', default=3))
  if coach_interval_s is None:
    coach_interval_s = float(cfg.get('coach', 'interval_s', default=4.0))
  if tip_hold_s is None:
    tip_hold_s = float(cfg.get('coach', 'tip_hold_s', default=3.0))
  if terminal_verbose is None:
    terminal_verbose = bool(cfg.get('global', 'terminal_verbose', default=False))

  def log(msg: str):
    """内部辅助函数。"""
    if terminal_verbose:
      print(msg, flush=True)

  min_conf = float(cfg.get('global', 'min_kpt_conf', default=0.25))
  smooth_alpha = float(cfg.get('global', 'smooth_alpha', default=0.35))
  init_check_s = float(cfg.get('global', 'init_check_seconds', default=3.0) or 3.0)

  # 1) 初始化核心组件：规则引擎、教练策略、YOLO 前端和平滑器。
  engine = _make_engine(exercise, cfg)
  coach = CoachPolicy(mode=coach_mode, group_n=coach_group_n, interval_s=coach_interval_s, goal_reps=goal_reps, user_mode=user_mode)
  yolo = YoloPoseFrontend(model_name=yolo_model, device='cpu', conf=0.25, imgsz=640)
  smoother = EMASmoother(alpha=smooth_alpha)
  log(f"[INIT] yolo_model={yolo_model} | config={cfg.name} | coach_mode={coach_mode}")

  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    raise RuntimeError(f'Cannot open video: {video_path}')
  fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
  w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
  h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
  frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
  log(f"[INIT] video={video_path} | fps={fps:.1f} | resolution={w}x{h} | frames={frames} | goal={goal_reps}")

  writer = None
  if save_video:
    out_video = outp / f"{exercise}_{cfg.name}_{ts_tag}_out.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (w, h))

  vlm_cfg = None
  rt_executor: Optional[ThreadPoolExecutor] = None
  rt_future: Optional[Future] = None
  last_vlm_t = -1e9
  vlm_submitted = vlm_calls = vlm_ok = vlm_fail = 0
  vlm_tokens: Dict[str, Any] = empty_usage_stats()
  vlm_last_rt_err = ''
  vlm_last_rt_latency_s = 0.0
  posthoc_submitted = posthoc_calls = posthoc_ok = posthoc_fail = 0
  posthoc_tokens: Dict[str, Any] = empty_usage_stats()
  posthoc_last_err = ''
  posthoc_latency_s = 0.0

  if enable_vlm_rt or enable_vlm_posthoc:
    try:
      vlm_cfg = VLMConfig.from_config(cfg)
      rt_executor = ThreadPoolExecutor(max_workers=1)
      log(f"[VLM] enabled | model={vlm_cfg.model} | rt={enable_vlm_rt} | posthoc={enable_vlm_posthoc}")
    except Exception as e:
      log(f"[VLM] disabled | err={e}")
      vlm_cfg = None

  last_tip = ''
  last_tip_src = 'auto'
  last_tip_expire_t = -1e9

  perf_sum = {k: 0.0 for k in ['yolo_infer_ms', 'smooth_ms', 'engine_ms', 'render_ms', 'total_loop_ms']}
  perf_n = 0
  recent_frames: deque = deque(maxlen=12)
  posthoc_raw_frames: List[np.ndarray] = []
  posthoc_pose_summaries: List[Dict[str, Any]] = []
  view_enabled = bool(cfg.get('global', 'view_experiment_enabled', default=True))
  view_vote_count = {'front_like': 0, 'side_like': 0, 'uncertain': 0}
  view_conf_sum = {'front_like': 0.0, 'side_like': 0.0, 'uncertain': 0.0}
  # 即使课后链路关闭了原始图像输入，我们仍会抽取少量时间点，
  # 以便记录这些代表性帧对应的 YOLO 结构化摘要。
  posthoc_sample_idx = set(np.linspace(0, max(0, frames - 1), int(cfg.get('vlm', 'inputs', 'posthoc', 'raw_frame_count', default=4))).astype(int).tolist()) if frames > 0 else {0}

  start_wall = time.perf_counter()
  frame_idx = 0
  last_kpts = None
  last_box = None

  while True:
    loop_t0 = time.perf_counter()
    ok, frame = cap.read()
    if not ok:
      break
    t_sec = frame_idx / fps if fps > 0 else 0.0

    # 2) 姿态感知：先做 YOLO Pose 推理，得到关键点和人物框。
    t0 = time.perf_counter()
    pose = yolo.infer(frame)
    perf_sum['yolo_infer_ms'] += (time.perf_counter() - t0) * 1000

    # 3) 关键点平滑：尽量减少逐帧抖动对规则阈值的影响。
    t0 = time.perf_counter()
    kpts = smoother.update(pose.kpts)
    perf_sum['smooth_ms'] += (time.perf_counter() - t0) * 1000
    last_kpts, last_box = kpts, pose.box
    cur_view_info = infer_view(exercise, kpts, min_conf=min_conf) if view_enabled else None
    if cur_view_info is not None and t_sec <= init_check_s:
      vt = str(cur_view_info.get('type', 'uncertain'))
      view_vote_count[vt] = view_vote_count.get(vt, 0) + 1
      view_conf_sum[vt] = view_conf_sum.get(vt, 0.0) + float(cur_view_info.get('confidence', 0.0))

    recent_frames.append(frame.copy())
    if frame_idx in posthoc_sample_idx:
      posthoc_raw_frames.append(frame.copy())
      posthoc_pose_summaries.append({
        'frame_idx': int(frame_idx),
        'time_s': round(t_sec, 3),
        'pose': pose_summary(kpts, pose.box, w, h, min_conf=min_conf),
        'view': cur_view_info,
      })

    # 4) 动作规则判断：推进状态机，决定是否完成一次 rep，以及当前有什么问题。
    t0 = time.perf_counter()
    fb = engine.update(kpts, t_sec, fps)
    perf_sum['engine_ms'] += (time.perf_counter() - t0) * 1000

    rep_issues = engine.last_rep.issues if engine.rep_completed and engine.last_rep is not None else None
    rep_grade = engine.last_rep.grade if engine.rep_completed and engine.last_rep is not None else None
    if engine.rep_completed and engine.last_rep is not None:
      log(f"[REP] #{engine.last_rep.rep_idx} | grade={engine.last_rep.grade} | issues={engine.last_rep.issues or ['none']}")
    # 5) 教练策略层：把规则引擎的逐帧信号转成当前应显示的教练文案。
    out = coach.step(
      t=t_sec,
      rep_count=engine.rep_count,
      frame_issues=engine.frame_issues,
      rep_completed=engine.rep_completed,
      rep_issues=rep_issues,
      rep_grade=rep_grade,
      rule_text=fb.text,
    )
    coach_text = out.text
    coach_src = out.src
    if coach_text:
      last_tip = coach_text
      last_tip_src = coach_src
      last_tip_expire_t = t_sec + (1e9 if coach_src in ('group', 'time') else float(tip_hold_s))

    if enable_vlm_rt and vlm_cfg is not None and last_tip and (t_sec - last_vlm_t) >= float(vlm_min_interval_s) and rt_future is None:
      base_text = last_tip
      if str(user_mode) == 'pro':
        sys_p = '你是专业健身教练。输出中文，简洁但专业，最多两句话，不要编造看不到的细节。'
      else:
        sys_p = '你是耐心的健身教练。输出中文，像跟朋友说话一样好懂，最多两句话，不要编造看不到的细节。'
      content_items, rt_meta = build_rt_content_items(
        cfg=cfg,
        exercise=exercise,
        rep_count=engine.rep_count,
        goal_reps=goal_reps,
        base_text=base_text,
        coach_src=coach_src,
        current_frame=frame.copy(),
        recent_frames=list(recent_frames),
        current_kpts=kpts,
        current_box=pose.box,
        frame_w=w,
        frame_h=h,
        min_conf=min_conf,
      )
      def _task_rt(_content_items=content_items, _rt_meta=rt_meta):
        """在线程池里执行一次实时 VLM 调用。"""
        t_call0 = time.perf_counter()
        txt, err, usage = safe_call_vlm_content(vlm_cfg, sys_p, _content_items, temperature=0.2, max_tokens=180)
        return txt, err, usage, time.perf_counter() - t_call0, _rt_meta
      rt_future = rt_executor.submit(_task_rt)
      vlm_submitted += 1
      log(f"[VLM][rt][SUBMIT] t={t_sec:.2f}s | inputs={cfg.get('vlm','inputs','realtime', default={})}")

    if rt_future is not None and rt_future.done():
      txt, err, usage, latency, rt_meta = rt_future.result()
      rt_future = None
      vlm_calls += 1
      vlm_last_rt_latency_s = float(latency)
      if txt and txt.strip():
        vlm_ok += 1
        last_tip = txt.strip().replace('\n', ' ')
        last_tip_src = 'vlm'
        last_tip_expire_t = max(last_tip_expire_t, t_sec + float(tip_hold_s))
        last_vlm_t = t_sec
        if vlm_debug:
          print(f"[VLM][rt][OK] t={t_sec:.2f}s latency={latency:.2f}s -> {last_tip}", flush=True)
      else:
        vlm_fail += 1
        vlm_last_rt_err = str(err or 'empty')[:300]
        if vlm_debug:
          print(f"[VLM][rt][FAIL] t={t_sec:.2f}s err={vlm_last_rt_err}", flush=True)
      accumulate_usage(vlm_tokens, usage, (rt_meta or {}).get('prompt_token_parts_est'), {'generated_text_est': estimate_text_tokens(txt) if txt else 0})

    # 6) 画面渲染：叠加骨架、计数、提示文案等可视化信息。
    t0 = time.perf_counter()
    _draw_skeleton(frame, kpts, min_conf=min_conf)
    if last_tip and t_sec <= last_tip_expire_t:
      coach_line = last_tip
      coach_src2 = last_tip_src
    else:
      coach_line = engine._motivation_line()
      coach_src2 = 'auto'
    lines = [
      f'Exercise: {exercise}',
      f'Rep: {engine.rep_count} / Goal: {goal_reps}',
      f'Coach({coach_src2}): {coach_line}',
      f'CoachMode: {coach_mode}',
      f'VLM: RT={"ON" if enable_vlm_rt else "OFF"} POST={"ON" if enable_vlm_posthoc else "OFF"}',
    ]
    if t_sec <= init_check_s:
      lines.extend(_init_check_lines(exercise, kpts, min_conf, w, h, cur_view_info))
    put_text_unicode(frame, lines, org=(20, 30), font_size=22, with_bg=True)
    perf_sum['render_ms'] += (time.perf_counter() - t0) * 1000

    if writer is not None:
      writer.write(frame)
    if show:
      cv2.imshow('Fitness Coach Demo', frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    perf_sum['total_loop_ms'] += (time.perf_counter() - loop_t0) * 1000
    perf_n += 1
    if realtime_sync and fps > 0:
      target = (frame_idx + 1) / fps
      elapsed = time.perf_counter() - start_wall
      sleep_s = target - elapsed
      if sleep_s > 0:
        time.sleep(sleep_s)
    frame_idx += 1

  total_wall = time.perf_counter() - start_wall
  cap.release()
  if writer is not None:
    writer.release()
  if show:
    try:
      cv2.destroyAllWindows()
    except Exception:
      pass

  if rt_future is not None and (not rt_future.done()):
    try:
      txt, err, usage, latency, rt_meta = rt_future.result(timeout=6.0)
      vlm_calls += 1
      vlm_last_rt_latency_s = float(latency)
      if txt and txt.strip():
        vlm_ok += 1
      else:
        vlm_fail += 1
        vlm_last_rt_err = str(err or 'empty')[:300]
      accumulate_usage(vlm_tokens, usage, (rt_meta or {}).get('prompt_token_parts_est'), {'generated_text_est': estimate_text_tokens(txt) if txt else 0})
    except Exception as e:
      vlm_fail += 1
      vlm_last_rt_err = f'{type(e).__name__}: {e}'[:300]
  if rt_executor is not None:
    try:
      rt_executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
      pass

  # 7) 视频处理结束后，汇总逐次 rep 记录，形成课后报告所需的 summary。
  reps = engine.finalize(total_t=frame_idx / fps if fps > 0 else 0.0)
  summary = summarize_reps(engine=exercise, reps=reps, group_n=coach_group_n, user_mode=str(user_mode))

  dominant_view = max(view_vote_count.items(), key=lambda x: x[1])[0] if sum(view_vote_count.values()) > 0 else 'uncertain'
  dominant_n = max(view_vote_count.values()) if view_vote_count else 0
  dominant_conf = (view_conf_sum.get(dominant_view, 0.0) / max(1, dominant_n)) if dominant_n > 0 else 0.0
  expected_view = 'front_like' if exercise == 'jumping_jack' else 'side_like'
  view_check = {
    'enabled': bool(view_enabled),
    'dominant_view': dominant_view,
    'dominant_confidence': round(float(dominant_conf), 3),
    'votes': view_vote_count,
    'expected': expected_view,
    'valid_for_exercise': bool(dominant_view == expected_view and dominant_conf >= float(cfg.get('global', 'view_conf_threshold', default=0.5))),
  }

  # 8) 整理工程侧元信息：性能、机位判断、输出目录、配置摘要等。
  engineering: Dict[str, Any] = {
    'video': video_path,
    'duration_s': float(frame_idx / fps) if fps > 0 else None,
    'fps': fps,
    'resolution': f'{w}x{h}',
    'frames': frame_idx,
    'config_digest': _digest_cfg(cfg, exercise),
    'view_check': view_check,
    'perf': {k: (perf_sum[k] / max(1, perf_n)) for k in perf_sum},
    'wall_clock_s': float(total_wall),
    'user_mode': str(user_mode),
    'output': {
      'config_name': cfg.name,
      'time_tag': ts_tag,
      'out_dir': str(outp.resolve()),
      'show_kb_appendix': bool(cfg.get('output', 'show_kb_appendix', default=False)),
      'show_scores_parenthetical': bool(cfg.get('output', 'show_scores_parenthetical', default=True)),
    },
  }
  summary['goal_reps'] = int(goal_reps)
  summary['coach_mode'] = str(coach_mode)
  summary['coach_group_n'] = int(coach_group_n)
  summary['coach_interval_s'] = float(coach_interval_s)

  # 9) 如果需要，也可以为课后报告准备知识库上下文。
  max_docs = int(cfg.get('vlm', 'rag_retrieval', 'max_docs', default=8))
  max_chars_per_doc = int(cfg.get('vlm', 'rag_retrieval', 'max_chars_per_doc', default=1200))
  kb_context, kb_docs = build_vlm_kb_context(
    exercise,
    summary.get('issue_counts', {}),
    max_docs=max_docs,
    max_chars_per_doc=max_chars_per_doc,
    include_overview=bool(cfg.get('vlm', 'rag_retrieval', 'include_overview', default=True)),
    include_safety=bool(cfg.get('vlm', 'rag_retrieval', 'include_safety', default=True)),
    include_camera=bool(cfg.get('vlm', 'rag_retrieval', 'include_camera', default=True)),
  ) if vlm_rag else ('', [])
  summary['kb'] = [{'doc_id': d.doc_id, 'title': d.title} for d in kb_docs]
  if bool(cfg.get('output', 'show_kb_appendix', default=False)):
    summary['kb_docs'] = [{'doc_id': d.doc_id, 'title': d.title, 'text': d.text} for d in kb_docs]

  # 课后 VLM 会在整段视频处理完成后只运行一次。
  # 当前默认给 VLM 的证据路线相对收敛，
  # 也就是默认只保留 YOLO 摘要，便于保持接口约定简单清晰。
  if enable_vlm_posthoc and vlm_cfg is not None:
    try:
      if not posthoc_raw_frames:
        # 如果无法得到预期帧数，就从最近若干帧里兜底抽样。
        posthoc_raw_frames = list(recent_frames)
      if str(user_mode) == 'pro':
        sys_p = '你是一个专业健身教练，负责训练复盘。请优先依据输入的复盘JSON、YOLO信息、图像和知识库，生成尽量完整的报告文字模块。只输出JSON，不要额外解释。'
      else:
        sys_p = '你是一个耐心、专业、好懂的健身教练。请优先依据输入的复盘JSON、YOLO信息、图像和知识库，生成尽量完整的报告文字模块。只输出JSON，不要额外解释。'
      dynamic_group_keys = {str(g.get('group')): '对该组的点评' for g in (summary.get('groups') or []) if g.get('group')}
      dynamic_issue_tags = list((summary.get('issue_counts') or {}).keys())
      section_schema = {
        'overall_comment': '总体评价（1段）',
        'group_comments': dynamic_group_keys,
        'issue_analysis': [
          {'tag': dynamic_issue_tags[0] if dynamic_issue_tags else '问题标签', 'count': 2, 'text': '把这个问题解释清楚，并给出可执行建议，尽量引用组或rep作为证据。不要写帧号、像素值或原始坐标。'}
        ],
        'next_goal': '下一次训练的一个可量化目标',
        'coach_summary': '一段更像教练的总结（可稍长）',
      }
      task_text = (
        '[输出要求]\n'
        '请只返回一个JSON对象，键必须包括：overall_comment, group_comments, issue_analysis, next_goal, coach_summary。\n'
        '1) overall_comment：口语化、鼓励为主；如果问题少可以夸奖。\n'
        f'2) group_comments：只能使用这些组号作为key：{list(dynamic_group_keys.keys())}。不要编造不存在的组。\n'
        f'3) issue_analysis：tag只能从这些标签里选：{dynamic_issue_tags}。逐条解释“问题→证据→建议”；如果没有明显问题，可以返回空数组。\n'
        '4) next_goal：给一个简短可量化目标。\n'
        '5) coach_summary：尽量覆盖整体报告里的主要文字内容，风格自然、专业。\n'
        '如果图片、YOLO输出和JSON有冲突，以图片和JSON为准。\n'
        '禁止出现这些底层字段术语：帧号、px、x=、y=、ratio=、bbox。请全部翻译成普通用户能理解的话。\n'
        '[JSON格式示例]\n' + json.dumps(section_schema, ensure_ascii=False, indent=2)
      )
      content_items, posthoc_meta = build_posthoc_content_items(
        cfg=cfg,
        summary=summary,
        engineering=engineering,
        raw_frames=posthoc_raw_frames,
        raw_pose_summaries=posthoc_pose_summaries,
        kb_context=kb_context,
      )
      content_items.insert(0, {'type': 'text', 'text': task_text})
      if isinstance(posthoc_meta, dict):
        bag = posthoc_meta.setdefault('prompt_token_parts_est', {})
        bag['task_schema'] = int(bag.get('task_schema', 0)) + int(estimate_text_tokens(task_text))
      posthoc_submitted += 1
      log(f"[VLM][posthoc][SUBMIT] inputs={cfg.get('vlm','inputs','posthoc', default={})} | kb_docs={len(kb_docs)}")
      _ph_t0 = time.perf_counter()
      txt, err, usage = safe_call_vlm_content(vlm_cfg, sys_p, content_items, temperature=0.2, max_tokens=900)
      posthoc_calls += 1
      posthoc_latency_s = float(time.perf_counter() - _ph_t0)
      accumulate_usage(posthoc_tokens, usage, (posthoc_meta or {}).get('prompt_token_parts_est'), {'generated_text_est': estimate_text_tokens(txt) if txt else 0})
      if txt and txt.strip():
        posthoc_ok += 1
        summary['vlm_posthoc'] = txt.strip()
        obj = _extract_json_obj(txt)
        if isinstance(obj, dict):
          obj = _sanitize_vlm_sections(obj, summary)
          summary['vlm_sections'] = obj
          if isinstance(obj.get('coach_summary'), str):
            summary['vlm_posthoc'] = obj['coach_summary']
        else:
          summary['vlm_posthoc'] = _sanitize_vlm_plain_text(summary['vlm_posthoc'], valid_groups=[str(g.get('group')) for g in (summary.get('groups') or []) if g.get('group')])
        log(f"[VLM][posthoc][OK] latency={posthoc_latency_s:.2f}s | chars={len(txt.strip())}")
      else:
        posthoc_fail += 1
        posthoc_last_err = str(err or 'empty')[:300]
        log(f"[VLM][posthoc][FAIL] latency={posthoc_latency_s:.2f}s | err={posthoc_last_err}")
    except Exception as e:
      posthoc_fail += 1
      posthoc_last_err = f'{type(e).__name__}: {e}'[:300]
      log(f"[VLM][posthoc][EXCEPTION] {posthoc_last_err}")

  # 10) 把 VLM 相关的开关、耗时、token 统计等统一写入工程信息。
  engineering['vlm'] = {
    'enabled': bool(vlm_cfg is not None),
    'model': getattr(vlm_cfg, 'model', '') if vlm_cfg is not None else '',
    'base_url': getattr(vlm_cfg, 'base_url', '') if vlm_cfg is not None else '',
    'realtime': {
      'enabled': bool(enable_vlm_rt),
      'inputs': cfg.get('vlm', 'inputs', 'realtime', default={}) or {},
      'submitted': int(vlm_submitted),
      'finished': int(vlm_calls),
      'ok': int(vlm_ok),
      'fail': int(vlm_fail),
      'last_err': str(vlm_last_rt_err),
      'last_latency_s': float(vlm_last_rt_latency_s),
      'tokens': finalize_usage(vlm_tokens),
    },
    'posthoc': {
      'enabled': bool(enable_vlm_posthoc),
      'inputs': cfg.get('vlm', 'inputs', 'posthoc', default={}) or {},
      'submitted': int(posthoc_submitted),
      'finished': int(posthoc_calls),
      'ok': int(posthoc_ok),
      'fail': int(posthoc_fail),
      'last_err': str(posthoc_last_err),
      'last_latency_s': float(posthoc_latency_s),
      'tokens': finalize_usage(posthoc_tokens),
    },
  }
  summary['engineering'] = engineering

  model_prefix = 'no_vlm'
  if vlm_cfg is not None and (enable_vlm_rt or enable_vlm_posthoc):
    model_prefix = _safe_tag(vlm_cfg.model.lower())
  report_base = f"{model_prefix}_{exercise}_{cfg.name}_{ts_tag}_report"
  out_json = str(outp / f"{report_base}.json")
  out_md = str(outp / f"{report_base}.md")
  save_report(out_json, out_md, summary)
  log(f"[DONE] report={out_md} | reps={summary.get('rep_count')} | avg_score={summary.get('avg_score',0.0):.1f}")
  return outp
