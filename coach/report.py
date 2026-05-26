"""
训练结果汇总、打分与报告生成。

这里会把逐次 rep 结果整理成结构化总结，
再进一步生成 markdown 报告和 json 报告。
"""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

from .engines.base import RepRecord
from .report_text import per_issue_advice, encourage_if_clean, overall_comment, group_comment, grade_cn


def _safe_stats(xs: List[float]) -> Dict[str, float]:
  """对数值列表计算均值、分位数等统计量；空列表时返回安全默认值。"""
  xs = [float(x) for x in xs if x is not None and isinstance(x, (int, float))]
  if not xs:
    return {}
  xs2 = sorted(xs)
  n = len(xs2)

  def pct(p: float) -> float:
    """内部辅助函数。"""
    if n == 1:
      return xs2[0]
    i = p * (n - 1)
    lo = int(i)
    hi = min(n - 1, lo + 1)
    w = i - lo
    return xs2[lo] * (1 - w) + xs2[hi] * w

  out = {
    'min': xs2[0],
    'p25': pct(0.25),
    'median': pct(0.5),
    'p75': pct(0.75),
    'max': xs2[-1],
    'mean': float(statistics.mean(xs2)),
  }
  if n >= 2:
    out['std'] = float(statistics.pstdev(xs2))
  return out


def _clamp01(x: float) -> float:
  """内部辅助函数。"""
  return max(0.0, min(1.0, float(x)))


def score_rep(engine: str, r: RepRecord) -> float:
  """根据动作类型与 rep 记录计算连续质量分数。"""
  m = r.metrics or {}
  g = (r.grade or 'good').lower()

  if engine == 'squat':
    knee = float(m.get('knee_bottom_med_deg', m.get('knee_min_deg', 92.0)))
    if g == 'perfect':
      score = 92.0 + 8.0 * (1.0 - _clamp01(abs(knee - 92.0) / 10.0))
    elif g == 'good':
      dist = 0.0
      if knee < 82.0:
        dist = 82.0 - knee
      elif knee > 102.0:
        dist = knee - 102.0
      score = 78.0 + 14.0 * (1.0 - _clamp01(dist / 15.0))
    else:
      dist = (knee - 112.0) if knee > 112.0 else (72.0 - knee)
      score = max(20.0, 68.0 - 0.6 * max(0.0, dist))
    lean = float(m.get('trunk_lean_bottom_med_deg', m.get('trunk_lean_max_deg', 0.0)))
    if 'torso_lean' in (r.issues or []) and lean > 28.0:
      score -= min(10.0, (lean - 28.0) * 0.35)
  elif engine == 'jumping_jack':
    open_hand = float(m.get('open_hand_ratio_max', 0.0))
    open_feet = float(m.get('open_feet_ratio_max', 0.0))
    close_hand = float(m.get('close_hand_ratio_min', 1.0))
    close_feet = float(m.get('close_feet_ratio_min', 1.0))

    # 这里使用连续子分数。核心想法是：计数可以相对宽松，
    # 但质量分数仍应在不同 rep 之间平滑变化，而不是直接塌缩成固定值。
    # 否则一旦被标成 poor，所有差动作都会变成同一个分数，区分度太差。
    def _rise(v: float, lo: float, hi: float) -> float:
      """把一个“越大越好”的指标映射到 0~1。"""
      return _clamp01((v - lo) / max(1e-6, hi - lo))

    def _fall(v: float, good: float, perfect: float) -> float:
      """把一个“越小越好”的指标映射到 0~1。"""
      # 这个指标越小越好，因此 perfect 阈值应小于 good 阈值。
      return _clamp01((good - v) / max(1e-6, good - perfect))

    s_open_hand = _rise(open_hand, 0.60, 0.95)
    s_open_feet = _rise(open_feet, 1.00, 1.55)
    s_close_hand = _fall(close_hand, 0.32, 0.10)
    s_close_feet = _fall(close_feet, 0.85, 0.30)

    # 在预览场景里，动作打开质量通常比合回质量更影响观感，因此稍微更重要。
    q = 0.30 * s_open_hand + 0.30 * s_open_feet + 0.20 * s_close_hand + 0.20 * s_close_feet
    score = 45.0 + 55.0 * q

    # 对具体问题做小幅扣分，但避免把所有 rep 都压成差不多的分数。
    for it in (r.issues or []):
      if it in ('hands_low', 'feet_narrow'):
        score -= 3.0
      if it in ('hands_not_closed', 'feet_not_closed'):
        score -= 2.0
  elif engine == 'curl':
    elbow = float(m.get('elbow_top_med_deg', m.get('elbow_min_deg', 50.0)))
    if g == 'perfect':
      score = 92.0 + 8.0 * (1.0 - _clamp01(elbow / 35.0))
    elif g == 'good':
      score = 78.0 + 14.0 * (1.0 - _clamp01((elbow - 35.0) / 20.0))
    else:
      score = max(20.0, 68.0 - 0.7 * max(0.0, elbow - 55.0))
    drift_ratio = float(m.get('elbow_drift_ratio', 0.0))
    swing = float(m.get('trunk_swing_deg', 0.0))
    if 'elbow_drift' in (r.issues or []) and drift_ratio > 0.12:
      score -= min(8.0, (drift_ratio - 0.12) * 40.0)
    if 'body_swing' in (r.issues or []) and swing > 14.0:
      score -= min(10.0, (swing - 14.0) * 0.4)
  elif engine == 'bench_press':
    bottom_elbow = float(m.get('elbow_bottom_med_deg', m.get('elbow_min_deg', 100.0)))
    top_lockout = float(m.get('elbow_top_max_deg', 150.0))
    if g == 'perfect':
      score = 90.0 + 10.0 * (1.0 - _clamp01(abs(bottom_elbow - 88.0) / 12.0))
    elif g == 'good':
      score = 78.0 + 12.0 * (1.0 - _clamp01(abs(bottom_elbow - 92.0) / 22.0))
    else:
      score = max(20.0, 66.0 - 0.6 * max(0.0, bottom_elbow - 110.0))
    score += 6.0 * _clamp01((top_lockout - 156.0) / 10.0)
    forearm_offset = float(m.get('bottom_forearm_offset_ratio_max', 0.0))
    hip_lift = float(m.get('hip_lift_ratio_max', 0.0))
    tempo_s = float(m.get('tempo_s', 1.0))
    if 'forearm_not_vertical' in (r.issues or []) and forearm_offset > 0.12:
      score -= min(8.0, (forearm_offset - 0.12) * 45.0)
    if 'hip_bridge' in (r.issues or []) and hip_lift > 0.09:
      score -= min(10.0, (hip_lift - 0.09) * 60.0)
    if 'lockout_incomplete' in (r.issues or []) and top_lockout < 156.0:
      score -= min(8.0, (156.0 - top_lockout) * 0.35)
    if 'tempo_too_fast' in (r.issues or []) and tempo_s < 0.75:
      score -= min(6.0, (0.75 - tempo_s) * 18.0)
  elif engine == 'crunch':
    top_torso = float(m.get('torso_angle_top_max_deg', 25.0))
    if g == 'perfect':
      score = 90.0 + 10.0 * (1.0 - _clamp01(abs(top_torso - 50.0) / 15.0))
    elif g == 'good':
      score = 78.0 + 12.0 * (1.0 - _clamp01(abs(top_torso - 45.0) / 25.0))
    else:
      score = max(20.0, 65.0 - 0.8 * max(0.0, 30.0 - top_torso))
    hip_drift = float(m.get('hip_drift_ratio_max', 0.0))
    knee_drift = float(m.get('knee_drift_ratio_max', 0.0))
    speed = float(m.get('angular_speed_max_deg_s', 0.0))
    ret = float(m.get('return_torso_angle_min_deg', 0.0))
    if 'hip_flexion_compensation' in (r.issues or []):
      score -= min(10.0, max(0.0, hip_drift - 0.10) * 45.0 + max(0.0, knee_drift - 0.12) * 35.0)
    if 'momentum_excessive' in (r.issues or []) and speed > 125.0:
      score -= min(8.0, (speed - 125.0) * 0.08)
    if 'return_incomplete' in (r.issues or []) and ret > 22.0:
      score -= min(6.0, (ret - 22.0) * 0.35)
  else:
    score = 100.0
    for _ in (r.issues or []):
      score -= 8.0
  return float(max(0.0, min(100.0, score)))


def summarize_reps(engine: str, reps: List[RepRecord], group_n: int = 3, user_mode: str = 'simple') -> Dict[str, Any]:
  """
  把逐次 rep 结果整理成整体 summary。
  
  输出包括：次数、均分、问题频次、分组总结以及最差/最好 reps 等。
  """
  seen = set()
  clean: List[RepRecord] = []
  for r in reps:
    key = (r.rep_idx, round(float(r.start_t), 3), round(float(r.end_t), 3))
    if key in seen:
      continue
    seen.add(key)
    clean.append(r)
  reps = clean

  rep_count = len(reps)
  issue_counts: Dict[str, int] = {}
  for r in reps:
    for it in (r.issues or []):
      issue_counts[it] = issue_counts.get(it, 0) + 1

  scored = [(r.rep_idx, score_rep(engine, r)) for r in reps]
  score_by_idx = {i: s for i, s in scored}
  avg_score = float(statistics.mean([s for _, s in scored])) if scored else 0.0

  metric_keys = set()
  for r in reps:
    metric_keys.update((r.metrics or {}).keys())
  metrics_summary = {k: _safe_stats([float((r.metrics or {}).get(k)) for r in reps if (r.metrics or {}).get(k) is not None]) for k in sorted(metric_keys)}
  durations = [float(r.end_t - r.start_t) for r in reps if r.end_t is not None and r.start_t is not None]
  pace = _safe_stats(durations)

  rep_by_idx = {r.rep_idx: r for r in reps}
  def _detail(idx: int, sc: float):
    """内部辅助函数。"""
    rr = rep_by_idx.get(idx)
    return {
      'rep_idx': idx,
      'score': float(sc),
      'start_t': rr.start_t if rr else None,
      'end_t': rr.end_t if rr else None,
      'duration_s': float(rr.end_t - rr.start_t) if rr else None,
      'issues': rr.issues if rr else [],
      'metrics': rr.metrics if rr else {},
      'grade': rr.grade if rr else 'good',
      'incomplete': bool(rr.incomplete) if rr else False,
    }
  worst_detail = [_detail(i, s) for i, s in sorted(scored, key=lambda x: x[1])[:3]]
  best_detail = [_detail(i, s) for i, s in sorted(scored, key=lambda x: x[1], reverse=True)[:3]]

  reps_out = []
  for r in reps:
    d = asdict(r)
    d['duration_s'] = float(r.end_t - r.start_t)
    d['score'] = float(score_by_idx.get(r.rep_idx, 0.0))
    d['level'] = grade_cn(r.grade)
    reps_out.append(d)

  gn = max(1, int(group_n))
  groups = []
  for g_start in range(1, rep_count + 1, gn):
    g_end = min(rep_count, g_start + gn - 1)
    sub = [r for r in reps if g_start <= r.rep_idx <= g_end]
    if not sub:
      continue
    g_scores = [float(score_by_idx.get(r.rep_idx, 0.0)) for r in sub]
    g_avg = float(statistics.mean(g_scores)) if g_scores else 0.0
    g_grade_counts: Dict[str, int] = {}
    for r in sub:
      g_grade_counts[r.grade] = g_grade_counts.get(r.grade, 0) + 1
    g_grade = sorted(g_grade_counts.items(), key=lambda x: x[1], reverse=True)[0][0] if g_grade_counts else 'good'
    g_issue_counts: Dict[str, int] = {}
    for r in sub:
      for it in r.issues:
        g_issue_counts[it] = g_issue_counts.get(it, 0) + 1
    top_issue = sorted(g_issue_counts.items(), key=lambda x: x[1], reverse=True)[0][0] if g_issue_counts else None
    groups.append({
      'group': f'{g_start}-{g_end}',
      'avg_score': g_avg,
      'top_issue': top_issue,
      'grade': g_grade,
      'level': grade_cn(g_grade),
      'comment': group_comment(engine, g_avg, top_issue, mode=user_mode),
    })

  return {
    'engine': engine,
    'rep_count': rep_count,
    'avg_score': avg_score,
    'issue_counts': issue_counts,
    'worst_reps': worst_detail,
    'best_reps': best_detail,
    'pace': pace,
    'metrics_summary': metrics_summary,
    'groups': groups,
    'reps': reps_out,
    'user_mode': str(user_mode),
  }




def _format_token_lines(prefix: str, tok: Dict[str, Any]) -> List[str]:
  """把 token 统计字典格式化为多行文本。"""
  lines: List[str] = []
  if not tok:
    return lines
  calls = int(tok.get('calls', 0) or 0)
  lines.append(f"- token汇总: calls={calls}, prompt_total={int(tok.get('prompt_tokens_total',0) or 0)}, completion_total={int(tok.get('completion_tokens_total',0) or 0)}, total={int(tok.get('total_tokens_total',0) or 0)}")
  lines.append(f"- token均值/次: prompt={float(tok.get('prompt_tokens_avg',0.0) or 0.0):.2f}, completion={float(tok.get('completion_tokens_avg',0.0) or 0.0):.2f}, total={float(tok.get('total_tokens_avg',0.0) or 0.0):.2f}")
  part_in = tok.get('parts_prompt_tokens_est_avg', {}) or {}
  if part_in:
    lines.append(f"- 输入token估算均值/次（按部分）:")
    for k, v in part_in.items():
      lines.append(f" - {k}: {float(v):.2f}")
  part_out = tok.get('parts_output_tokens_est_avg', {}) or {}
  if part_out:
    lines.append(f"- 输出token估算均值/次（按部分）:")
    for k, v in part_out.items():
      lines.append(f" - {k}: {float(v):.2f}")
  return lines

def _overall_level(avg_score: float) -> str:
  """把平均分映射成整体等级标签。"""
  if avg_score >= 92:
    return '优'
  if avg_score >= 80:
    return '良'
  return '差'


def _vlm_section_text(report: Dict[str, Any], key: str, default: str = '') -> str:
  """从 report 中安全读取指定 VLM 文本字段。"""
  sec = report.get('vlm_sections', {}) or {}
  val = sec.get(key)
  return str(val).strip() if isinstance(val, str) and val.strip() else default


def make_markdown(report: Dict[str, Any]) -> str:
  """把结构化 report 生成 markdown 文本。"""
  engine = report.get('engine', '')
  user_mode = str(report.get('user_mode') or report.get('engineering', {}).get('user_mode') or 'simple')
  eng = report.get('engineering', {}) or {}
  perf = eng.get('perf', {}) or {}
  vlm = eng.get('vlm', {}) or {}
  output_meta = eng.get('output', {}) or {}
  show_scores = bool(output_meta.get('show_scores_parenthetical', True))

  rep_count = int(report.get('rep_count', 0))
  avg_score = float(report.get('avg_score', 0.0))
  issue_counts = report.get('issue_counts', {}) or {}
  groups = report.get('groups', []) or []

  lines: List[str] = []
  lines.append(f"# 训练复盘（{engine}）")
  lines.append('')
  lines.append('---')
  lines.append('## Part A：工程参数与性能统计')
  lines.append('')
  if eng.get('video'):
    lines.append(f"- 视频：`{eng.get('video')}`")
  if eng.get('duration_s') is not None:
    lines.append(f"- 时长：**{float(eng.get('duration_s')):.1f}s**")
  if eng.get('fps') is not None:
    lines.append(f"- FPS：**{float(eng.get('fps')):.1f}**")
  if eng.get('resolution'):
    lines.append(f"- 分辨率：{eng.get('resolution')}")
  if eng.get('frames') is not None:
    lines.append(f"- 帧数：{int(eng.get('frames'))}")
  if eng.get('config_digest'):
    lines.append(f"- 配置摘要：{eng.get('config_digest')}")
  vc = eng.get('view_check') or {}
  if vc.get('enabled'):
    lines.append(f"- 视角实验：dominant={vc.get('dominant_view')} conf={float(vc.get('dominant_confidence',0.0)):.2f} | expected={vc.get('expected')} | valid={vc.get('valid_for_exercise')}")
  if output_meta.get('config_name'):
    lines.append(f"- config：`{output_meta.get('config_name')}`")
  if output_meta.get('out_dir'):
    lines.append(f"- 输出目录：`{output_meta.get('out_dir')}`")

  if perf:
    lines.append('')
    lines.append('### 时延（ms，均值）')
    for k in ['yolo_infer_ms', 'smooth_ms', 'engine_ms', 'render_ms', 'total_loop_ms']:
      if k in perf:
        lines.append(f"- {k}: {float(perf[k]):.2f}")

  if vlm:
    lines.append('')
    lines.append('### VLM')
    lines.append(f"- enabled: {bool(vlm.get('enabled', False))}")
    if vlm.get('model'):
      lines.append(f"- model: `{vlm.get('model')}`")
    if vlm.get('base_url'):
      lines.append(f"- base_url: `{vlm.get('base_url')}`")
    rt = vlm.get('realtime', {}) or {}
    ph = vlm.get('posthoc', {}) or {}
    lines.append('')
    lines.append('#### 实时链路 VLM')
    lines.append(f"- enabled: {bool(rt.get('enabled', False))}")
    lines.append(f"- inputs: {rt.get('inputs', {})}")
    lines.append(f"- calls: submitted={int(rt.get('submitted', 0) or 0)}, finished={int(rt.get('finished', 0) or 0)} (ok={int(rt.get('ok',0) or 0)}, fail={int(rt.get('fail',0) or 0)})")
    lines.append(f"- last latency: {float(rt.get('last_latency_s', 0.0) or 0.0):.2f}s")
    if rt.get('last_err'):
      lines.append(f"- last err: {rt.get('last_err')}")
    for _ln in _format_token_lines('rt', rt.get('tokens') or {}):
      lines.append(_ln)
    lines.append('')
    lines.append('#### 课后链路 VLM')
    lines.append(f"- enabled: {bool(ph.get('enabled', False))}")
    lines.append(f"- inputs: {ph.get('inputs', {})}")
    lines.append(f"- calls: submitted={int(ph.get('submitted', 0) or 0)}, finished={int(ph.get('finished', 0) or 0)} (ok={int(ph.get('ok',0) or 0)}, fail={int(ph.get('fail',0) or 0)})")
    lines.append(f"- last latency: {float(ph.get('last_latency_s', 0.0) or 0.0):.2f}s")
    if ph.get('last_err'):
      lines.append(f"- last err: {ph.get('last_err')}")
    for _ln in _format_token_lines('posthoc', ph.get('tokens') or {}):
      lines.append(_ln)

  lines.append('')
  lines.append('---')
  lines.append('## Part B：用户体验复盘（动作与感受）')
  lines.append('')
  lines.append(f"- 总次数：**{rep_count}**")
  if show_scores:
    lines.append(f"- 平均质量：**{_overall_level(avg_score)}**（分数 {avg_score:.1f}，仅供调试）")
  else:
    lines.append(f"- 平均质量：**{_overall_level(avg_score)}**")
  lines.append(f"- 表达模式：**{user_mode}**")

  lines.append('')
  lines.append('### 总体评价')
  lines.append(_vlm_section_text(report, 'overall_comment', overall_comment(rep_count, issue_counts, mode=user_mode)))

  if groups:
    lines.append('')
    lines.append('### 分组点评')
    vlm_groups = (report.get('vlm_sections', {}) or {}).get('group_comments') or {}
    for g in groups:
      g_text = None
      if isinstance(vlm_groups, dict):
        g_text = vlm_groups.get(g.get('group'))
      if show_scores:
        lines.append(f"- 组 {g.get('group')}: **{g.get('level','-')}**（{float(g.get('avg_score',0.0)):.1f}，仅供调试） | {str(g_text or g.get('comment') or '').strip()}")
      else:
        lines.append(f"- 组 {g.get('group')}: **{g.get('level','-')}** | {str(g_text or g.get('comment') or '').strip()}")

  lines.append('')
  lines.append('### 关键问题与建议')
  vlm_issue_list = (report.get('vlm_sections', {}) or {}).get('issue_analysis')
  if isinstance(vlm_issue_list, list) and vlm_issue_list:
    for item in vlm_issue_list:
      tag = str(item.get('tag') or item.get('issue') or 'issue')
      cnt = issue_counts.get(tag, item.get('count', 0))
      text = str(item.get('text') or item.get('advice') or '').strip()
      if text:
        lines.append(f"- {tag}（{cnt} 次）：{text}")
  else:
    if issue_counts:
      for tag, adv in per_issue_advice(issue_counts, mode=user_mode):
        lines.append(f"- {tag}（{issue_counts.get(tag,0)} 次）：{adv}")
    else:
      for ln in encourage_if_clean(engine, mode=user_mode):
        lines.append(f"- {ln}")

  if report.get('vlm_posthoc'):
    lines.append('')
    lines.append('### VLM教练复盘（课后链路）')
    lines.append(str(report.get('vlm_posthoc')).strip())

  lines.append('')
  lines.append('### 最需要关注的 reps（最差 3 个）')
  for w in report.get('worst_reps', []) or []:
    lvl = grade_cn(w.get('grade','good')) if isinstance(w.get('grade'), str) else '-'
    if show_scores:
      lines.append(f"- Rep {w.get('rep_idx')} | {lvl}（{float(w.get('score',0.0)):.1f}，仅供调试） | {float(w.get('start_t',0.0)):.2f}s–{float(w.get('end_t',0.0)):.2f}s | issues: {', '.join(w.get('issues',[])) or 'none'}")
    else:
      lines.append(f"- Rep {w.get('rep_idx')} | {lvl} | {float(w.get('start_t',0.0)):.2f}s–{float(w.get('end_t',0.0)):.2f}s | issues: {', '.join(w.get('issues',[])) or 'none'}")

  lines.append('')
  lines.append('### 表现最好的 reps（最好 3 个）')
  for b in report.get('best_reps', []) or []:
    lvl = grade_cn(b.get('grade','good')) if isinstance(b.get('grade'), str) else '-'
    if show_scores:
      lines.append(f"- Rep {b.get('rep_idx')} | {lvl}（{float(b.get('score',0.0)):.1f}，仅供调试） | {float(b.get('start_t',0.0)):.2f}s–{float(b.get('end_t',0.0)):.2f}s | issues: {', '.join(b.get('issues',[])) or 'none'}")
    else:
      lines.append(f"- Rep {b.get('rep_idx')} | {lvl} | {float(b.get('start_t',0.0)):.2f}s–{float(b.get('end_t',0.0)):.2f}s | issues: {', '.join(b.get('issues',[])) or 'none'}")

  ms = report.get('metrics_summary', {}) or {}
  if ms:
    lines.append('')
    lines.append('### 关键指标统计（用于解释打分/问题）')
    for k, st in ms.items():
      if not st:
        continue
      lines.append(f"- {k}: mean={st.get('mean',0.0):.2f}, median={st.get('median',0.0):.2f}, min={st.get('min',0.0):.2f}, max={st.get('max',0.0):.2f}")

  reps = report.get('reps', []) or []
  if reps:
    lines.append('')
    lines.append('### 每次 rep 明细（节选/完整见 reps.tsv）')
    key_cols = []
    if engine == 'squat':
      key_cols = ['knee_bottom_med_deg', 'knee_min_deg', 'hip_drop_max_ratio', 'trunk_lean_bottom_med_deg']
    elif engine == 'jumping_jack':
      key_cols = ['open_hand_ratio_max', 'open_feet_ratio_max', 'close_hand_ratio_min', 'close_feet_ratio_min']
    elif engine == 'curl':
      key_cols = ['elbow_top_med_deg', 'elbow_min_deg', 'elbow_drift_ratio', 'trunk_swing_deg']
    elif engine == 'pushup':
      key_cols = ['elbow_min_deg', 'body_bend_max_deg']
    elif engine == 'bench_press':
      key_cols = ['elbow_bottom_med_deg', 'elbow_top_max_deg', 'bottom_forearm_offset_ratio_max', 'hip_lift_ratio_max', 'tempo_s']
    elif engine == 'crunch':
      key_cols = ['torso_angle_top_max_deg', 'hip_drift_ratio_max', 'knee_drift_ratio_max', 'angular_speed_max_deg_s', 'return_torso_angle_min_deg']
    cols = ['rep', 'time(s)', 'dur', '等级'] + (['分数(调试)'] if show_scores else []) + ['issues'] + key_cols
    lines.append('|' + '|'.join(cols) + '|')
    lines.append('|' + '|'.join(['---'] * len(cols)) + '|')
    for r in reps[:80]:
      st = r.get('start_t')
      et = r.get('end_t')
      dur = r.get('duration_s')
      sc = r.get('score')
      issues = ','.join(r.get('issues') or [])
      m = r.get('metrics') or {}
      row = [
        str(r.get('rep_idx')),
        f"{float(st):.2f}-{float(et):.2f}" if st is not None and et is not None else '-',
        f"{float(dur):.2f}" if dur is not None else '-',
        str(r.get('level') or '-'),
      ]
      if show_scores:
        row.append(f"{float(sc):.1f}" if sc is not None else '-')
      row.append(issues or '-')
      for k in key_cols:
        v = m.get(k)
        row.append(f"{float(v):.2f}" if isinstance(v, (int, float)) else '-')
      lines.append('|' + '|'.join(row) + '|')
    lines.append('')

  next_goal_text = _vlm_section_text(report, 'next_goal')
  if next_goal_text:
    lines.append('### 下次目标')
    lines.append(f'- {next_goal_text}')

  show_kb = bool((eng.get('output', {}) or {}).get('show_kb_appendix', False))
  kb = report.get('kb_docs', []) or []
  if show_kb and kb:
    lines.append('---')
    lines.append('## 参考知识库（Demo内置）')
    for d in kb:
      lines.append(f"\n### {d.get('title')} ({d.get('doc_id')})\n")
      lines.append(d.get('text', '').strip())
  return '\n'.join(lines)


def save_report(out_json: str, out_md: str, report: Dict[str, Any]) -> None:
  """把 report 同时保存为 json 和 markdown 文件。"""
  with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
  with open(out_md, 'w', encoding='utf-8') as f:
    f.write(make_markdown(report))

  out_tsv = out_md[:-3] + 'tsv' if out_md.lower().endswith('.md') else out_md + '.tsv'
  try:
    reps = report.get('reps', []) or []
    metric_keys = set()
    for r in reps:
      metric_keys.update((r.get('metrics') or {}).keys())
    metric_keys = sorted(metric_keys)
    header = ['rep_idx', 'start_t', 'end_t', 'duration_s', 'score', 'grade', 'level', 'issues'] + metric_keys
    with open(out_tsv, 'w', encoding='utf-8', newline='') as f2:
      w = csv.writer(f2, delimiter='\t')
      w.writerow(header)
      for r in reps:
        row = [
          r.get('rep_idx'),
          f"{float(r.get('start_t',0.0)):.3f}" if r.get('start_t') is not None else '',
          f"{float(r.get('end_t',0.0)):.3f}" if r.get('end_t') is not None else '',
          f"{float(r.get('duration_s',0.0)):.3f}" if r.get('duration_s') is not None else '',
          f"{float(r.get('score',0.0)):.1f}",
          str(r.get('grade','')),
          str(r.get('level','')),
          ','.join(r.get('issues') or []),
        ]
        m = r.get('metrics') or {}
        for k in metric_keys:
          v = m.get(k)
          row.append(f"{float(v):.3f}" if isinstance(v, (int, float)) else '')
        w.writerow(row)
  except Exception:
    pass
