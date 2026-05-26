"""报告和实时提示使用到的中文文案模板。"""

from __future__ import annotations

from typing import Dict, List, Tuple


def grade_cn(grade: str) -> str:
  """把 perfect / good / poor 映射成中文等级。"""
  g = (grade or '').lower()
  if g == 'perfect':
    return '优'
  if g == 'good':
    return '良'
  return '差'


# 同一类建议文案提供两种风格，当前默认使用 simple。
SUGGESTIONS_SIMPLE = {
  'depth_insufficient': '下去慢一点，底部多停留一会儿，再起来。脚跟踩实，先稳再深。',
  'depth_too_deep': '蹲得有点太深了，控制到你更舒服、更稳的深度。',
  'torso_lean': '胸抬起来、核心收紧。先把动作做稳，再慢慢加深度。',
  'hands_low': '手抬高一点，尽量过头顶。先把幅度做出来。',
  'feet_narrow': '脚再打开一点，落地稳一点。膝踝不舒服就先做不跳版本。',
  'hands_not_closed': '合回时手回到身体两侧，动作更干净。',
  'feet_not_closed': '合回时脚并拢一点，动作更利索。',
  'rom_insufficient': '顶端再卷一点，下降放慢一点，别急着做完。',
  'elbow_drift': '肘固定在身体两侧别跑，必要时减重量。',
  'body_swing': '身体别借力，慢一点做，控制住节奏。',
  'lockout_incomplete': '顶部把手臂再推直一点，完成锁定。',
  'forearm_not_vertical': '底部让手腕尽量叠在肘正上方，前臂更竖直。',
  'hip_bridge': '臀部不要顶起，保持肩臀稳定贴住凳/垫。',
  'tempo_too_fast': '下放和回程都慢一点，把离心控制住。',
  'hip_flexion_compensation': '卷腹时别用腿和骨盆带动作，重点放在上腹卷起。',
  'momentum_excessive': '不要甩起来，卷起和回程都更受控一些。',
  'return_incomplete': '回程再放低一点，让腹部重新被拉长。',
}

SUGGESTIONS_PRO = {
  'depth_insufficient': '下去更慢一点，底部加短暂停顿再起来；脚跟踩实，膝盖跟脚尖同向。',
  'depth_too_deep': '下蹲过深：把最低点控制在更舒适的范围，避免膝关节“硬顶”。',
  'torso_lean': '躯干前倾偏大：核心收紧、胸抬起；必要时减深度或改前抱式。',
  'hands_low': '手臂上举不足：手尽量过头顶再下来，保持与脚同步。',
  'feet_narrow': '脚打开幅度不足：脚踝间距再大一点，落地缓冲。',
  'hands_not_closed': '闭合阶段手回落不够：合回时手回到体侧。',
  'feet_not_closed': '闭合阶段脚合回不够：合回时脚并拢一点。',
  'rom_insufficient': '弯举幅度不足：顶端收紧、下降受控（慢放），避免半程。',
  'elbow_drift': '肘漂移：大臂固定贴身，必要时靠墙/坐姿限制借力。',
  'body_swing': '身体借力：核心收紧、减重量，下降更慢。',
  'lockout_incomplete': '锁定不足：推起末端把肘伸直，但不要用肩前顶代偿。',
  'forearm_not_vertical': '底部前臂不够垂直：下放到底时尽量保持腕在肘正上方。',
  'hip_bridge': '桥起代偿：臀部保持稳定，避免借腰桥起把重量顶上去。',
  'tempo_too_fast': '节奏过快：离心下放与回程都应更受控，避免弹起式发力。',
  'hip_flexion_compensation': '髋屈代偿：卷腹时减少骨盆前移和腿部参与，把力量集中到腹部卷曲。',
  'momentum_excessive': '惯性过大：起身不要甩，回程不要砸回垫面。',
  'return_incomplete': '回程不足：每次回到更接近平躺的位置，再开始下一次卷起。',
}


def advice_for(tag: str, mode: str = 'simple') -> str:
  """根据问题标签返回对应建议文案。"""
  if mode == 'pro':
    return SUGGESTIONS_PRO.get(tag, '先放慢一点做稳，再逐步加次数。')
  return SUGGESTIONS_SIMPLE.get(tag, '先放慢一点做稳，再逐步加次数。')


def per_issue_advice(issue_counts: Dict[str, int], mode: str = 'simple') -> List[Tuple[str, str]]:
  """把问题频次统计转换成按频次排序的建议列表。"""
  out = []
  for tag, _cnt in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
    out.append((tag, advice_for(tag, mode)))
  return out


def encourage_if_clean(engine: str, mode: str = 'simple') -> List[str]:
  """当问题较少时，返回鼓励性文案。"""
  if mode == 'pro':
    if engine == 'squat':
      return ['动作质量不错，节奏稳定。', '下一次可以在最低点更稳一些，再考虑加量。']
    if engine == 'jumping_jack':
      return ['开合跳幅度与节奏都不错。', '下一次保持幅度不变，略微提高频率即可。']
    if engine == 'curl':
      return ['弯举控制得很干净。', '想进阶的话，下降阶段再慢一点。']
    return ['整体做得不错，继续保持。']

  if engine == 'squat':
    return ['这次深蹲很稳！', '如果想再进步：下去更慢一点，底部多停一会儿。']
  if engine == 'jumping_jack':
    return ['开合跳节奏很好！', '保持这个幅度，慢慢加快一点点就行。']
  if engine == 'curl':
    return ['弯举做得很干净！', '想更有感觉：下降慢一点，别着急。']
  if engine == 'bench_press':
    return ['卧推动作整体很稳！', '继续保持下放控制和顶部锁定。']
  if engine == 'crunch':
    return ['卷腹控制得不错！', '继续保持卷起高度和回程控制。']
  return ['整体做得不错，继续保持！']


def overall_comment(rep_count: int, issue_counts: Dict[str, int], mode: str = 'simple') -> str:
  """在未启用 posthoc VLM 时，生成一段总体评价。"""
  total_issues = sum(issue_counts.values())
  if rep_count <= 0:
    return '这段视频里我没稳定识别到完整次数。先把机位和入镜调好，再来一次就会更准。'

  ratio = total_issues / max(1, rep_count)
  if total_issues == 0:
    return '整体非常棒：动作干净、节奏稳定，可以放心加量！'
  if ratio <= 0.3:
    return '整体很不错：问题很少，修一两个小细节就更标准了。'
  if ratio <= 1.0:
    return '整体还可以：有一些常见小问题，先把最常见的 1–2 个修好，进步会很快。'
  return '别灰心：这次动作还没完全稳定。我们先把节奏放慢、把最关键的问题修掉，很快就会好很多。'


def group_comment(engine: str, avg_score: float, top_issue: str | None, mode: str = 'simple') -> str:
  """为每一组动作生成一段简短点评。"""
  if top_issue is None:
    if avg_score >= 92:
      return '这一组很稳，继续保持！'
    return '这一组整体不错，保持节奏就行。'
  return f'这一组主要问题：{top_issue}。{advice_for(top_issue, mode)}'
