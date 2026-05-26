"""本地知识库文档读取与检索工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class KBDoc:
  """知识库文档结构。"""
  doc_id: str
  title: str
  text: str


def _kb_dir() -> Path:
  """返回内置知识库目录路径。"""
  return Path(__file__).resolve().parent / 'kb'


def load_kb() -> Dict[str, KBDoc]:
  """读取 coach/kb 下的所有 markdown 文档。"""
  docs: Dict[str, KBDoc] = {}
  d = _kb_dir()
  if not d.exists():
    return docs
  for p in sorted(d.glob('*.md')):
    raw = p.read_text(encoding='utf-8').strip()
    title = raw.splitlines()[0].lstrip('# ').strip() if raw else p.stem
    docs[p.stem] = KBDoc(doc_id=p.stem, title=title, text=raw)
  return docs


ENGINE_OVERVIEW = {
  'squat': 'squat_overview',
  'jumping_jack': 'jumping_jack_overview',
  'curl': 'curl_overview',
  'pushup': 'pushup_overview',
  'bench_press': 'bench_press_overview',
  'crunch': 'crunch_overview',
}

CATEGORY_DOCS = {
  'camera': 'common_camera_setup',
  'safety': 'common_safety',
}

TAG_TO_DOC: Dict[Tuple[str, str], str] = {
  ('squat', 'depth_insufficient'): 'squat_depth_insufficient',
  ('squat', 'depth_too_deep'): 'squat_depth_too_deep',
  ('squat', 'torso_lean'): 'squat_torso_lean',
  ('jumping_jack', 'hands_low'): 'jj_hands_low',
  ('jumping_jack', 'feet_narrow'): 'jj_feet_narrow',
  ('jumping_jack', 'hands_not_closed'): 'jj_hands_not_closed',
  ('jumping_jack', 'feet_not_closed'): 'jj_feet_not_closed',
  ('curl', 'rom_insufficient'): 'curl_rom_insufficient',
  ('curl', 'elbow_drift'): 'curl_elbow_drift',
  ('curl', 'body_swing'): 'curl_body_swing',
  ('bench_press', 'rom_insufficient'): 'bench_press_rom_insufficient',
  ('bench_press', 'forearm_not_vertical'): 'bench_press_forearm_not_vertical',
  ('bench_press', 'hip_bridge'): 'bench_press_hip_bridge',
  ('bench_press', 'lockout_incomplete'): 'bench_press_lockout_incomplete',
  ('crunch', 'rom_insufficient'): 'crunch_rom_insufficient',
  ('crunch', 'hip_flexion_compensation'): 'crunch_hip_flexion_compensation',
  ('crunch', 'momentum_excessive'): 'crunch_momentum_excessive',
  ('crunch', 'return_incomplete'): 'crunch_return_incomplete',
}


def retrieve_kb(engine: str, issue_counts: Dict[str, int], max_docs: int = 8, include_overview: bool = True, include_safety: bool = True, include_camera: bool = True) -> List[KBDoc]:
  """根据动作类型和高频问题选择一小批相关知识库文档。"""
  kb = load_kb()
  if not kb:
    return []
  chosen: List[str] = []
  if include_overview:
    ov = ENGINE_OVERVIEW.get(engine)
    if ov and ov in kb:
      chosen.append(ov)
  if include_camera:
    did = CATEGORY_DOCS['camera']
    if did in kb and did not in chosen:
      chosen.append(did)
  if include_safety:
    did = CATEGORY_DOCS['safety']
    if did in kb and did not in chosen:
      chosen.append(did)
  ranked = sorted((issue_counts or {}).items(), key=lambda x: x[1], reverse=True)
  for tag, _cnt in ranked:
    did = TAG_TO_DOC.get((engine, tag))
    if did and did in kb and did not in chosen:
      chosen.append(did)
    if len(chosen) >= max_docs:
      break
  if 'demo_overview' in kb and len(chosen) < max_docs and 'demo_overview' not in chosen:
    chosen.append('demo_overview')
  return [kb[x] for x in chosen if x in kb][:max_docs]


def build_vlm_kb_context(
  engine: str,
  issue_counts: Dict[str, int],
  *,
  max_docs: int = 8,
  max_chars_per_doc: int = 1200,
  include_overview: bool = True,
  include_safety: bool = True,
  include_camera: bool = True,
) -> tuple[str, List[KBDoc]]:
  """把选中的知识库文档拼成可附加到 VLM 提示词中的长文本。"""
  docs = retrieve_kb(engine, issue_counts, max_docs=max_docs, include_overview=include_overview, include_safety=include_safety, include_camera=include_camera)
  if not docs:
    return '', []
  blocks: List[str] = []
  for d in docs:
    txt = d.text.strip()
    if max_chars_per_doc > 0:
      txt = txt[:max_chars_per_doc]
    blocks.append(f'### {d.title} ({d.doc_id})\n{txt}')
  header = '以下内容是给教练模型的专业参考，不要求逐字照搬。'
  return header + '\n\n' + '\n\n'.join(blocks), docs
