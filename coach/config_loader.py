"""配置读取与合并工具。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def load_json(path: str) -> Dict[str, Any]:
  """读取 UTF-8 编码的 JSON 文件并返回字典对象。"""
  p = Path(path)
  if not p.exists():
    raise FileNotFoundError(f'Config not found: {path}')
  return json.loads(p.read_text(encoding='utf-8'))


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
  """
  递归合并配置字典。
  
  override 中出现的字段会覆盖 base 中对应字段；
  如果值仍然是字典，则继续递归合并。
  """
  out = dict(base)
  for k, v in override.items():
    if isinstance(v, dict) and isinstance(out.get(k), dict):
      out[k] = deep_merge(out[k], v)
    else:
      out[k] = v
  return out


@dataclass
class DemoConfig:
  """
  对合并后的配置字典做一层轻包装。
  
  主要提供安全的多级 get 接口。
  """

  raw: Dict[str, Any]
  name: str = 'default'
  source_path: Optional[str] = None

  def get(self, *keys: str, default=None):
    """
    安全读取多级配置项。
    
    如果中间任一级不存在，则返回 default。
    """
    cur: Any = self.raw
    for k in keys:
      if not isinstance(cur, dict) or k not in cur:
        return default
      cur = cur[k]
    return cur


def _config_name_from_path(path: Optional[str]) -> str:
  """把配置文件路径转换成适合输出目录使用的短名称。"""
  if not path:
    return 'default'
  stem = Path(path).stem
  safe = ''.join(ch if (ch.isalnum() or ch in ('-', '_')) else '_' for ch in stem)
  return safe or 'default'


def load_config(default_path: str, user_path: Optional[str] = None) -> DemoConfig:
  """读取默认配置，并按需叠加用户配置。"""
  base = load_json(default_path)
  source_path = default_path
  if user_path:
    override = load_json(user_path)
    base = deep_merge(base, override)
    source_path = user_path
  return DemoConfig(raw=base, name=_config_name_from_path(source_path), source_path=source_path)
