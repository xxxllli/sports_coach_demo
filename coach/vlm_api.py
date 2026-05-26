"""
OpenAI 兼容 VLM 调用封装。

本文件负责把文本、图片等输入打包成 chat-completions 所需的内容结构，
并提供安全调用包装，避免接口异常直接打断主流程。
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

import httpx


@dataclass
class VLMConfig:
  """
  VLM 连接配置。
  
  包含 base_url、api_key、model_name、timeout_s 等字段。
  """

  base_url: str
  api_key: str
  model: str
  timeout_s: float = 30.0

  @staticmethod
  def from_config(cfg=None, prefix: str = 'VLM_') -> 'VLMConfig':
    """
    从项目配置中读取 VLM 配置。
    
    优先读取 config 文件；
    如果配置里缺失，再回退到环境变量。
    """
    timeout_s = 30.0
    if cfg is not None:
      timeout_s = float(cfg.get('vlm', 'timeout_s', default=30.0) or 30.0)

    base_url = str(cfg.get('vlm', 'base_url', default='') or '') if cfg is not None else ''
    api_key = str(cfg.get('vlm', 'api_key', default='') or '') if cfg is not None else ''
    model = str(cfg.get('vlm', 'model_name', default='') or '') if cfg is not None else ''
    if not model and cfg is not None:
      model = str(cfg.get('vlm', 'model', default='') or '')

    # 环境变量仍然保留为兜底方案，方便临时调试。
    if not base_url:
      base_url = os.getenv(prefix + 'BASE_URL', '').strip()
    if not api_key:
      api_key = os.getenv(prefix + 'API_KEY', '').strip()
    if not model:
      model = os.getenv(prefix + 'MODEL', '').strip()
    if cfg is None:
      timeout_s = float(os.getenv(prefix + 'TIMEOUT_S', '30'))

    if not base_url:
      raise ValueError('Missing VLM base_url in config.vlm.base_url')
    if not api_key:
      raise ValueError('Missing VLM api_key in config.vlm.api_key')
    if not model:
      raise ValueError('Missing VLM model_name in config.vlm.model_name')
    return VLMConfig(base_url=base_url, api_key=api_key, model=model, timeout_s=timeout_s)

  # 保留旧方法名作为兼容别名，避免旧代码调用失败。
  from_env = from_config


def _normalize_base_url(base_url: str) -> str:
  """规范化 base_url，确保其以 /v1 结尾。"""
  b = base_url.rstrip('/')
  if '/v1' in b:
    return b
  return b + '/v1'


def _img_to_data_url_jpeg(img_bgr, quality: int = 85) -> str:
  """把 BGR 图像编码为 data:image/jpeg;base64 字符串。"""
  import cv2
  ok, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(max(30, min(quality, 95)))])
  if not ok:
    raise RuntimeError('Failed to encode image to JPEG')
  b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
  return f'data:image/jpeg;base64,{b64}'


@lru_cache(maxsize=8)
def _cached_openai_client(base_url: str, api_key: str, timeout_s: float):
  """按 base_url / api_key / timeout 复用 OpenAI 客户端，避免重复创建。"""
  from openai import OpenAI

  # 可选的自定义 HTTPX 模式。
  # 某些内网或自签名证书环境下，可能需要把 verify 设为 False。
  use_custom = os.getenv('VLM_USE_CUSTOM_HTTPX', '').strip().lower() in {'1', 'true', 'yes'}
  if use_custom:
    return OpenAI(
      base_url=base_url,
      api_key=api_key,
      timeout=timeout_s,
      http_client=httpx.Client(transport=httpx.HTTPTransport(verify=False, proxy=None)),
      max_retries=0,
    )
  return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=0)


def _client(cfg: VLMConfig):
  """根据当前 VLMConfig 获取客户端对象。"""
  return _cached_openai_client(_normalize_base_url(cfg.base_url), cfg.api_key, float(cfg.timeout_s))


def call_openai_compat_vlm_content(
  cfg: VLMConfig,
  system_prompt: str,
  content_items: List[dict],
  temperature: float = 0.2,
  max_tokens: int = 200,
) -> Tuple[str, Optional[dict]]:
  """
  调用 OpenAI 兼容 VLM 接口。
  
  参数说明：
  - cfg：VLM 连接配置
  - system_prompt：系统提示词
  - content_items：用户侧内容列表，可混合文本和图片
  - temperature / max_tokens：采样与输出长度控制
  
  返回值：
  - 文本结果
  - usage 统计（若接口返回）
  """
  client = _client(cfg)
  resp = client.chat.completions.create(
    model=cfg.model,
    messages=[
      {'role': 'system', 'content': system_prompt},
      {'role': 'user', 'content': content_items},
    ],
    temperature=float(temperature),
    max_tokens=int(max_tokens),
  )
  txt = (resp.choices[0].message.content or '').strip()
  usage = None
  try:
    usage = resp.usage.model_dump() if getattr(resp, 'usage', None) is not None else None
  except Exception:
    usage = None
  return txt, usage


def safe_call_vlm_content(*args, **kwargs) -> Tuple[str, Optional[str], Optional[dict]]:
  """安全包装版本：发生异常时不抛出，而是返回 (text, err, usage)。"""
  try:
    txt, usage = call_openai_compat_vlm_content(*args, **kwargs)
    return txt, None, usage
  except Exception as e:
    return '', f'{type(e).__name__}: {e}', None


def call_openai_compat_vlm(cfg: VLMConfig, system_prompt: str, user_text: str, image_bgr=None, temperature: float = 0.2, max_tokens: int = 200):
  """常用简化接口：一段文本 + 一张图。"""
  content = []
  if user_text:
    content.append({'type': 'text', 'text': user_text})
  if image_bgr is not None:
    content.append({'type': 'image_url', 'image_url': {'url': _img_to_data_url_jpeg(image_bgr)}})
  return call_openai_compat_vlm_content(cfg, system_prompt, content, temperature=temperature, max_tokens=max_tokens)


def safe_call_vlm(*args, **kwargs):
  """常用简化接口的安全包装版本。"""
  try:
    txt, usage = call_openai_compat_vlm(*args, **kwargs)
    return txt, None, usage
  except Exception as e:
    return '', f'{type(e).__name__}: {e}', None
