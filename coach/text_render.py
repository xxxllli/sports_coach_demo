"""
在图像上安全绘制中文文本的工具。

内部通过 PIL 绘字，再转回 OpenCV 使用。
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _default_font_paths() -> List[str]:
  """返回一组常见字体路径候选，用于尽量兼容不同环境。"""
  return [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyh.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
  ]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
  """按指定字号加载可用字体。"""
  for p in _default_font_paths():
    try:
      return ImageFont.truetype(p, size=size)
    except Exception:
      continue
  return ImageFont.load_default()


def put_text_unicode(
  img_bgr: np.ndarray,
  lines: List[str],
  org: Tuple[int, int] = (20, 30),
  font_size: int = 22,
  line_gap: int = 6,
  with_bg: bool = True,
  bg_alpha: float = 0.55,
  padding: int = 8,
) -> None:
  """在图像上绘制多行中文文本，并可选加半透明底板。"""
  if img_bgr is None or len(lines) == 0:
    return

  img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
  pil_img = Image.fromarray(img_rgb)
  draw = ImageDraw.Draw(pil_img)
  font = _load_font(font_size)

  x, y = org

  if with_bg:
    widths, heights = [], []
    for ln in lines:
      try:
        bbox = draw.textbbox((0, 0), ln, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
      except Exception:
        w, h = draw.textsize(ln, font=font)
      widths.append(w)
      heights.append(h)
    panel_w = max(widths) + 2 * padding
    panel_h = sum(heights) + (len(lines) - 1) * line_gap + 2 * padding
    px1, py1 = x - padding, y - padding
    px2, py2 = px1 + panel_w, py1 + panel_h

    overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rounded_rectangle([px1, py1, px2, py2], radius=10, fill=(0, 0, 0, int(255 * bg_alpha)))
    pil_img = Image.alpha_composite(pil_img.convert('RGBA'), overlay).convert('RGB')
    draw = ImageDraw.Draw(pil_img)

  yy = y
  for ln in lines:
    draw.text((x, yy), ln, font=font, fill=(255, 255, 255))
    yy += font_size + line_gap

  img_bgr[:, :, :] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
