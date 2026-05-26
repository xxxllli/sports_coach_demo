"""
独立的 VLM 连通性测试脚本。

用于在不跑完整视频流程的情况下，快速验证
base_url / api_key / model_name 是否配置正确。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from coach.config_loader import load_config
from coach.vlm_api import VLMConfig, safe_call_vlm


def main() -> int:
  """
  测试 VLM 是否连通。
  
  该函数会读取配置、构造一张简单测试图，
  然后调用 VLM 接口，方便快速确认连通性。
  """
  parser = argparse.ArgumentParser(description='Test VLM connectivity (OpenAI-compatible vision endpoint).')
  parser.add_argument('--text', default='请用中文简短描述这张测试图片，并回复“VLM连通正常”。', help='User text prompt.')
  parser.add_argument('--save_test_image', action='store_true', help='Save the generated test image to tools/test_vlm_image.jpg.')
  parser.add_argument('--config', default=None, help='Optional config json that provides base_url / api_key / model_name.')
  args = parser.parse_args()

  cfg_obj = None
  if args.config:
    default_cfg = str(Path(__file__).resolve().parents[1] / 'config.default.json')
    cfg_obj = load_config(default_cfg, args.config)

  try:
    cfg = VLMConfig.from_config(cfg_obj)
  except Exception as e:
    print(f'[TEST][VLM][INIT][FAIL] {type(e).__name__}: {e}', flush=True)
    return 2

  print(f'[TEST][VLM][INIT] model={cfg.model} | base={cfg.base_url}', flush=True)

  # 构造一张简单的测试图片，避免依赖外部图像文件。
  img = np.zeros((240, 400, 3), dtype=np.uint8)
  img[:] = (250, 250, 250)
  cv2.rectangle(img, (20, 20), (380, 220), (0, 180, 255), 3)
  cv2.putText(img, 'Fitness VLM Test', (40, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2, cv2.LINE_AA)
  cv2.putText(img, 'If you see this, image upload works.', (35, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2, cv2.LINE_AA)

  if args.save_test_image:
    out = Path(__file__).resolve().parent / 'test_vlm_image.jpg'
    cv2.imwrite(str(out), img)
    print(f'[TEST][VLM][IMAGE] saved={out}', flush=True)

  sys_prompt = '你是一个图像理解助手。请用中文回答，简短直接。'
  print('[TEST][VLM][SUBMIT] sending request...', flush=True)
  txt, err, usage = safe_call_vlm(cfg, sys_prompt, args.text, image_bgr=img, temperature=0.2, max_tokens=120)

  if txt and txt.strip():
    print('[TEST][VLM][OK] response:', flush=True)
    print(txt.strip(), flush=True)
    if usage:
      print(f'[TEST][VLM][USAGE] {usage}', flush=True)
    return 0

  print(f'[TEST][VLM][FAIL] err={err}', flush=True)
  if usage:
    print(f'[TEST][VLM][USAGE] {usage}', flush=True)
  return 1


if __name__ == '__main__':
  raise SystemExit(main())
