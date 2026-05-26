"""
命令行入口。

这个文件只暴露当前工程实际需要的主要参数，
便于直接运行与阅读。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from coach.config_loader import load_config
from coach.demo import run_demo


def build_arg_parser() -> argparse.ArgumentParser:
  """
  构建命令行参数解析器。
  
  这里保留当前工程常用的核心参数：
  - 输入视频与动作类型
  - 可视化/保存开关
  - 少量教练策略覆盖项
  - 课后 VLM 开关
  """
  parser = argparse.ArgumentParser(
    description='Fitness Video Coach Demo'
  )

  # -----------------------------
  # 核心业务输入参数
  # -----------------------------
  parser.add_argument('--video', required=True, help='Path to the input video file.')
  parser.add_argument(
    '--exercise',
    required=True,
    choices=['squat', 'jumping_jack', 'curl', 'pushup', 'bench_press', 'crunch'],
    help='Exercise type. Supported values: squat, jumping_jack, curl, pushup, bench_press, crunch.',
  )
  parser.add_argument('--goal', type=int, default=20, help='Goal reps used by the motivation logic.')

  # -----------------------------
  # 可视化与产物输出相关参数
  # -----------------------------
  parser.add_argument('--show', action='store_true', help='Show the realtime preview window.')
  parser.add_argument('--save_video', action='store_true', help='Save the annotated output video.')

  # -----------------------------
  # 模型与配置文件相关参数
  # -----------------------------
  parser.add_argument(
    '--yolo_model',
    default='yolov8n-pose.pt',
    help='YOLO pose model filename or path. Usually a local .pt file.',
  )
  parser.add_argument(
    '--config',
    default=None,
    help='Optional custom config json. It will override config.default.json.',
  )

  # -----------------------------
  # 工程调试/运行便捷开关
  # -----------------------------
  parser.add_argument('--realtime', action='store_true', help='Sync preview speed to the original FPS.')
  parser.add_argument('--verbose', action='store_true', help='Print more logs to terminal.')

  # -----------------------------
  # 实时教练策略覆盖项
  # -----------------------------
  # 虽然默认配置已经针对预览做了调优，
  # 但如果想临时试一下不同的反馈节奏，
  # 这些命令行参数仍然很有用，无需直接改配置文件。
  parser.add_argument('--coach_mode', choices=['event', 'group', 'time'], default=None, help='Override coach.mode from config.')
  parser.add_argument('--coach_group_n', type=int, default=None, help='Override coach.group_n from config.')
  parser.add_argument('--coach_interval_s', type=float, default=None, help='Override coach.interval_s from config.')
  parser.add_argument('--tip_hold_s', type=float, default=None, help='Override coach.tip_hold_s from config.')

  # -----------------------------
  # 只保留课后 VLM 相关开关
  # -----------------------------
  # 这个入口只提供课后 VLM 开关，用于控制是否生成课后总结。
  parser.add_argument('--vlm_posthoc', dest='vlm_posthoc', action='store_true', help='Enable posthoc VLM summary.')
  parser.add_argument('--no_vlm_posthoc', dest='vlm_posthoc', action='store_false', help='Disable posthoc VLM summary.')
  parser.set_defaults(vlm_posthoc=None)

  return parser


def main() -> None:
  """
  脚本主入口。
  
  流程：解析命令行参数 -> 读取配置 -> 固定 user_mode=simple
  并强制关闭实时 VLM -> 调用 demo 主流程。
  """
  parser = build_arg_parser()
  args = parser.parse_args()

  # 先读取项目默认配置，再按需叠加用户自定义配置。
  default_cfg = str(Path(__file__).resolve().parent / 'config.default.json')
  cfg = load_config(default_cfg, args.config)

  # 先从配置文件读取默认的 VLM 开关。
  # 只有用户在命令行显式传参时，才覆盖配置文件中的默认值。
  cfg_vlm_post = bool(cfg.get('vlm', 'posthoc', default=False))
  cfg_vlm_debug = bool(cfg.get('vlm', 'debug', default=False))
  enable_vlm_posthoc = cfg_vlm_post if args.vlm_posthoc is None else bool(args.vlm_posthoc)

  # 这里关闭实时 VLM，并把 user_mode 设为 simple。
  out_dir = run_demo(
    video_path=args.video,
    exercise=args.exercise,
    goal_reps=int(args.goal),
    cfg=cfg,
    show=bool(args.show),
    save_video=bool(args.save_video),
    yolo_model=args.yolo_model,
    realtime_sync=bool(args.realtime) if args.realtime else None,
    user_mode='simple',
    coach_mode=args.coach_mode,
    coach_group_n=args.coach_group_n,
    coach_interval_s=args.coach_interval_s,
    tip_hold_s=args.tip_hold_s,
    enable_vlm_rt=False,
    vlm_min_interval_s=3.0, # 当前流程不会用到这个参数，这里仅传入一个占位值
    enable_vlm_posthoc=enable_vlm_posthoc,
    vlm_rag=False,
    vlm_debug=cfg_vlm_debug,
    terminal_verbose=bool(args.verbose) if args.verbose else None,
  )

  print(f'\nDone. Outputs saved to: {out_dir.resolve()}')


if __name__ == '__main__':
  main()
