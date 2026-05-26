"""
YOLO Pose 前端封装。

负责权重路径解析、单帧推理以及简单的人物跟踪。
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# 某些内部 Windows 环境里会用到的默认权重来源目录。
_DEFAULT_WEIGHTS_SRC_WIN = r"D:\16_运动指导\data\weights\YOLO"


def _project_root() -> Path:
  """返回工程根目录路径。"""
  return Path(__file__).resolve().parents[1]


def _weights_dir(project_root: Path) -> Path:
  """返回本地 weights 目录，不存在时自动创建。"""
  d = project_root / 'weights'
  d.mkdir(parents=True, exist_ok=True)
  return d


def _try_seed_weights_from_src(model_name: str, project_root: Path) -> Optional[str]:
  """尝试从预设来源目录复制权重到本地 weights 目录。"""
  src_dir = os.getenv('YOLO_WEIGHTS_SRC_DIR', '').strip()
  if not src_dir and os.name == 'nt':
    src_dir = _DEFAULT_WEIGHTS_SRC_WIN
  if not src_dir:
    return None

  src_root = Path(src_dir)
  if not src_root.exists():
    return None

  dst_root = _weights_dir(project_root)

  # 先尝试按完全同名的权重文件查找。
  src = src_root / model_name
  if src.exists() and src.is_file():
    dst = dst_root / src.name
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
      shutil.copy2(src, dst)
    return str(dst)

  # 对默认的轻量姿态模型，再额外尝试一次更宽松的兜底查找。
  if model_name == 'yolov8n-pose.pt':
    cands = sorted(src_root.glob('*pose*.pt'))
    if cands:
      src2 = cands[0]
      dst2 = dst_root / src2.name
      if not dst2.exists() or dst2.stat().st_size != src2.stat().st_size:
        shutil.copy2(src2, dst2)
      return str(dst2)

  return None


def resolve_model_path(model_name_or_path: str) -> str:
  """把模型名或路径解析成可直接加载的权重路径。"""
  p = Path(model_name_or_path)
  if p.exists():
    return str(p)

  project_root = _project_root()
  wdir = _weights_dir(project_root)

  seeded = _try_seed_weights_from_src(model_name_or_path, project_root)
  if seeded and Path(seeded).exists():
    return seeded

  cand = wdir / model_name_or_path
  if cand.exists():
    return str(cand)

  raise FileNotFoundError(
    f"YOLO weight not found: '{model_name_or_path}'.\n"
    f"Fix: copy the weight file to one of these locations and rerun:\n"
    f" - {Path.cwd() / model_name_or_path}\n"
    f" - {wdir / model_name_or_path}\n\n"
    f"Also supported: set env YOLO_WEIGHTS_SRC_DIR to a local folder containing weights\n"
    f"(Windows default: {_DEFAULT_WEIGHTS_SRC_WIN}).\n"
  )


def _iou(a: np.ndarray, b: np.ndarray) -> float:
  """计算两个轴对齐框的 IoU，用于简单人物跟踪。"""
  x1 = max(float(a[0]), float(b[0]))
  y1 = max(float(a[1]), float(b[1]))
  x2 = min(float(a[2]), float(b[2]))
  y2 = min(float(a[3]), float(b[3]))
  iw = max(0.0, x2 - x1)
  ih = max(0.0, y2 - y1)
  inter = iw * ih
  area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
  area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
  union = area_a + area_b - inter
  return float(inter / union) if union > 1e-6 else 0.0


@dataclass
class PoseResult:
  """统一的单人姿态推理结果。"""
  kpts: Optional[np.ndarray] # 形状：(17, 3)，分别表示 x、y、置信度
  box: Optional[np.ndarray]  # 形状：(4,)，分别表示 x1、y1、x2、y2


class YoloPoseFrontend:
  """
  Ultralytics YOLO Pose 前端。
  
  除推理外，还负责用一个非常简单的 IoU 规则在连续帧中跟踪同一个人。
  """

  def __init__(self, model_name: str = 'yolov8n-pose.pt', device: str = 'cpu', conf: float = 0.25, imgsz: int = 640):
    """初始化 YOLO Pose 模型与跟踪状态。"""
    from ultralytics import YOLO
    model_path = resolve_model_path(model_name)
    self.model = YOLO(model_path)
    self.device = device
    self.conf = float(conf)
    self.imgsz = int(imgsz)
    self._prev_box: Optional[np.ndarray] = None
    self._lost_frames = 0
    self.max_lost_frames = 8

  def reset_track(self):
    """忘记上一帧跟踪的人，强制下一帧重新选择目标。"""
    self._prev_box = None
    self._lost_frames = 0

  def infer(self, frame_bgr) -> PoseResult:
    """对单帧图像执行 YOLO Pose 推理，并选出当前目标人物。"""
    res_list = self.model.predict(frame_bgr, verbose=False, conf=self.conf, imgsz=self.imgsz, device=self.device)
    if not res_list:
      self._lost_frames += 1
      return PoseResult(None, None)
    r = res_list[0]
    if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
      self._lost_frames += 1
      return PoseResult(None, None)

    boxes = r.boxes.xyxy.cpu().numpy()
    kxy = r.keypoints.xy.cpu().numpy()
    kconf = r.keypoints.conf.cpu().numpy()

    n = boxes.shape[0]
    idx = 0

    if self._prev_box is not None and self._lost_frames < self.max_lost_frames:
      ious = np.array([_iou(boxes[i], self._prev_box) for i in range(n)], dtype=np.float32)
      idx = int(np.argmax(ious))
      if float(ious[idx]) < 0.05:
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        idx = int(np.argmax(areas))
    else:
      areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
      idx = int(np.argmax(areas))

    box = boxes[idx].astype(np.float32)
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 0:2] = kxy[idx]
    kpts[:, 2] = kconf[idx]

    self._prev_box = box
    self._lost_frames = 0
    return PoseResult(kpts, box)
