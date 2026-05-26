"""
Repeatable benchmark runner for Fitness Video Coach Demo.

Examples:
  python benchmark.py --video test_assets/squat_360x640_8s.mp4 --exercise squat
  python benchmark.py --video test_assets/squat_360x640_8s.mp4 --exercise squat --runs 3 --warmup-runs 1
  python benchmark.py --video test_assets/squat_360x640_8s.mp4 --exercise squat --posthoc-vlm mock --mock-vlm-delay-s 1.0
  python benchmark.py --video your_video.mp4 --exercise squat --posthoc-vlm real --config config.local.json
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

try:
  import psutil
except Exception:  # pragma: no cover - benchmark still works without psutil
  psutil = None


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from coach.config_loader import DemoConfig, load_config
from coach.demo import run_demo
from coach.token_stats import estimate_text_tokens


SUPPORTED_EXERCISES = ['squat', 'jumping_jack', 'curl', 'pushup', 'bench_press', 'crunch']
PROXY_ENV_KEYS = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'NO_PROXY', 'no_proxy']


def build_arg_parser() -> argparse.ArgumentParser:
  """Build the CLI parser for benchmark runs."""
  parser = argparse.ArgumentParser(description='Benchmark Fitness Video Coach Demo.')
  parser.add_argument('--video', required=True, help='Path to the input video file.')
  parser.add_argument('--exercise', required=True, choices=SUPPORTED_EXERCISES, help='Exercise type.')
  parser.add_argument('--goal', type=int, default=20, help='Goal reps passed into run_demo.')
  parser.add_argument('--config', default=None, help='Optional config override JSON.')
  parser.add_argument('--yolo-model', default='yolov8n-pose.pt', help='YOLO pose weight filename or path.')
  parser.add_argument('--runs', type=int, default=3, help='Measured benchmark runs.')
  parser.add_argument('--warmup-runs', type=int, default=1, help='Warmup runs that are not included in the summary.')
  parser.add_argument('--show', action='store_true', help='Show preview windows during benchmark.')
  parser.add_argument('--save-video', action='store_true', help='Save annotated output videos during benchmark.')
  parser.add_argument('--realtime-sync', action='store_true', help='Sync processing to source FPS.')
  parser.add_argument('--verbose', action='store_true', help='Enable terminal logs inside run_demo.')
  parser.add_argument('--user-mode', choices=['simple', 'pro'], default='simple', help='Coach wording style.')
  parser.add_argument('--coach-mode', choices=['event', 'group', 'time'], default='group', help='Coach policy mode.')
  parser.add_argument('--coach-group-n', type=int, default=1, help='Coach group size for group mode.')
  parser.add_argument('--coach-interval-s', type=float, default=4.0, help='Coach interval for time mode.')
  parser.add_argument('--tip-hold-s', type=float, default=2.0, help='How long a frame-level tip stays on screen.')
  parser.add_argument('--posthoc-vlm', choices=['off', 'mock', 'real'], default='off', help='Posthoc VLM mode.')
  parser.add_argument('--mock-vlm-delay-s', type=float, default=0.0, help='Artificial delay added by the local mock VLM server.')
  parser.add_argument('--vlm-rag', action='store_true', help='Enable KB retrieval when posthoc VLM is on.')
  parser.add_argument('--vlm-debug', action='store_true', help='Print VLM debug logs from run_demo.')
  parser.add_argument('--cpu-sample-interval-s', type=float, default=0.1, help='Sampling interval for process CPU/RSS metrics.')
  parser.add_argument('--output-json', default=None, help='Optional path for the benchmark summary JSON.')
  parser.add_argument('--label', default='', help='Optional short label included in the output filename.')
  parser.add_argument('--cleanup-run-outputs', action='store_true', help='Delete each run output directory after metrics are extracted.')
  return parser


def safe_slug(text: str, default: str = 'benchmark') -> str:
  """Convert text into a filesystem-friendly slug."""
  raw = ''.join(ch if (ch.isalnum() or ch in ('-', '_')) else '_' for ch in str(text or '').strip())
  return raw[:80] if raw else default


def clone_config(cfg: DemoConfig) -> DemoConfig:
  """Deep-copy a DemoConfig so each run can mutate it independently."""
  return DemoConfig(raw=copy.deepcopy(cfg.raw), name=cfg.name, source_path=cfg.source_path)


def now_utc_iso() -> str:
  """Return an ISO timestamp in UTC."""
  return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def calc_stats(values: Iterable[float]) -> Dict[str, float]:
  """Compute summary statistics for a numeric sequence."""
  xs = [float(v) for v in values]
  if not xs:
    return {}
  out = {
    'mean': float(statistics.mean(xs)),
    'min': float(min(xs)),
    'max': float(max(xs)),
  }
  if len(xs) >= 2:
    out['std'] = float(statistics.pstdev(xs))
  return out


def collect_gpu_info() -> List[Dict[str, Any]]:
  """Best-effort GPU metadata from nvidia-smi."""
  try:
    cmd = [
      'nvidia-smi',
      '--query-gpu=name,driver_version,memory.total',
      '--format=csv,noheader,nounits',
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
  except Exception:
    return []

  gpus: List[Dict[str, Any]] = []
  for line in proc.stdout.splitlines():
    parts = [part.strip() for part in line.split(',')]
    if len(parts) < 3:
      continue
    gpu: Dict[str, Any] = {
      'name': parts[0],
      'driver_version': parts[1],
    }
    try:
      gpu['memory_total_mb'] = float(parts[2])
    except Exception:
      gpu['memory_total_mb'] = parts[2]
    gpus.append(gpu)
  return gpus


def collect_system_info() -> Dict[str, Any]:
  """Collect host metadata that helps interpret benchmark numbers."""
  info: Dict[str, Any] = {
    'platform': platform.platform(),
    'python': sys.version,
    'processor': platform.processor(),
    'cpu_logical': os.cpu_count(),
    'cwd': str(PROJECT_ROOT),
    'gpus': collect_gpu_info(),
  }
  if psutil is not None:
    try:
      info['cpu_physical'] = psutil.cpu_count(logical=False)
      info['memory_total_mb'] = round(psutil.virtual_memory().total / (1024 * 1024), 2)
    except Exception:
      pass
  return info


def is_local_base_url(base_url: str) -> bool:
  """Return True when a base_url points to localhost."""
  try:
    parsed = urlparse(base_url)
  except Exception:
    return False
  host = (parsed.hostname or '').lower()
  return host in {'127.0.0.1', 'localhost'}


@contextlib.contextmanager
def localhost_no_proxy(enabled: bool):
  """Ensure localhost requests bypass system HTTP proxies."""
  if not enabled:
    yield
    return

  previous = {key: os.environ.get(key) for key in PROXY_ENV_KEYS}
  try:
    for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
      os.environ[key] = ''
    existing = previous.get('NO_PROXY') or previous.get('no_proxy') or ''
    parts = [chunk.strip() for chunk in existing.split(',') if chunk.strip()]
    for item in ['127.0.0.1', 'localhost']:
      if item not in parts:
        parts.append(item)
    joined = ','.join(parts)
    os.environ['NO_PROXY'] = joined
    os.environ['no_proxy'] = joined
    yield
  finally:
    for key, value in previous.items():
      if value is None:
        os.environ.pop(key, None)
      else:
        os.environ[key] = value


class ProcessSampler:
  """Sample process CPU and RSS usage during a benchmark run."""

  def __init__(self, interval_s: float):
    self.interval_s = max(0.02, float(interval_s))
    self.samples: List[Dict[str, float]] = []
    self._stop = threading.Event()
    self._thread: Optional[threading.Thread] = None

  def start(self) -> None:
    """Start the background sampler."""
    if psutil is None:
      return
    proc = psutil.Process(os.getpid())
    proc.cpu_percent(None)

    def _worker() -> None:
      while not self._stop.is_set():
        try:
          cpu_pct = proc.cpu_percent(interval=self.interval_s)
          rss_mb = proc.memory_info().rss / (1024 * 1024)
          self.samples.append({'cpu_pct': float(cpu_pct), 'rss_mb': float(rss_mb)})
        except Exception:
          return

    self._thread = threading.Thread(target=_worker, daemon=True)
    self._thread.start()

  def stop(self) -> Dict[str, Any]:
    """Stop sampling and return a compact summary."""
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout=max(1.0, self.interval_s * 2))

    if not self.samples:
      return {}

    logical = max(1, int(os.cpu_count() or 1))
    cpu_values = [sample['cpu_pct'] for sample in self.samples]
    rss_values = [sample['rss_mb'] for sample in self.samples]
    return {
      'sample_count': len(self.samples),
      'cpu_pct': {
        'mean': round(statistics.mean(cpu_values), 2),
        'peak': round(max(cpu_values), 2),
        'mean_of_one_core_pct': round(statistics.mean(cpu_values) / logical, 2),
        'peak_of_one_core_pct': round(max(cpu_values) / logical, 2),
      },
      'rss_mb': {
        'mean': round(statistics.mean(rss_values), 2),
        'peak': round(max(rss_values), 2),
      },
    }


class ReusableHTTPServer(ThreadingHTTPServer):
  """HTTP server with address reuse enabled."""

  allow_reuse_address = True


class MockVLMServer:
  """Small OpenAI-compatible mock server for posthoc benchmark runs."""

  def __init__(self, delay_s: float):
    self.delay_s = max(0.0, float(delay_s))
    self.requests: List[Dict[str, Any]] = []
    self._server: Optional[ReusableHTTPServer] = None
    self._thread: Optional[threading.Thread] = None

  @staticmethod
  def _estimate_prompt_tokens(messages: List[dict]) -> int:
    """Approximate prompt tokens from chat-completions messages."""
    total = 0
    for message in messages or []:
      total += 6
      content = message.get('content')
      if isinstance(content, str):
        total += estimate_text_tokens(content)
        continue
      if isinstance(content, list):
        for item in content:
          if not isinstance(item, dict):
            continue
          if item.get('type') == 'text':
            total += estimate_text_tokens(str(item.get('text') or ''))
          elif item.get('type') == 'image_url':
            total += 255
    return int(total)

  def start(self) -> None:
    """Start the mock server on a free localhost port."""
    parent = self

    class Handler(BaseHTTPRequestHandler):
      def do_POST(self) -> None:
        length = int(self.headers.get('Content-Length', '0') or '0')
        raw_body = self.rfile.read(length)
        try:
          body = json.loads(raw_body.decode('utf-8'))
        except Exception:
          body = {}

        prompt_tokens = parent._estimate_prompt_tokens(body.get('messages') or [])
        response_obj = {
          'overall_comment': '整体动作完成得比较稳，节奏也比较统一。',
          'group_comments': {
            '1-1': '第一组完成得比较顺。',
            '2-2': '第二组保持得也比较稳。',
            '3-3': '第三组注意再把躯干稳一点。',
            '4-4': '最后一组整体完成度也不错。',
          },
          'issue_analysis': [
            {
              'tag': 'torso_lean',
              'count': 4,
              'text': '下蹲时身体前倾稍微偏多，建议收紧核心、抬胸，让重心更稳定。',
            }
          ],
          'next_goal': '下一次把每次深蹲都做得更稳一些。',
          'coach_summary': '整体不错，下一步重点放在躯干稳定和下蹲控制上。',
        }
        completion_text = json.dumps(response_obj, ensure_ascii=False)
        completion_tokens = estimate_text_tokens(completion_text)

        parent.requests.append({
          'path': self.path,
          'request_bytes': len(raw_body),
          'prompt_tokens_est': prompt_tokens,
          'completion_tokens_est': completion_tokens,
        })

        if parent.delay_s > 0:
          time.sleep(parent.delay_s)

        payload = {
          'id': 'chatcmpl-mock',
          'object': 'chat.completion',
          'created': int(time.time()),
          'model': 'mock-vlm',
          'choices': [
            {
              'index': 0,
              'message': {'role': 'assistant', 'content': completion_text},
              'finish_reason': 'stop',
            }
          ],
          'usage': {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
          },
        }
        data = json.dumps(payload).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

      def log_message(self, format: str, *args: Any) -> None:
        return

    self._server = ReusableHTTPServer(('127.0.0.1', 0), Handler)
    self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
    self._thread.start()

  @property
  def base_url(self) -> str:
    """Return the server base URL."""
    if self._server is None:
      raise RuntimeError('Mock VLM server has not been started.')
    host, port = self._server.server_address
    return f'http://{host}:{port}'

  def stop(self) -> None:
    """Stop the server."""
    if self._server is not None:
      self._server.shutdown()
      self._server.server_close()
    if self._thread is not None:
      self._thread.join(timeout=1.0)

  def summary(self) -> Dict[str, Any]:
    """Return a compact request summary."""
    return {
      'delay_s': self.delay_s,
      'request_count': len(self.requests),
      'requests': self.requests,
    }

  def summary_since(self, start_idx: int) -> Dict[str, Any]:
    """Return only the requests captured after a given index."""
    subset = self.requests[start_idx:]
    return {
      'delay_s': self.delay_s,
      'request_count': len(subset),
      'requests': subset,
    }


def validate_real_vlm_config(cfg: DemoConfig) -> None:
  """Ensure real posthoc VLM mode has the needed config fields."""
  missing = []
  for key in ['base_url', 'api_key', 'model_name']:
    if not str(cfg.get('vlm', key, default='') or '').strip():
      missing.append(f'vlm.{key}')
  if missing:
    joined = ', '.join(missing)
    raise ValueError(f'--posthoc-vlm real requires these config fields: {joined}')


def prepare_config(base_cfg: DemoConfig, posthoc_vlm: str, mock_server: Optional[MockVLMServer]) -> DemoConfig:
  """Build the per-run config used by run_demo."""
  cfg = clone_config(base_cfg)
  cfg.raw.setdefault('vlm', {})
  if posthoc_vlm == 'off':
    cfg.raw['vlm']['posthoc'] = False
  elif posthoc_vlm == 'mock':
    if mock_server is None:
      raise RuntimeError('Mock VLM mode requested without a mock server.')
    cfg.raw['vlm']['posthoc'] = True
    cfg.raw['vlm']['base_url'] = mock_server.base_url
    cfg.raw['vlm']['api_key'] = 'mock-key'
    cfg.raw['vlm']['model_name'] = 'mock-vlm'
  elif posthoc_vlm == 'real':
    cfg.raw['vlm']['posthoc'] = True
    validate_real_vlm_config(cfg)
  else:
    raise ValueError(f'Unsupported posthoc VLM mode: {posthoc_vlm}')
  return cfg


def find_report_json(out_dir: Path) -> Path:
  """Locate the report JSON emitted by run_demo."""
  matches = sorted(out_dir.glob('*_report.json'))
  if not matches:
    raise FileNotFoundError(f'No report JSON found in {out_dir}')
  return matches[-1]


def extract_run_metrics(
  out_dir: Path,
  report_json: Path,
  end_to_end_s: float,
  sampler_summary: Dict[str, Any],
  mock_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
  """Build the benchmark payload for one run."""
  report = json.loads(report_json.read_text(encoding='utf-8'))
  eng = report.get('engineering', {}) or {}
  perf = eng.get('perf', {}) or {}
  frames = int(eng.get('frames') or 0)
  loop_wall_s = float(eng.get('wall_clock_s') or 0.0)
  throughput_fps = float(frames / loop_wall_s) if loop_wall_s > 0 else 0.0

  return {
    'out_dir': str(out_dir.resolve()),
    'report_json': str(report_json.resolve()),
    'video_duration_s': eng.get('duration_s'),
    'frames': frames,
    'input_fps': eng.get('fps'),
    'resolution': eng.get('resolution'),
    'rep_count': report.get('rep_count'),
    'avg_score': report.get('avg_score'),
    'issue_counts': report.get('issue_counts'),
    'end_to_end_s': float(end_to_end_s),
    'loop_wall_s': loop_wall_s,
    'tail_after_loop_s': float(max(0.0, end_to_end_s - loop_wall_s)),
    'throughput_fps': throughput_fps,
    'perf_ms': {
      'yolo_infer_ms': perf.get('yolo_infer_ms'),
      'smooth_ms': perf.get('smooth_ms'),
      'engine_ms': perf.get('engine_ms'),
      'render_ms': perf.get('render_ms'),
      'total_loop_ms': perf.get('total_loop_ms'),
    },
    'process_usage': sampler_summary,
    'vlm': (eng.get('vlm') or {}),
    'mock_vlm': mock_summary,
  }


def run_single_benchmark(
  args: argparse.Namespace,
  base_cfg: DemoConfig,
  run_idx: int,
  measured: bool,
  mock_server: Optional[MockVLMServer],
) -> Dict[str, Any]:
  """Run one warmup or measured benchmark pass."""
  cfg = prepare_config(base_cfg, args.posthoc_vlm, mock_server)
  base_url = str(cfg.get('vlm', 'base_url', default='') or '')
  sampler = ProcessSampler(args.cpu_sample_interval_s)
  mock_start_idx = len(mock_server.requests) if mock_server is not None else 0

  if measured:
    sampler.start()
  try:
    with localhost_no_proxy(is_local_base_url(base_url)):
      t0 = time.perf_counter()
      out_dir = run_demo(
        video_path=args.video,
        exercise=args.exercise,
        goal_reps=int(args.goal),
        cfg=cfg,
        show=bool(args.show),
        save_video=bool(args.save_video),
        yolo_model=args.yolo_model,
        realtime_sync=bool(args.realtime_sync),
        user_mode=args.user_mode,
        coach_mode=args.coach_mode,
        coach_group_n=int(args.coach_group_n),
        coach_interval_s=float(args.coach_interval_s),
        tip_hold_s=float(args.tip_hold_s),
        enable_vlm_rt=False,
        vlm_min_interval_s=3.0,
        enable_vlm_posthoc=bool(cfg.get('vlm', 'posthoc', default=False)),
        vlm_rag=bool(args.vlm_rag),
        vlm_debug=bool(args.vlm_debug),
        terminal_verbose=bool(args.verbose),
      )
      end_to_end_s = time.perf_counter() - t0
  except Exception:
    if measured:
      sampler.stop()
    raise

  out_path = Path(out_dir)
  report_json = find_report_json(out_path)

  if not measured:
    shutil.rmtree(out_path, ignore_errors=True)
    return {
      'run_index': run_idx,
      'kind': 'warmup',
      'end_to_end_s': float(end_to_end_s),
    }

  sampler_summary = sampler.stop()
  mock_summary = mock_server.summary_since(mock_start_idx) if mock_server is not None else None
  result = extract_run_metrics(
    out_dir=out_path,
    report_json=report_json,
    end_to_end_s=end_to_end_s,
    sampler_summary=sampler_summary,
    mock_summary=mock_summary,
  )
  result['run_index'] = run_idx
  result['kind'] = 'measured'

  if args.cleanup_run_outputs:
    shutil.rmtree(out_path, ignore_errors=True)
    result['out_dir'] = '<cleaned>'
    result['report_json'] = '<cleaned>'

  return result


def build_output_path(args: argparse.Namespace) -> Path:
  """Resolve the output JSON path for the benchmark summary."""
  if args.output_json:
    out_path = Path(args.output_json)
    if not out_path.is_absolute():
      out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path

  out_root = PROJECT_ROOT / 'outputs' / 'benchmarks'
  out_root.mkdir(parents=True, exist_ok=True)
  video_stem = safe_slug(Path(args.video).stem, default='video')
  label = f"_{safe_slug(args.label, default='label')}" if args.label else ''
  ts = datetime.now().strftime('%Y%m%d_%H%M%S')
  filename = f'benchmark_{args.exercise}_{video_stem}_{args.posthoc_vlm}{label}_{ts}.json'
  return out_root / filename


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
  """Aggregate measured benchmark runs into a concise summary."""
  if not results:
    return {}

  def values(path: List[str]) -> List[float]:
    out: List[float] = []
    for item in results:
      cur: Any = item
      ok = True
      for key in path:
        if not isinstance(cur, dict) or key not in cur or cur[key] is None:
          ok = False
          break
        cur = cur[key]
      if ok:
        out.append(float(cur))
    return out

  agg = {
    'end_to_end_s': calc_stats(values(['end_to_end_s'])),
    'loop_wall_s': calc_stats(values(['loop_wall_s'])),
    'tail_after_loop_s': calc_stats(values(['tail_after_loop_s'])),
    'throughput_fps': calc_stats(values(['throughput_fps'])),
    'perf_ms': {
      'yolo_infer_ms': calc_stats(values(['perf_ms', 'yolo_infer_ms'])),
      'render_ms': calc_stats(values(['perf_ms', 'render_ms'])),
      'total_loop_ms': calc_stats(values(['perf_ms', 'total_loop_ms'])),
    },
    'process_usage': {
      'cpu_pct_mean': calc_stats(values(['process_usage', 'cpu_pct', 'mean'])),
      'cpu_pct_peak': calc_stats(values(['process_usage', 'cpu_pct', 'peak'])),
      'rss_mb_peak': calc_stats(values(['process_usage', 'rss_mb', 'peak'])),
    },
  }
  return agg


def print_run_summary(result: Dict[str, Any], total_runs: int) -> None:
  """Print a concise per-run summary to stdout."""
  run_idx = result.get('run_index')
  perf = result.get('perf_ms', {}) or {}
  usage = result.get('process_usage', {}) or {}
  rss = (usage.get('rss_mb') or {}).get('peak')
  print(
    f"[run {run_idx}/{total_runs}] "
    f"end_to_end={result.get('end_to_end_s', 0.0):.3f}s "
    f"loop={result.get('loop_wall_s', 0.0):.3f}s "
    f"throughput={result.get('throughput_fps', 0.0):.2f}fps "
    f"total_loop={float(perf.get('total_loop_ms') or 0.0):.2f}ms "
    f"peak_rss={float(rss or 0.0):.2f}MB"
  )


def main() -> None:
  """CLI entrypoint."""
  os.chdir(PROJECT_ROOT)
  parser = build_arg_parser()
  args = parser.parse_args()

  if args.runs <= 0:
    raise ValueError('--runs must be >= 1')
  if args.warmup_runs < 0:
    raise ValueError('--warmup-runs must be >= 0')

  video_path = Path(args.video)
  if not video_path.is_absolute():
    video_path = PROJECT_ROOT / video_path
  if not video_path.exists():
    raise FileNotFoundError(f'Video not found: {video_path}')
  args.video = str(video_path.resolve())

  base_cfg = load_config(str(PROJECT_ROOT / 'config.default.json'), args.config)
  output_json = build_output_path(args)

  print(f'[benchmark] video={args.video}')
  print(f'[benchmark] exercise={args.exercise} | runs={args.runs} | warmup={args.warmup_runs} | posthoc_vlm={args.posthoc_vlm}')

  mock_server: Optional[MockVLMServer] = None
  if args.posthoc_vlm == 'mock':
    mock_server = MockVLMServer(delay_s=args.mock_vlm_delay_s)
    mock_server.start()
    print(f'[benchmark] mock VLM server started at {mock_server.base_url} (delay={args.mock_vlm_delay_s:.3f}s)')

  warmups: List[Dict[str, Any]] = []
  measured: List[Dict[str, Any]] = []
  try:
    for idx in range(1, args.warmup_runs + 1):
      print(f'[warmup {idx}/{args.warmup_runs}] starting...')
      warmups.append(run_single_benchmark(args, base_cfg, idx, measured=False, mock_server=mock_server))

    for idx in range(1, args.runs + 1):
      result = run_single_benchmark(args, base_cfg, idx, measured=True, mock_server=mock_server)
      measured.append(result)
      print_run_summary(result, args.runs)
  finally:
    if mock_server is not None:
      mock_server.stop()

  summary = {
    'created_at_utc': now_utc_iso(),
    'script': str((PROJECT_ROOT / 'benchmark.py').resolve()),
    'argv': sys.argv[1:],
    'system': collect_system_info(),
    'settings': {
      'video': args.video,
      'exercise': args.exercise,
      'goal': args.goal,
      'config': args.config,
      'config_name': base_cfg.name,
      'yolo_model': args.yolo_model,
      'runs': args.runs,
      'warmup_runs': args.warmup_runs,
      'show': bool(args.show),
      'save_video': bool(args.save_video),
      'realtime_sync': bool(args.realtime_sync),
      'user_mode': args.user_mode,
      'coach_mode': args.coach_mode,
      'coach_group_n': args.coach_group_n,
      'coach_interval_s': args.coach_interval_s,
      'tip_hold_s': args.tip_hold_s,
      'posthoc_vlm': args.posthoc_vlm,
      'mock_vlm_delay_s': args.mock_vlm_delay_s,
      'vlm_rag': bool(args.vlm_rag),
      'vlm_debug': bool(args.vlm_debug),
      'cpu_sample_interval_s': args.cpu_sample_interval_s,
      'cleanup_run_outputs': bool(args.cleanup_run_outputs),
    },
    'warmups': warmups,
    'runs': measured,
    'aggregate': aggregate_results(measured),
  }

  output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
  print(f'[benchmark] summary written to {output_json.resolve()}')


if __name__ == '__main__':
  main()
