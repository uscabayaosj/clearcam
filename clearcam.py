from tinygrad.tensor import Tensor
from utils.runtime_paths import model_asset as fetch
from utils import native_session
from detection.yolov9 import YOLOv9
from detection import coreml_yolo


def make_detector(model_size, model_res):
  """Prefer the Core ML model on the Neural Engine; fall back to tinygrad YOLO."""
  if os.environ.get('CLEARCAM_NO_COREML') != '1':
    for candidate in (os.environ.get('CLEARCAM_MODEL_DIR'), 'models'):
      if not candidate: continue
      package = Path(candidate) / coreml_yolo.MODEL_FILE
      if package.exists():
        try:
          detector = coreml_yolo.CoreMLYolo(package)
          print('Detection: Core ML (Neural Engine),', coreml_yolo.MODEL_FILE)
          return detector
        except Exception as error:
          print('Core ML unavailable, using tinygrad YOLO:', error)
  print('Detection: tinygrad YOLO', model_size)
  return YOLOv9(model_size, model_res)
import numpy as np
from pathlib import Path
import cv2
from collections import defaultdict, deque
import time, sys
import json
import http
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import threading
import shutil
from datetime import datetime, time as time_obj
import uuid
import urllib
from urllib.parse import urlparse, parse_qs
from pathlib import Path
import struct
from urllib.parse import unquote, quote
import zlib
from utils.db import db
from utils import keychain
from utils import household
from utils import macos_notifications
from utils.local_descriptions import LocalDescriptions, read_description
import multiprocessing
import re
import base64
from utils.helpers import send_notif, find_ffmpeg, export_clip, upload_file, encrypt_file, export_and_upload, jit_infer
import pickle
import signal
import math
from utils.recording_timeline import contained_path, read_timeline, event_timing, write_event_time, expired_recording_dirs, position_at, live_playlist

# RTSP URL
# Video capture thread
import subprocess
import threading
import time
import numpy as np
import cv2
from datetime import datetime
import os
import threading
from utils.helpers import BASE_DIR
from ocsort_tracker import ocsort

(BASE_DIR / "cameras").mkdir(parents=True, exist_ok=True)
models = {1: "t", 2: "s", 3: "m", 4: "c", 5: "e", 6: "nano", 7: "small", 8:"medium", 9:"large"}
local_descriptions = LocalDescriptions()
household_store = household.HouseholdStore(BASE_DIR)
notifications_muted_until = 0.0  # wall clock; 0 means notifications are on


def notifications_muted():
  return time.time() < notifications_muted_until


def _face_regions(frame):
  """BlazeFace is a close-range detector: distant faces on a wide camera frame
  need zoomed regions. Full frame first, then an overlapping half-size grid."""
  yield frame
  h, w = frame.shape[:2]
  if min(h, w) < 220: return
  half_h, half_w = h // 2, w // 2
  for top in (0, h // 4, h - half_h):
    for left in (0, w // 4, w - half_w):
      yield frame[top:top + half_h, left:left + half_w]


def face_embedding(frame_bgr):
  """Embedding for the first detectable face in a frame, or None. Main loop only."""
  object_finder.init_face()
  for region in _face_regions(frame_bgr):
    face = object_finder.img_to_face(region)
    if face is not None:
      return object_finder.adaface(Tensor(face).contiguous()).numpy().flatten().tolist()
  return None


def recognize_household(image_path, frame_bgr):
  """Match the event frame against enrolled members; recognition never blocks recording."""
  if not household_store.has_members(): return None
  try:
    embedding = face_embedding(frame_bgr)
    match = household_store.match(embedding) if embedding else None
    household.write_people(image_path, [match['name']] if match else [])
    return match
  except Exception as error:
    print('Household recognition failed:', error)
    return None


def enroll_household_face(name, image_path):
  """Enroll one face from a saved event image. Runs on the main loop via the
  task queue, so it must never raise — an exception here would stop recording."""
  try:
    frame = cv2.imread(str(image_path))
    if frame is None: return dict(error='Could not read that image')
    embedding = face_embedding(frame)
    if embedding is None: return dict(error='No face was found in that image')
    member_id = household_store.add_sample(name, embedding)
    match = household_store.match(embedding)
    household.write_people(image_path, [match['name']] if match else [])
    return dict(id=member_id)
  except Exception as error:
    print('Household enrollment failed:', error)
    return dict(error='Enrollment failed — see the engine log')

def camera_sources():
  """Return usable sources, migrating legacy plaintext RTSP URLs into Keychain."""
  stored = database.run_get("links", None)
  sources = {}
  for camera_name, value in stored.items():
    if not isinstance(value, str) or not value.strip():
      continue
    if keychain.is_reference(value):
      sources[camera_name] = keychain.retrieve(camera_name, value)
    else:
      sources[camera_name] = value
      if value.startswith("rtsp://"):
        database.run_put("links", camera_name, keychain.store(camera_name, value))
  return sources

class RollingClassCounter:
  def __init__(self, window_seconds=None, max=None, classes=None, sched=[[0,86399],True,True,True,True,True,True,True],cam_name=None, desc=None, threshold=0.28):
    self.window = window_seconds
    self.data = defaultdict(deque)
    self.max = max
    self.classes = classes
    self.last_det = 0
    self.sched = sched
    self.cam_name = cam_name
    self.is_on = True
    self.is_notif = True
    self.zone = True
    self.reset = False
    self.new = True
    self.desc = desc
    self.desc_emb = None
    self.threshold = threshold

  def add(self, class_id):
    if self.classes is not None and class_id not in self.classes: return
    now = time.time()
    self.data[class_id].append(now)
    self.cleanup(class_id, now)

  def cleanup(self, class_id, now):
    q = self.data[class_id]
    window = self.window if self.window else (60 if self.is_notif else 1)
    while window and q and now - q[0] > window:
      q.popleft()

  def reset_counts(self):
    for class_id, _ in self.data.items():
       self.data[class_id] = deque() # todo, use in reset endpoint?
    self.reset = True

  def get_counts(self):
    window = self.window if self.window else (60 if self.is_notif else 1)
    max_reached = False
    now = time.time()
    counts = {}
    for class_id, q in self.data.items():
      while window and q and now - q[0] > window:
        q.popleft()
      if q:
        counts[class_id] = len(q)
        if self.max and len(q) >= self.max: max_reached = True
    return counts, max_reached
  
  def is_active(self, offset=0):
    # .get: an alert row can exist before init_cam registers the camera (hot-add race).
    if not alerts_on.get(self.cam_name, True): return False
    if not getattr(self, "is_on", False): return False
    if not self.sched: return True
    # A malformed stored schedule must never disable detection for the camera.
    if len(self.sched) < 8 or not isinstance(self.sched[0], (list, tuple)) or len(self.sched[0]) < 2: return True
    now = time.localtime()
    time_of_day = now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec
    if not self.sched[now.tm_wday + 1]: return False
    window = self.window if self.window else (60 if self.is_notif else 1)
    return time_of_day < self.sched[0][1] and time_of_day > ((self.sched[0][0] - window) + offset)

def write_png(filename, array):
    array = array[..., ::-1]  # BGR to RGB
    height, width, _ = array.shape
    png_signature = b"\x89PNG\r\n\x1a\n"
    def chunk(chunk_type, data):
        return (struct.pack("!I", len(data)) +
                chunk_type +
                data +
                struct.pack("!I", zlib.crc32(chunk_type + data) & 0xffffffff))
    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw_data = b"".join(b"\x00" + array[y].tobytes() for y in range(height))
    compressed = zlib.compress(raw_data, 9)
    png_bytes = (
        png_signature +
        chunk(b"IHDR", ihdr) +
        chunk(b"IDAT", compressed) +
        chunk(b"IEND", b"")
    )
    with open(filename, "wb") as f:
        f.write(png_bytes)

import numpy as np

def draw_rectangle_numpy(img, pt1, pt2, color, thickness=1):
    x1, y1 = pt1
    x2, y2 = pt2
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1]-1, x2), min(img.shape[0]-1, y2)
    if thickness == -1:  # fill
        img[y1:y2+1, x1:x2+1] = color
    else:
        img[y1:y1+thickness, x1:x2+1] = color
        img[y2-thickness+1:y2+1, x1:x2+1] = color
        img[y1:y2+1, x1:x1+thickness] = color
        img[y1:y2+1, x2-thickness+1:x2+1] = color
    return img


def is_vod(cam_name): return (BASE_DIR / "cameras" / cam_name / "streams" / "video").is_dir()

def _get_stream_resolution(src):
  ffmpeg_path = find_ffmpeg()
  command = [ffmpeg_path, "-i", src]
  try:
    result = subprocess.run(
        command,
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        text=True,
        timeout=10
    )
    output = result.stderr
    match = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", output)
    if match:
      width, height = map(int, match.groups())
      return width, height
    return 1920, 1080
  except Exception as e:
    return 1920, 1080

class VideoCapture:
  def __init__(self):
    self.stopping = threading.Event()
    self.restart_lock = threading.RLock()
    self.output_dir_raw = {}
    self.frame_num = {}
    self.last_frame_num = {}
    self.vod = {}
    # objects in scene count
    self.counter = {}
    self.object_set = {}
    self.object_set_zone = {}

    self.src = {}
    self.width, self.height = {}, {}
    self.proc = {}
    self.hls_proc = {}
    self.running = {}

    self.raw_frame = {}
    self.last_preds = {}
    self.last_frames = {}

    self.settings = {}
    self.count = {}
    self.prev_time = {}
    self.current_stream_dir_raw = {}
    self.pred_occs = {}
    self.tracker = {}
    self.cap = {}
    self.src_fps = {}
    self.last_det = {}
    self.last_live_check = {}
    self.last_counter_update = {}
    self.last_preview_time = {}
    self.last_live_seg = {}
    self.start_time = {}
    self.filename = {}
    self.alert_counters = {}
    self.live_link = {}
    self.live_link_lock = {}
    self.pipeline = {}

    #self.last_shapes_time = time.time()
    #self.det_shapes = []

  def init_cam(self, cam_name, src):
    self.pipeline[cam_name] = {"last_frame": None, "last_inference": None, "last_event": None, "state": "connecting", "error": None}
    self.counter[cam_name] = RollingClassCounter(cam_name=cam_name, window_seconds=float('inf'))
    self.src[cam_name] = src # todo
    self.last_frames[cam_name] = deque(maxlen=2)
    self.vod[cam_name] = src.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
    self.frame_num[cam_name] = -1
    self.last_frame_num[cam_name] = -1
    self.object_set[cam_name] = set()
    self.object_set_zone[cam_name] = set()
    self.running[cam_name] = True
    self.output_dir_raw[cam_name] = BASE_DIR / "cameras" / f'{cam_name}' / "streams"
    self.last_preds[cam_name] = []
    self.raw_frame[cam_name] = None
    self.width[cam_name], self.height[cam_name] = _get_stream_resolution(src)
    self.settings[cam_name] = None
    self.start_time[cam_name] = None
    
    self.alert_counters[cam_name] = database.run_get("alerts",cam_name)
    if not self.alert_counters[cam_name]:
      self.alert_counters[cam_name] = dict()

    self.last_det[cam_name] = -1
    self.last_live_check[cam_name] = time.time()
    self.last_live_seg[cam_name] = time.time()
    self.last_preview_time[cam_name] = None
    self.last_counter_update[cam_name] = time.time()
    self.pred_occs[cam_name] = {}
    self.hls_proc[cam_name], self.proc[cam_name] = self._open_ffmpeg(cam_name)
    self.tracker[cam_name] = ocsort.OCSort(max_age=100)
    self.count[cam_name] = 0
    self.prev_time[cam_name] = time.time()
    self.current_stream_dir_raw[cam_name] = self._get_new_stream_dir(cam_name)
    self.filename[cam_name] = None
    self.live_link_lock[cam_name] = threading.Lock()
    alerts_on[cam_name] = True

  def start(self):
    cam_check = time.time()
    # A stale or partially-created camera row must not be treated as a feed.
    # This is especially important on first launch, when an empty value can be
    # present in the local cache before the user has completed camera setup.
    cams = camera_sources()
    for cam_name in cams.keys():
      print("Starting camera:", cam_name)
      self.init_cam(cam_name=cam_name, src=cams[cam_name])
      threading.Thread(target=self.frame_loop, args=(cam_name,), daemon=True).start() # todo non vod only!
    while True:
      if time.time() - cam_check >= 5:
        cam_check = time.time()
        new_cams = camera_sources()
        for cam_name in new_cams.keys():
          if type(new_cams[cam_name]) != str: continue # todo find cause
          if cam_name not in cams:
            print("Starting camera:", cam_name)
            self.init_cam(cam_name=cam_name, src=new_cams[cam_name])
            if not self.vod[cam_name]: threading.Thread(target=self.frame_loop, args=(cam_name,), daemon=True).start() # todo non vod only
          else:
            if new_cams[cam_name] != cams[cam_name]:
              cams[cam_name] = new_cams[cam_name]
              self.src[cam_name] = new_cams[cam_name]
              self.hls_proc[cam_name], self.proc[cam_name] = self._open_ffmpeg(cam_name)
        cams = new_cams
      for cam_name in cams.keys():
        self.process_frame(cam_name=cam_name) # todo rename alerts_on?
      process_queue()
      if len(object_queue) > 0:
        try:
          img = cv2.imread(object_queue[0])
          img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
          clip_latest_img(img)
          process_latest_face(img)
        except Exception as e: print("error in object processing", object_queue[0], e)
        del object_queue[0] 
      time.sleep(0.01)  # Yield while waiting for new frames; don't spin a CPU core.
           


  def _get_new_stream_dir(self, cam_name):
      timestamp = "video" if self.vod[cam_name] else datetime.now().strftime("%Y-%m-%d")
      stream_dir_raw = self.output_dir_raw[cam_name] / timestamp
      stream_dir_raw.mkdir(parents=True, exist_ok=True)
      return stream_dir_raw

  def _safe_kill_process(self, proc):
    if proc:
      try:
        proc.terminate()
        proc.wait(timeout=5)
      except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
      except Exception:
        pass

  def _open_ffmpeg(self, cam_name):
    with self.restart_lock:
      if self.stopping.is_set(): return self.hls_proc.get(cam_name), self.proc.get(cam_name)
      result = self._open_ffmpeg_locked(cam_name)
      if result is not None: self.hls_proc[cam_name], self.proc[cam_name] = result
      return result

  def _open_ffmpeg_locked(self, cam_name):
    path = self._get_new_stream_dir(cam_name)
    if cam_name in self.proc: self._safe_kill_process(self.proc[cam_name])
    if cam_name in self.hls_proc: self._safe_kill_process(self.hls_proc[cam_name])
    src = self.src[cam_name]
    if type(src) != str: return # todo, fixes a crash, fix cause

    ffmpeg_path = find_ffmpeg()
    
    is_rtsp = src.startswith("rtsp")
    if self.vod[cam_name]:
      command = [
        ffmpeg_path,
        "-i", src,
        "-c:v", "copy",
        "-an",
        "-f", "hls",
        "-hls_time", "2",
        "-hls_list_size", "0",
        "-hls_flags", "independent_segments",
        "-hls_segment_type", "fmp4",
        "-hls_fmp4_init_filename", "init.mp4",
        "-hls_segment_filename", str(path / "seg_%06d.m4s"),
        str(path / "stream.m3u8"),
      ]
      return subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL), None
        
    else:  # Live streams
      # Original live stream pipeline
      command = [
          ffmpeg_path,
          "-loglevel", "error",
          *(["-rtsp_transport", "tcp"] if is_rtsp else []),
          "-fflags", "+genpts",
          "-avoid_negative_ts", "make_zero",
          "-i", src,
          "-c", "copy",
          "-an",
          "-f", "hls",
          "-hls_time", "2",
          "-hls_list_size", "0",
          "-hls_playlist_type", "event",
          "-hls_flags", "append_list+independent_segments+temp_file+program_date_time",
          "-hls_segment_filename", str(path / f"stream_{uuid.uuid4().hex}_%06d.ts"),
          str(path / "stream.m3u8")
      ]
      # Inherit stderr so camera connection failures reach the engine log.
      hls_proc = subprocess.Popen(command, stdout=subprocess.DEVNULL)
      self.hls_proc[cam_name] = hls_proc
      if is_rtsp:
        # Detection decodes its own RTSP session instead of tailing the recorded
        # playlist. Tailing stalls on recorded-timeline damage (a reconnect once
        # wrote a 4.5-hour EXTINF that wedged every decoder session), and a direct
        # feed starts immediately instead of waiting for segments.
        time.sleep(2)
        if self.start_time[cam_name] is None: self.start_time[cam_name] = time.time()
        command = [
            ffmpeg_path,
            "-loglevel", "error",
            "-rtsp_transport", "tcp",
            "-i", src,
            "-an",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-vf", f"scale={self.width[cam_name]}:{self.height[cam_name]}",
            "-fps_mode", "vfr",
            "-threads", "1",
            "-"
        ]
        return hls_proc, subprocess.Popen(command, stdout=subprocess.PIPE)
      time.sleep(15)
      if self.start_time[cam_name] is None: self.start_time[cam_name] = time.time()

      command = [
          ffmpeg_path,
          "-live_start_index", "-1",
          "-i", str(path / "stream.m3u8"),
          "-loglevel", "error",
          "-an",
          "-f", "rawvideo",
          "-pix_fmt", "bgr24",
          "-vf", f"scale={self.width[cam_name]}:{self.height[cam_name]}",
          "-fps_mode", "vfr",
          "-threads", "1",
          "-"
      ]
      return hls_proc, subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

  def save_object(self, p, ts=0, cam_name=None):
    p = np.array([p.tlwh[0],p.tlwh[1],(p.tlwh[0]+p.tlwh[2]),(p.tlwh[1]+p.tlwh[3]),p.score,p.class_id,p.track_id])
    timestamp = "video" if self.vod[cam_name] else datetime.now().strftime("%Y-%m-%d")
    filepath = BASE_DIR / "cameras" / f"{cam_name}/objects/{timestamp}"
    filepath.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "cameras" / f"{cam_name}/faces/{timestamp}").mkdir(parents=True, exist_ok=True)
    object_filename = filepath / f"{ts}_{int(p[6])}_{int(p[5])}.jpg"
    x1, y1, x2, y2 = map(int, (p[0], p[1], p[2], p[3]))
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    hw = (x2 - x1) // 2
    hh = (y2 - y1) // 2
    hw *= 2
    hh *= 2
    x1_new = cx - hw
    x2_new = cx + hw
    y1_new = cy - hh
    y2_new = cy + hh
    H, W = self.last_frames[cam_name][-1].shape[:2]
    x1_new = max(0, min(x1_new, W))
    x2_new = max(0, min(x2_new, W))
    y1_new = max(0, min(y1_new, H))
    y2_new = max(0, min(y2_new, H))
    if (y2_new - y1_new) < 100 or (x2_new - x1_new) < 100: return # too small
    crop = self.last_frames[cam_name][-1][y1_new:y2_new, x1_new:x2_new]
    cv2.imwrite(str(object_filename), crop)
    if global_settings.use_clip or global_settings.use_face: object_queue.append(object_filename)

  def frame_loop(self, cam_name):
    fail_count = 0
    frame_size = self.width[cam_name] * self.height[cam_name] * 3
    while not self.stopping.is_set() and (BASE_DIR / "cameras" / cam_name).exists():
      try:
        if self.hls_proc[cam_name].poll() is not None:
          self.pipeline[cam_name].update(state="recorder_offline", last_frame=None)
          self.hls_proc[cam_name], self.proc[cam_name] = self._open_ffmpeg(cam_name)
          continue
        raw_bytes = self.proc[cam_name].stdout.read(frame_size)
        if len(raw_bytes) != frame_size:
          fail_count += 1
          if fail_count > 5:
            print(f"{cam_name} FFmpeg frame read failed (count={fail_count}), restarting stream")
            self.hls_proc[cam_name], self.proc[cam_name] = self._open_ffmpeg(cam_name)
            fail_count = 0
          time.sleep(0.5)
          continue  # Never reshape a partial/empty read into an image.
        else:
          fail_count = 0
        self.raw_frame[cam_name] = np.frombuffer(raw_bytes, np.uint8).reshape((self.height[cam_name], self.width[cam_name], 3))
        self.frame_num[cam_name] += 1
        self.pipeline[cam_name]["last_frame"] = time.time()
        time.sleep(1 / 100)
      except Exception as e:
        print("Error in frame_loop:", e, cam_name)
        time.sleep(1)

  def process_frame(self, cam_name):
    try:
      if self.vod[cam_name]:
        if cam_name not in self.cap:
          self.cap[cam_name] = cv2.VideoCapture(self.src[cam_name])
          self.src_fps[cam_name] = self.cap[cam_name].get(cv2.CAP_PROP_FPS) or 30

        self.cap[cam_name].grab()  # skip for max fps
        ret, frame = self.cap[cam_name].read()
        self.last_frames[cam_name].append(frame)
        if not ret or cam_name not in database.run_get("links", None):
          self.running[cam_name] = False
          if "Processing" not in database.run_get("analysis_prog", cam_name): database.run_put("analysis_prog", cam_name, {"Tracking":100}) # todo stop when done?
        else:
          self.last_preds[cam_name], _ = self.run_inference(frame, cam_name=cam_name)
          database.run_put("analysis_prog", cam_name, {"Tracking":self.cap[cam_name].get(cv2.CAP_PROP_POS_FRAMES)/self.cap[cam_name].get(cv2.CAP_PROP_FRAME_COUNT)*100})
      else:
        frame_num = self.frame_num[cam_name]
        last_frame_num = self.last_frame_num[cam_name]
        if self.raw_frame[cam_name] is None: return
        frame = self.raw_frame[cam_name].copy()
        if frame_num == last_frame_num:
          # Watchdog: a decoder can stay alive but stop producing frames (its
          # blocking read never returns a short read). Kill it so frame_loop's
          # EOF triggers the normal stream restart.
          stale_since = self.pipeline[cam_name].get('last_frame')
          decoder = self.proc.get(cam_name)
          last_kick = getattr(self, 'last_decoder_kick', {}).get(cam_name, 0)
          if (stale_since and time.time() - stale_since > 30 and time.time() - last_kick > 30
              and decoder is not None and decoder.poll() is None):
            print(f"{cam_name} decoder stalled for {int(time.time() - stale_since)}s; restarting stream")
            if not hasattr(self, 'last_decoder_kick'): self.last_decoder_kick = {}
            self.last_decoder_kick[cam_name] = time.time()
            self._safe_kill_process(decoder)
          return

        # don't run inference when no active scheds
        if not any(counter.is_active() for _, counter in self.alert_counters[cam_name].items()): self.last_preds[cam_name] = [] # to remove annotation when no alerts active
        else:
          if not global_settings.userID or alerts_on.get(cam_name, True):
            preds, frame = self.run_inference(frame, cam_name=cam_name)
            self.last_frames[cam_name].append(frame.numpy().copy() if hasattr(frame, 'numpy') else np.array(frame, copy=True))
            self.last_preds[cam_name] = preds.copy()
            self.pipeline[cam_name].update(last_inference=time.time(), state="detecting", error=None)
            self.last_frame_num[cam_name] = self.frame_num[cam_name]

            curr_time = time.time()
            fps = 1 / (curr_time - self.prev_time[cam_name])
            self.prev_time[cam_name] = curr_time
            print(f"\rFPS: {fps:.2f}", end="", flush=True)
          else:
            self.last_frame_num[cam_name] = self.frame_num[cam_name]
            self.last_preds = []

        filtered_preds = self.last_preds[cam_name]

        if self.count[cam_name] > 10:
          if self.last_preview_time[cam_name] is None or time.time() - self.last_preview_time[cam_name] >= 3600: # preview every hour
            self.last_preview_time[cam_name] = time.time()
            self.filename[cam_name] = BASE_DIR / "cameras" / f"{cam_name}/preview.png"
            write_png(self.filename[cam_name], self.raw_frame[cam_name])
          for _,alert in self.alert_counters[cam_name].items():
              if alert.desc is not None: continue
              if not alert.is_active():
                alert.reset_counts()
                continue
              window = alert.window if alert.window else (60 if alert.is_notif else 1)
              if alert.get_counts()[1]:
                if time.time() - alert.last_det >= window and (time.time() - alert.last_det >= window):
                  timestamp = "video" if self.vod[cam_name] else datetime.now().strftime("%Y-%m-%d")
                  filepath = BASE_DIR / "cameras" / f"{cam_name}/event_images/{timestamp}"
                  filepath.mkdir(parents=True, exist_ok=True)
                  annotated_frame = draw_predictions(self.last_frames[cam_name][-1].copy(), filtered_preds, color_dict)
                  # todo alerts can be sent with the wrong thumbnail if two happen quickly, use map
                  ts = int(self.cap[cam_name].get(cv2.CAP_PROP_POS_FRAMES) / self.src_fps[cam_name]) - 5 if self.vod[cam_name] else int(time.time() - self.start_time[cam_name] - 5)
                  event_id = time.time_ns()
                  self.filename[cam_name] = filepath / f"{event_id}_notif.jpg" if alert.is_notif else filepath / f"{event_id}.jpg"
                  if not self.vod[cam_name]: cv2.imwrite(str(self.filename[cam_name]), annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85]) # we've 10MB limit for video file, raw png is 3MB!
                  if not self.vod[cam_name]: write_event_time(self.filename[cam_name], time.time())
                  recognized = None
                  if not self.vod[cam_name]:
                    # Match against enrolled household faces on the un-annotated frame.
                    recognized = recognize_household(self.filename[cam_name], self.last_frames[cam_name][-1])
                  if global_settings.userID is not None and not self.vod[cam_name] and alert.is_notif:
                    title = f"Event Detected ({cam_name})"
                    threading.Thread(target=send_notif, args=(global_settings.userID,title,None), daemon=True).start()
                    if global_settings.key:
                      threading.Thread(target=export_and_upload, kwargs={"cam_name": cam_name, "thumbnail": self.filename[cam_name], "userID": global_settings.userID, "key": global_settings.key, "start": ts, "wait":True}, daemon=True).start()
                  elif not self.vod[cam_name] and alert.is_notif and not notifications_muted():
                    title = f"{recognized['name']} — {cam_name}" if recognized else f"Event detected — {cam_name}"
                    threading.Thread(target=macos_notifications.send, args=(title,), daemon=True).start()
                  if not self.vod[cam_name] and global_settings.use_qwen:
                    local_descriptions.submit(self.filename[cam_name], cam_name, notify=alert.is_notif)
                  self.last_det[cam_name] = time.time()
                  self.pipeline[cam_name]["last_event"] = time.time()
                  alert.last_det = time.time()
          
          if (time.time() - self.last_live_check[cam_name]) >= 5:
            self.last_live_check[cam_name] = time.time()
            link = camera_sources().get(cam_name)
            if type(link) == list: link = link[0] # todo, flakey?
            if link != self.src[cam_name]:
              self.src[cam_name] = link
              self.hls_proc[cam_name], self.proc[cam_name] = self._open_ffmpeg(cam_name)
            if global_settings.userID and not self.vod[cam_name]: threading.Thread(target=self.check_upload_link, args=(cam_name,), daemon=True).start()
          if (time.time() - self.last_counter_update[cam_name]) >= 5: #update counter every 5 secs
            self.last_counter_update[cam_name] = time.time()

            counters = database.run_get("counters", cam_name)
            if counters not in [None, {}]:
              if counters.reset:
                self.counter[cam_name].reset_counts()
                self.counter[cam_name].reset = False
            database.run_put("counters", cam_name, self.counter[cam_name])
            
            alerts = database.run_get("alerts", cam_name)
            for id,a in alerts.items():
              if not a.new: continue
              a.new = False
              database.run_put("alerts", cam_name, a, id=id)
              if a is None:
                del self.alert_counters[cam_name][id]
                continue
              self.alert_counters[cam_name][id] = a
              for c in a.classes: classes.add(str(c))

            self.alert_counters[cam_name] = {i:a for i,a in self.alert_counters[cam_name].items() if i in alerts}
            
            new_settings = database.run_get("settings", cam_name)
            if self.settings[cam_name] is not None and new_settings != self.settings[cam_name] and is_vod(cam_name):
              self.reset_vod(cam_name)
              if "reset" in new_settings: del new_settings["reset"]
            self.settings[cam_name] = new_settings
              
          if global_settings.userID and not self.vod[cam_name] and cam_name in self.live_link and (link:=self.live_link[cam_name]) and (time.time() - self.last_live_seg[cam_name]) >= 4:
            self.last_live_seg[cam_name] = time.time()
            threading.Thread(target=self.upload_live_segment, args=(link, cam_name,), daemon=True).start()
        else: self.count[cam_name]+=1

    except Exception as e:
      print("Error in process_frame:", e, cam_name)
      self.pipeline[cam_name].update(state="error", error=type(e).__name__)
      time.sleep(1)

  def upload_live_segment(self, link, cam_name):
    self.last_live_seg[cam_name] = time.time()
    mp4_filename = f"segment.mp4"
    export_clip(self.current_stream_dir_raw[cam_name], Path(mp4_filename), live=True)
    encrypt_file(Path(mp4_filename), Path(f"""{mp4_filename}.aes"""), global_settings.key)
    Path(mp4_filename).unlink()
    upload_to_r2(file_path=Path(f"""{mp4_filename}.aes"""), signed_url=link)

  def check_upload_link(self, cam_name="camera"):
      query_params = urllib.parse.urlencode({
          "name": quote(cam_name),
          "session_token": global_settings.userID
      })
      url = f"https://clearcam.org/get_stream_upload_link?{query_params}"
      
      req = urllib.request.Request(url)
      with urllib.request.urlopen(req) as response:
          if response.status == 200:
              response_data = json.loads(response.read().decode('utf-8'))
              upload_link = response_data.get("upload_link")
              alerts_on_res = response_data.get("alerts_on")
              with self.live_link_lock[cam_name]: self.live_link[cam_name] = upload_link
              alerts_on[cam_name] = (alerts_on_res == 1)
          else:
              if cam_name in self.live_link: self.live_link[cam_name] = None

  def reset_vod(self, cam_name):
    self.cap[cam_name] = cv2.VideoCapture(self.src[cam_name]) # reset video on settings change
    shutil.rmtree(BASE_DIR / "cameras" / cam_name / "objects", ignore_errors=True)
    shutil.rmtree(BASE_DIR / "cameras" / cam_name / "faces", ignore_errors=True)
    shutil.rmtree(BASE_DIR / "cameras" / cam_name / "event_images", ignore_errors=True)

  def run_inference(self, frame, cam_name):
    global model
    if getattr(model, 'kind', None) == 'coreml':
      preds = model(frame)  # numpy in, numpy out; runs on the Neural Engine
    else:
      frame = Tensor(frame)
      preds = jit_infer(model, frame, yolo_jit_cache).numpy()
    thresh = (self.settings[cam_name].get("threshold") if self.settings[cam_name] else 0.5) or 0.5 #todo clean!
    online_targets = self.tracker[cam_name].update(preds, thresh)
    online_targets = [p for p in online_targets if (classes is None or str(int(p.class_id)) in classes)]
    preds = []
    for x in online_targets:
      if x.tracklet_len < 1: continue # dont alert for 1 frame, too many false positives.  
      # add to objects, regarless of speed
      if x.track_id not in self.pred_occs[cam_name]: self.pred_occs[cam_name][x.track_id] = [time.time()]
      if (len(self.pred_occs[cam_name][x.track_id]) < 20 and (time.time() - self.pred_occs[cam_name][x.track_id][-1]) > 1) or (time.time() - self.pred_occs[cam_name][x.track_id][-1]) > 10:
        self. pred_occs[cam_name][x.track_id].append(time.time())
        ts = round((self.cap[cam_name].get(cv2.CAP_PROP_POS_FRAMES) / self.src_fps[cam_name]) - 5,1) if self.vod[cam_name] else round((time.time() - self.start_time[cam_name] - 5),1)
        self.save_object(x, ts, cam_name=cam_name)

      if x.speed < 2.5: continue #min speed, don't detect still objects, they jitter too. # TODO what's the best min value?
      outside = False
      if hasattr(self, "settings") and self.settings[cam_name] is not None and self.settings[cam_name].get("coords"):
        scaled_coors = np.array(self.settings[cam_name]["coords"])
        scaled_coors[:,] *= [frame.shape[1], frame.shape[0]] # decimal to full
        outside = point_not_in_polygon([[x.tlwh[0], x.tlwh[1]],[(x.tlwh[0]+x.tlwh[2]), x.tlwh[1]],[(x.tlwh[0]), (x.tlwh[1]+x.tlwh[3])],[(x.tlwh[0]+x.tlwh[2]), (x.tlwh[1]+x.tlwh[3])]], scaled_coors)
        outside = outside ^ self.settings[cam_name]["outside"]
      non_zone_alert = False
      if outside: # check if any alerts don't use zone
        for _, alert in self.alert_counters[cam_name].items():
          if not alert.zone:
            non_zone_alert = True
            break
        if not non_zone_alert and outside: continue
      preds.append(np.array([x.tlwh[0],x.tlwh[1],(x.tlwh[0]+x.tlwh[2]),(x.tlwh[1]+x.tlwh[3]),x.score,x.class_id,x.track_id]))
      if (classes is None or str(int(x.class_id)) in classes):
        new = int(x.track_id) not in self.object_set[cam_name]
        new_in_zone = int(x.track_id) not in self.object_set_zone[cam_name] and not outside
        if new:
          self.object_set[cam_name].add(int(x.track_id))
          self.counter[cam_name].add(int(x.class_id))
        if new_in_zone: self.object_set_zone[cam_name].add(int(x.track_id))
        for _, alert in self.alert_counters[cam_name].items():
          if not alert.get_counts()[1] and ((new and not alert.zone) or (new_in_zone and alert.zone)): alert.add(int(x.class_id))
  
    preds = np.array(preds)
    return preds, frame

  def release(self, cam_name):
      self.running[cam_name] = False
      if cam_name in self.proc: self.proc[cam_name].kill()
      if cam_name in self.hls_proc: self.hls_proc[cam_name].kill()    

def is_bright_color(color):
  r, g, b = color
  brightness = (r * 299 + g * 587 + b * 114) / 1000
  return brightness > 127

def draw_predictions(frame, preds, color_dict):
  for x1, y1, x2, y2, conf, cls, _ in preds:
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    label = f"{class_labels[int(cls)]}:{conf:.2f}"
    color = color_dict[class_labels[int(cls)]]
    frame = draw_rectangle_numpy(frame, (x1, y1), (x2, y2), color, 3)
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    font_color = (0, 0, 0) if is_bright_color(color) else (255, 255, 255)
    frame = draw_rectangle_numpy(frame, (x1, y1 - text_height - 10), (x1 + text_width + 2, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1, cv2.LINE_AA)
  return frame

def point_not_in_polygon(coords, poly):
    n = len(poly)
    for j in range(len(coords)):
      inside = False
      p1x, p1y = poly[0]
      for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if coords[j][1] > min(p1y, p2y):
          if coords[j][1] <= max(p1y, p2y):
            if coords[j][0] <= max(p1x, p2x):
              if p1y != p2y:
                x_intersect = (coords[j][1] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
              else:
                x_intersect = p1x
              if p1x == p2x or coords[j][0] <= x_intersect: inside = not inside
        p1x, p1y = p2x, p2y
      if inside:
        return False
    return True

def run_encode_text(clip, text): return clip.model._encode_text(text, realize=True)

def run_search(clip, image_text, top_k, cam_name, selected_dir): return clip.search(image_text, top_k, cam_name, selected_dir)

def run_clip(clip, im, top_k, cam_name, selected_dir, is_face):
  im = clip.preprocess_face(im) if is_face else clip.preprocess_clip(im)
  if im is not None:
    embedding = clip.adaface(Tensor(im)).numpy() if is_face else jit_infer(clip.model.precompute_embedding, Tensor(im), jit_cache=jit_cache).numpy()
    res = clip.search(None, top_k, cam_name, selected_dir, embedding, is_face)
  else:
    res = []
  return res

class HLSRequestHandler(BaseHTTPRequestHandler):
    def parse_request(self):
        if not super().parse_request(): return False
        if not native_session.authorized(self.headers, self.server.server_port):
            self.send_error(403, 'This engine belongs to the ClearCam app')
            return False
        return True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args): pass # don't print stuff

    def send_results(self, results, start=0, count=100):
      image_data = []
      for path_str, score in results:
        if score < 0.21: break
        img_path = Path(path_str).resolve()
        ts = event_img_info(img_path.stem)["ts"]
        parts = img_path.parts
        cam_index = parts.index("cameras") + 1
        cam = parts[cam_index]
        rel = img_path.relative_to(BASE_DIR / "cameras")
        image_url = '/' + quote(str(rel), safe='/')
        timing = event_timing(img_path)
        image_data.append({
          "url": image_url,
          "timestamp": timing['playback_offset'],
          "filename": img_path.name,
          "cam_name": cam,
          "folder": img_path.parts[-2],
          "score": score,
          "description": read_description(img_path),
          "people": household.read_people(img_path),
          **timing,
        })
      image_data = image_data[start:start+count]
      response_data = {
        "images": image_data,
        "count": len(image_data),
      }
      self.send_200(response_data)

    def send_200(self, body=None):
      self.send_response(200)
      self.send_header("Content-Type", "application/json")
      self.end_headers()
      self.wfile.write(json.dumps(body).encode("utf-8"))

    def get_camera_path(self, cam_name=None):
        if cam_name:
            return BASE_DIR / "cameras" / cam_name / "streams"
        return BASE_DIR / "cameras"
    
    def do_GET(self):
        global notifications_muted_until
        parsed_path = urlparse(self.path)
        parsed_path = parsed_path._replace(path=unquote(parsed_path.path))
        query = parse_qs(parsed_path.query)
        if parsed_path.path == '/native_notifications':
          self.send_200({'notifications': native_session.take_notifications()})
          return
        cam_name = query.get("cam", [None])[0]
        for key in ('cam', 'cam_name'):
          name = query.get(key, [None])[0]
          if name is not None:
            try:
              if not name.strip() or name in ('.', '..') or any(c in name for c in '/\\\x00'): raise ValueError()
              contained_path(BASE_DIR / 'cameras', name)
            except ValueError:
              self.send_error(400, 'Invalid camera name')
              return

        if parsed_path.path == '/replay_position':
          try:
            folder = query.get('folder', [''])[0]
            datetime.strptime(folder, '%Y-%m-%d')
            at = float(query.get('at', [''])[0])
            if not cam_name or not math.isfinite(at): raise ValueError()
            playlist = contained_path(BASE_DIR / 'cameras', cam_name + '/streams/' + folder + '/stream.m3u8')
            self.send_200({'playback_offset': position_at(read_timeline(playlist), at)})
          except ValueError:
            self.send_error(400, 'Invalid replay request')
          return

        if parsed_path.path == "/set_max_storage":
          try:
            max_gb = float(query.get("max", [None])[0])
            if not math.isfinite(max_gb) or max_gb < 1: raise ValueError()
          except (ValueError, TypeError):
            self.send_error(400, "Storage limit must be at least 1 GB")
            return
          self.server.max_gb = max_gb
          database.run_put("max_storage", "all", max_gb)
          self.send_200()
          return
        
        if parsed_path.path == "/get_global_settings":
          self.send_200(secret_settings(global_settings).__dict__)
          return
        if parsed_path.path == "/local_ai_status":
          self.send_200(local_descriptions.status())
          return
        if parsed_path.path == "/get_max_storage":
          self.send_200(body={"max_gb":self.server.max_gb, "warning":getattr(self.server, 'storage_warning', None)})
          return

        if parsed_path.path == "/list_cameras":
          cams = {name: value for name, value in database.run_get("links", None).items() if isinstance(value, str) and value.strip()}
          progs = database.run_get("analysis_prog", None)
          cam_progress = {cam_name: progs.get(cam_name, None) for cam_name in cams}
          self.send_200(cam_progress)
          return

        if parsed_path.path == "/engine_status":
          states = {}
          now = time.time()
          for name, snapshot in list(cam.pipeline.items()):
            state = dict(snapshot)
            counters = list(cam.alert_counters.get(name, {}).values())
            state["active_rules"] = sum(bool(rule.is_active()) for rule in counters)
            state["notification_rules"] = sum(bool(rule.is_active() and rule.is_notif) for rule in counters)
            state["frames"] = cam.frame_num.get(name, -1) + 1
            state["decoder_running"] = bool(cam.proc.get(name) and cam.proc[name].poll() is None)
            state["recorder_running"] = bool(cam.hls_proc.get(name) and cam.hls_proc[name].poll() is None)
            if not state["recorder_running"]: state["state"] = "recorder_offline"
            elif not state["last_frame"] or now - state["last_frame"] > 20: state["state"] = "no_frames"
            elif not state["active_rules"]: state["state"] = "no_active_rules"
            elif not state["last_inference"] or now - state["last_inference"] > 30: state["state"] = "inference_pending"
            states[name] = state
          self.send_200({"cameras": states, "notifications_muted_until": notifications_muted_until or None})
          return

        if parsed_path.path == "/vendor/hls.min.js":
          asset = Path(__file__).parent / "vendor" / "hls.min.js"
          if not asset.is_file():
            self.send_error(503, "Local video player is missing")
            return
          self.send_response(200)
          self.send_header("Content-Type", "application/javascript")
          self.end_headers()
          self.wfile.write(asset.read_bytes())
          return

        if parsed_path.path == "/list_days":          
          base_path = BASE_DIR / "cameras"
          days = set()
          date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
          if os.path.exists(base_path):
            for cam_name in os.listdir(base_path):
              cam_path = os.path.join(base_path, cam_name, "streams")    
              if os.path.exists(cam_path):
                for date_folder in os.listdir(cam_path):
                  date_folder_path = os.path.join(cam_path, date_folder)
                  if os.path.isdir(date_folder_path) and date_pattern.match(date_folder): days.add(date_folder)
          days_list = sorted(list(days), reverse=True, key=lambda x: datetime.strptime(x, "%Y-%m-%d"))
          self.send_200(days_list)
          return

        if parsed_path.path == '/add_camera':
            cam_name = query.get("cam_name", [None])[0]
            src = query.get("src", [None])[0]
            
            if not cam_name or not src:
                self.send_error(400, "Missing cam_name or src")
                return
            if len(cam_name) > 100 or cam_name in (".", "..") or any(c in cam_name for c in ('/', '\\', '\x00')):
                self.send_error(400, "Camera name must be a single name of up to 100 characters")
                return
            
            database.run_put("links", cam_name, keychain.store(cam_name, src))
            self.send_response(302)
            self.send_header('Location', '/')
            self.end_headers()
            return
        
        if parsed_path.path == "/edit_settings":
            if not cam_name:
                self.send_error(400, "Missing cam or id")
                return
            zone = database.run_get("settings", cam_name)
            if zone is None: zone = {}
            coords_json = query.get("coords", [None])[0]
            if coords_json is not None:
              coords = json.loads(coords_json)
              if isinstance(coords, list):
                if len(coords) >= 3:
                  zone["coords"] = [[float(x), float(y)] for x, y in coords]
                else:
                  if "coords" in zone: del zone["coords"]
            zone["is_notif"] = (str(is_notif).lower() == "true") if (is_notif := query.get("is_notif", [None])[0]) is not None else zone.get("is_notif")
            zone["outside"] = (str(outside).lower() == "true") if (outside := query.get("outside", [None])[0]) is not None else zone.get("outside")
            query.get("threshold", [None])[0] is not None and zone.update({"threshold": float(query.get("threshold", [None])[0])}) #need the val  
            database.run_put("settings", cam_name, zone) # todo, key for each
            if (url := query.get("url")) is not None: database.run_put("links", cam_name, keychain.store(cam_name, url[0]))

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return

        if parsed_path.path == "/edit_alert":
            if not cam_name:
                self.send_error(400, "Missing cam or id")
                return

            raw_alerts = database.run_get("alerts", cam_name)
            alert = None
            alert_id = query.get("id", [None])[0]
            is_on = query.get("is_on", [None])[0]
            zone = query.get("zone", [None])[0]
            is_notif = query.get("is_notif", [None])[0]
            desc = query.get("desc", [None])[0]
            threshold = query.get("threshold", [None])[0]
            if threshold is not None: threshold = float(threshold) / 100
            if alert_id is None: # no id, add alert
                window = query.get("window", [None])[0]
                max_count = query.get("max", [None])[0]
                class_ids = query.get("class_ids", [None])[0]
                # Canonical schedule shape: [[start,end], mon..sun booleans].
                sched = json.loads(query.get("sched", ["[[0,86399],true,true,true,true,true,true,true]"])[0])
                if window: window = int(window)
                max_count = int(max_count)
                classes = [int(c.strip()) for c in class_ids.split(",")]
                alert_id = str(uuid.uuid4())
                alert = RollingClassCounter(
                        window_seconds=window,
                        max=max_count,
                        classes=classes,
                        sched=sched,
                        cam_name=cam_name,
                        desc=desc,
                        threshold=threshold
                    )
                raw_alerts[alert_id] = alert
            else:
              if is_on is not None or is_notif is not None or zone is not None:
                if is_on is not None: raw_alerts[alert_id].is_on = str(is_on).lower() == "true"
                if is_notif is not None: raw_alerts[alert_id].is_notif = str(is_notif).lower() == "true"
                if zone is not None: raw_alerts[alert_id].zone = str(zone).lower() == "true"
                if desc is not None: raw_alerts[alert_id].desc = desc
                if threshold is not None: raw_alerts[alert_id].threshold = threshold
                alert = raw_alerts[alert_id]
                alert.new = True
              else:
                del raw_alerts[alert_id]
            if alert is not None:
              database.run_put("alerts", cam_name, alert, alert_id)
            else:
              database.run_delete("alerts", cam_name, alert_id)
            
            # make vod reset
            settings = database.run_get("settings", cam_name)
            if settings is None: settings = {}
            settings["reset"] = True
            database.run_put("settings", cam_name, settings)

 
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return

        if parsed_path.path == "/get_settings":
            zone = database.run_get("settings",cam_name)
            if zone is not None:
              if cam_name in zone and "settings" in zone[cam_name]: zone = zone[cam_name]["settings"]
            else:
              zone = {}
            
            self.send_200(zone)
            return

        if parsed_path.path == "/get_alerts":
            if not cam_name:
                self.send_error(400, "Missing cam parameter")
                return

            raw_alerts = database.run_get("alerts",cam_name)
            alert_info = []
            for key,alert in raw_alerts.items():
                sched = alert.sched if alert.sched else [[0,86399],True,True,True,True,True,True,True]
                alert_info.append({
                    "window": alert.window,
                    "max": alert.max,
                    "classes": list(alert.classes),
                    "id": str(key),
                    "sched": sched,
                    "is_on": alert.is_on,
                    "is_notif": alert.is_notif,
                    "zone": alert.zone,
                    "desc": alert.desc,
                    "threshold": alert.threshold,
                })
            self.send_200(alert_info)
            return

        if parsed_path.path == '/pause_notifications':
            try:
                minutes = float(query.get('minutes', ['60'])[0])
                if not (0 <= minutes <= 24 * 60): raise ValueError()
            except ValueError:
                self.send_error(400, 'minutes must be 0-1440')
                return
            notifications_muted_until = time.time() + minutes * 60 if minutes else 0.0
            self.send_200({'muted_until': notifications_muted_until or None})
            return

        if parsed_path.path == '/household':
            self.send_200(household_store.list_members())
            return

        if parsed_path.path == '/household_delete':
            member_id = query.get('id', [None])[0]
            try:
                removed = household_store.remove(member_id)
            except ValueError:
                self.send_error(400, 'Invalid member id')
                return
            if not removed:
                self.send_error(404, 'No such household member')
                return
            self.send_200({'status': 'deleted'})
            return

        if parsed_path.path == '/household_enroll':
            name = query.get('name', [None])[0]
            image = query.get('image', [None])[0]
            if not name or not image:
                self.send_error(400, 'Missing name or image')
                return
            try:
                image_path = contained_path(BASE_DIR / 'cameras', image.removeprefix('/cameras/').lstrip('/'))
                if image_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'): raise ValueError()
                clean = household.clean_name(name)
            except ValueError as error:
                self.send_error(400, str(error) or 'Invalid enrollment request')
                return
            result = add_to_queue(enroll_household_face, clean, image_path)
            if result.get('error'):
                self.send_error(422, result['error'])
                return
            self.send_200({'status': 'ok', 'id': result['id']})
            return

        if parsed_path.path == '/delete_camera':
            cam_name = query.get("cam_name", [None])[0]
            if not cam_name:
                self.send_error(400, "Missing cam_name parameter")
                return
            
            try:
              shutil.rmtree(BASE_DIR / "cameras" / cam_name, ignore_errors=True)
              if os.path.isfile(database.run_get("links", None)[cam_name]): os.remove(database.run_get("links", None)[cam_name])
              # todo clean
              alerts = database.run_get("alerts", cam_name)
              for id, _ in alerts.items():
                database.run_delete("alerts", cam_name, id=id)
              database.run_delete("links", cam_name)
              keychain.remove(cam_name)
              database.run_delete("analysis_prog", cam_name)
              database.run_delete("settings", cam_name)
              database.run_delete("counters", cam_name)
            except Exception as e:
              self.send_error(500, f"Error deleting camera: {e}")
              return

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"deleted"}')
            return

        if parsed_path.path == "/get_counts": # todo error fetching counts on first
            if not cam_name:
                self.send_error(400, "Missing cam parameter")
                return
            
            counter = database.run_get("counters", cam_name)
            if counter:
              labeled_counts = {
                class_labels[int(k)]: len(v)
                for k, v in counter.data.items()
                if int(k) < len(class_labels)
              }
              self.send_200(labeled_counts)
              return
            else:
              database.run_put("counters", cam_name, RollingClassCounter(cam_name=cam_name))
              self.send_200([])
      

        if parsed_path.path == "/reset_counts":
          if not cam_name:
            self.send_error(400, "Missing cam parameter")
            return
          counter = database.run_get("counters",cam_name)
          if counter: counter.reset_counts()
          database.run_put("counters", cam_name, counter)
          self.send_response(200)
          self.send_header("Content-Type", "application/json")
          self.end_headers()
          self.wfile.write(b"{}")
          return


        if parsed_path.path == '/' and "cam" not in query:
          with open("mainview.html", "r", encoding="utf-8") as f: html = f.read()
          self.send_response(200)
          self.send_header('Content-type', 'text/html')
          self.end_headers()
          self.wfile.write(html.encode('utf-8'))
          return
                            
        if parsed_path.path == '/' or parsed_path.path == f'/{cam_name}':
            # The legacy standalone template is not shipped. Use the maintained
            # main surface rather than returning 200 and crashing mid-response.
            if not Path('cameraview.html').is_file():
                self.send_response(302)
                self.send_header('Location', '/')
                self.end_headers()
                return
            selected_dir = parse_qs(parsed_path.query).get("folder", [datetime.now().strftime("%Y-%m-%d")])[0]
            start_param = parse_qs(parsed_path.query).get("start", [None])[0]
            try:
                start_time = max(float(start_param),0) if start_param is not None else None
            except ValueError:
                start_time = None

            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('cameraview.html', 'r', encoding='utf-8') as f: html = f.read()
            replacements = {
                '{selected_dir}': selected_dir,
                '{class_labels}': json.dumps(class_labels),
                '{start_time}': str(start_time) if start_time is not None else 'null',
                '{cam_name}': cam_name
            }
            for placeholder, value in replacements.items(): html = html.replace(placeholder, value)
            self.wfile.write(html.encode("utf-8"))
            return
        
        requested_path = parsed_path.path.lstrip('/')
        if requested_path.startswith("cameras/"):
            requested_path = requested_path[len("cameras/"):]

        try: # todo
          cam_name = requested_path[:requested_path.index("/")]
          vod = is_vod(cam_name)
          if vod and "preview.png" not in requested_path: requested_path = requested_path.rsplit("/", 2)[0] + "/video/" + requested_path.rsplit("/", 1)[1]
        except Exception:
          pass

        try:
            file_path = contained_path(BASE_DIR / "cameras", requested_path)
        except ValueError:
            self.send_error(403, "Outside camera storage")
            return

        if file_path.name == 'live.m3u8':
            # A wide window keeps a briefly-stalled player inside the playlist
            # instead of chasing segments that already rolled out.
            content = live_playlist(file_path.with_name('stream.m3u8'), window=8)
            if content is None:
                self.send_error(404)
                return
            body = content.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/vnd.apple.mpegurl')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if not file_path.is_file():
            self.send_error(404)
            return

        self.send_response(200)
        if file_path.suffix == '.m3u8':
            self.send_header('Content-Type', 'application/vnd.apple.mpegurl')
            self.send_header('Cache-Control', 'no-cache')
        elif file_path.suffix == '.ts':
            self.send_header('Content-Type', 'video/MP2T')
        elif file_path.suffix == '.png':
            self.send_header('Content-Type', 'image/png')
        elif file_path.suffix in ('.jpg', '.jpeg'):
            self.send_header('Content-Type', 'image/jpeg')
        elif file_path.suffix in ('.mp4', '.m4s'):
            self.send_header('Content-Type', 'video/mp4')
        self.end_headers()

        with open(file_path, 'rb') as f:
            shutil.copyfileobj(f, self.wfile)

    def do_POST(self):
        parsed_path = urlparse(self.path)

        if self.path.startswith("/edit_settings"):
          content_length = int(self.headers.get('Content-Length', 0))
          body = self.rfile.read(content_length)
          data = json.loads(body.decode('utf-8'))
          if os.environ.get('CLEARCAM_NATIVE') == '1' and (data.get('use_clip') or data.get('use_face') or data.get('model_size', 't') != 't' or str(data.get('qwen_size', 2)) != '2'):
            self.send_error(400, 'This alpha includes YOLO tiny and Qwen 2B only')
            return
          # keep userid and key if "True"
          if data["userID"] == True: data["userID"] = global_settings.userID
          if data["key"] == True: data["key"] = global_settings.key
          add_to_queue(db.run_put, database, "global_settings", "all", GlobalSettings(**data))
          add_to_queue(set_settings, GlobalSettings(**data))
          self.send_200([])
          return

        if self.path.startswith("/analyse-footage"):
          params = parse_qs(parsed_path.query)
          filename = params.get("filename", [None])[0]
          chunk = int(params.get("chunk", [0])[0])
          total = int(params.get("total", [1])[0])
          if not filename:
            self.send_error(400, "Missing filename")
            return
          filename = os.path.basename(filename)
          upload_dir = BASE_DIR / "cameras"
          upload_dir.mkdir(exist_ok=True)
          length = int(self.headers.get("Content-Length", 0))
          if length <= 0:
            self.send_error(411, "Content-Length required")
            return
          final_path = upload_dir / filename
          temp_path = upload_dir / f"{filename}.part"
          with open(temp_path, "ab") as f:
            remaining = length
            while remaining > 0:
              data = self.rfile.read(min(1024 * 1024, remaining))
              if not data: break
              f.write(data)
              remaining -= len(data)
          if chunk == total - 1: temp_path.rename(final_path)
          self.send_200([])

        if parsed_path.path == "/event_thumbs":
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length)

            try:
                data = json.loads(raw_body)
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON")
                return

            cam_name     = data.get("cam")
            selected_dir = data.get("folder")
            name_contains = data.get("name_contains")
            image_text   = data.get("image_text")
            similar_img  = data.get("similar_img")
            start = data.get("start")
            count = data.get("count")
            is_face = data.get("is_face") or False
            if is_face and not global_settings.use_face:
              self.send_error(400, "Face search is disabled")
              return
            try:
              start, count = int(start or 0), int(count if count is not None else 100)
              if start < 0 or not 1 <= count <= 200: raise ValueError()
              if cam_name: contained_path(BASE_DIR / "cameras", cam_name)
              if selected_dir and selected_dir != 'video': datetime.strptime(selected_dir, '%Y-%m-%d')
            except (ValueError, TypeError):
              self.send_error(400, "Invalid camera, date or pagination")
              return
            uploaded_image = data.get("uploaded_image")
            if uploaded_image:
              if ',' in uploaded_image: uploaded_image = uploaded_image.split(',')[1]
              uploaded_image = base64.b64decode(uploaded_image)

            if cam_name:
              camera_dirs = [BASE_DIR / "cameras" / cam_name]
            else:
              camera_dirs = [d for d in (BASE_DIR / "cameras").iterdir() if d.is_dir()]

            if selected_dir:
              selected_dirs = [selected_dir]
            else:
              selected_dirs = list({
                subdir.name 
                for camera_dir in camera_dirs
                if (camera_dir / "streams").is_dir()
                for subdir in (camera_dir / "streams").iterdir() 
                if subdir.is_dir()
              })
            if selected_dir is None and "video" not in selected_dirs: selected_dirs.append("video")

            if image_text and global_settings.use_clip: add_to_queue(object_finder._load_all_embeddings)
            if (uploaded_image or similar_img) and (global_settings.use_clip or global_settings.use_face): add_to_queue(object_finder._load_all_embeddings, is_face)
            
            if uploaded_image and (global_settings.use_clip or is_face):
              results = add_to_queue(run_clip, object_finder, uploaded_image, start+count, cam_name, selected_dir, is_face)
              self.send_results(results, start, count)
              return
            
            if similar_img and (global_settings.use_clip or is_face):
              results = add_to_queue(run_clip, object_finder, similar_img, start+count, cam_name, selected_dir, is_face) # todo one with above
              self.send_results(results, start, count)
              return

            if image_text and global_settings.use_clip:
              results = add_to_queue(run_search, object_finder, image_text, start+count, cam_name, selected_dir)
              self.send_results(results, start, count)
              return

            image_data = []
            for camera_dir in camera_dirs:
              for selected_dir in selected_dirs:
                event_image_path = camera_dir / "event_images" / selected_dir
                if not event_image_path.exists(): continue
                timeline = read_timeline(camera_dir / "streams" / selected_dir / "stream.m3u8")
                event_images = sorted(
                  event_image_path.glob("*.jpg"),
                  key=lambda p: p.stat().st_mtime,
                  reverse=True
                )
                for img in event_images:
                  if name_contains and name_contains not in img.name: continue
                  timing = event_timing(img, timeline)
                  ts = timing['playback_offset']
                  image_url = '/' + quote(str(img.relative_to(BASE_DIR)), safe='/')
                  image_data.append({
                    "url": image_url,
                    "timestamp": ts,
                    "filename": img.name,
                    "cam_name": camera_dir.name,
                    "folder": selected_dir,
                    "description": read_description(img),
                    "people": household.read_people(img),
                    **timing,
                  })

            image_data.sort(key=image_sort_key, reverse=True)
            if start is not None and count is not None: image_data = image_data[start:start+count]

            response_data = {
              "images": image_data,
              "count": len(image_data),
            }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode('utf-8'))
            return

def image_sort_key(item):
  if item.get("captured_at") is not None: return item['captured_at']
  try: return datetime.strptime(item["folder"], "%Y-%m-%d").timestamp() + item["timestamp"]
  except ValueError: return -1

def schedule_daily_restart(cam, restart_time):
    while True:
        now = datetime.now().time()
        target = time_obj(restart_time[0], restart_time[1])
        if now >= target:
          delta = (24 * 3600) - ((now.hour * 3600 + now.minute * 60 + now.second) - (target.hour * 3600 + target.minute * 60))
        else:
          delta = ((target.hour * 3600 + target.minute * 60) - 
            (now.hour * 3600 + now.minute * 60 + now.second))
        time.sleep(delta)
        cams = database.run_get("links", None)
        for cam_name in cams.keys():
          cam.start_time[cam_name] = None
          cam.hls_proc[cam_name], cam.proc[cam_name] = cam._open_ffmpeg(cam_name)
          cam.current_stream_dir_raw[cam_name] = cam._get_new_stream_dir(cam_name)



def get_lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def get_executable_args(): return ([sys.argv[0]], sys.argv[1:]) if getattr(sys, "frozen", False) else ([sys.executable, sys.argv[0]], sys.argv[1:])

def event_img_info(image): return {"ts": int(float(image.split('_')[0])), "object_id":int(image.split('_')[1]), "class_id":int(image.split('_')[2])}

def upload_to_r2(file_path: Path, signed_url: str, max_retries: int = 0) -> bool:
    try:
      url_parts = urllib.parse.urlparse(signed_url)
      if url_parts.scheme == 'https':
        conn = http.client.HTTPSConnection(url_parts.netloc)
      else:
        conn = http.client.HTTPConnection(url_parts.netloc)
      
      with file_path.open('rb') as f:
        file_size = file_path.stat().st_size
        headers = {'Content-Type': 'application/octet-stream', "Content-Length": str(file_size)}
        conn.request("PUT", url_parts.path + "?" + url_parts.query, body=f, headers=headers)
        response = conn.getresponse()
        if 200 <= response.status < 300: return True
        return False
    except Exception as e:
      print(f"Error uploading to R2: {e}")
      return False

import queue
task_queue = queue.Queue()
def add_to_queue(fn, *args):
  result_queue = queue.Queue(maxsize=1)
  task_queue.put((fn, args, result_queue))
  return result_queue.get()

def process_queue():
  try:
    fn, args, result_queue = task_queue.get_nowait()
  except queue.Empty: return
  result = fn(*args)
  result_queue.put(result)

def process_latest_face(img):
  if global_settings.use_face and str(object_queue[0]).endswith("_0.jpg"):
    face_img = object_finder.img_to_face(img)
    
    if face_img is not None:
      cv2.imwrite(str(object_queue[0]).replace("/objects/", "/faces/"), face_img)
      date = object_queue[0].parent.name
      pkl_path = object_queue[0].parent.parent.parent / "faces" / date /  "embeddings.pkl"
      face_emb = object_finder.adaface(Tensor(face_img).contiguous()).numpy()
      data = pickle.load(open(pkl_path, "rb")) if pkl_path.exists() else {}
      if "embeddings" not in data: data["embeddings"] = {}
      data["embeddings"][str(object_queue[0])] = face_emb
      pkl_path.parent.mkdir(parents=True, exist_ok=True)
      pickle.dump(data, open(pkl_path, "wb"))  

def set_settings(x): # todo, save to db, do logic in GlobalSettings class, sanitize inputs so yolo doesn't crash
  global global_settings
  global model
  global yolo_res
  global yolo_jit_cache
  if x.use_clip:
    object_finder.init_clip()
  else:
    object_finder.turn_off_clip()

  if x.use_face:
    object_finder.init_face()
  else:
    object_finder.turn_off_face()

  if x.model_size != global_settings.model_size or x.model_res != global_settings.model_res:
    yolo_jit_cache = {}
    model = make_detector(x.model_size, x.model_res)

  if x.key == None: # cloud notifications require a key; local AI does not
    x.userID = None

  local_descriptions.configure(x.use_qwen, x.qwen_size)
  global_settings = x

def clip_latest_img(img):
  if global_settings.use_clip:
    img = object_finder.preprocess(img)
    try:
      data = pickle.load(open(object_queue[0].parent / 'embeddings.pkl', 'rb')) if os.path.exists(object_queue[0].parent / 'embeddings.pkl') else {}
    except Exception: data = {}
    if "embeddings" not in data: data["embeddings"] = {}
    emb = jit_infer(object_finder.model.precompute_embedding, Tensor(img).unsqueeze(0), jit_cache).numpy()
    data["embeddings"][str(object_queue[0])] = emb
    with open(object_queue[0].parent / 'embeddings.pkl', "wb") as f: pickle.dump(data, f)
  
    if global_settings.userID:
      cam_name = object_queue[0].parts[object_queue[0].parts.index("cameras")+1:object_queue[0].parts.index("objects")][0]
      alerts = database.run_get("alerts", cam_name) # todo, get cam_name from file path!
      for k, v in alerts.items():
        if time.time() - v.last_det < 60 or not v.is_active(): continue
        if v.desc is None: continue
        if not hasattr(v, "desc_emb") or v.desc_emb is None:
          v.desc_emb = run_encode_text(object_finder, v.desc)
          database.run_put("alerts", cam_name, v, id=k)

        similarity = (v.desc_emb @ emb.T).item()
        print("sim =",similarity,v.desc,object_queue[0])
        if similarity > v.threshold:
          send_notif(global_settings.userID, f"Event Detected ({cam_name}: {v.desc})")
          alerts[k].last_det = time.time()
          database.run_put("alerts", cam_name, alerts[k], k)
          seen_time = event_img_info(str(object_queue[0]).split("/")[-1].split(".jpg")[0])["ts"]
          threading.Thread(target=export_and_upload, kwargs={"cam_name": cam_name, "thumbnail": object_queue[0], "userID": global_settings.userID, "key": global_settings.key, "start": seen_time, "length": 20, "wait": True}, daemon=True).start()
          break

cams = dict()
active_subprocesses = []
import socket
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    def __init__(self, server_address, RequestHandlerClass):
      ThreadingMixIn.__init__(self)
      HTTPServer.__init__(self, server_address, RequestHandlerClass)
      self.cleanup_stop_event = threading.Event()
      self.cleanup_thread = None
      max_gb = database.run_get("max_storage", None)
      if max_gb == {}:
        database.run_put("max_storage", "all", 1 if os.environ.get('CLEARCAM_NATIVE') == '1' else 256)
        max_gb = database.run_get("max_storage", None)
      self.max_gb = max_gb["all"]
      self.object_finder_stop_event = threading.Event()
      self.object_finder_thread = None
      self._setup_cleanup_and_clip_thread()

    def _setup_cleanup_and_clip_thread(self):
      if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
        self.cleanup_stop_event.clear()
        self.cleanup_thread = threading.Thread(target=self._cleanup_task, daemon=True, name="StorageCleanup")
        self.cleanup_thread.start()

    def _cleanup_task(self):
      while not self.cleanup_stop_event.is_set():
          try:
              self._check_and_cleanup_storage()
          except Exception as e:
              print(f"Cleanup error: {e}")
          self.cleanup_stop_event.wait(timeout=600)

    def _check_and_cleanup_storage(self):
      total_size = sum(f.stat().st_size for f in (BASE_DIR / "cameras").glob('**/*') if f.is_file())
      size_gb = total_size / (1000 ** 3)
      free_gb = shutil.disk_usage(BASE_DIR / "cameras").free / (1000 ** 3)
      if size_gb > self.max_gb or free_gb < 5: self._cleanup_oldest_files() # todo unhardcode
      else: self.storage_warning = None

    def _cleanup_oldest_files(self):
      candidates = expired_recording_dirs(BASE_DIR / "cameras", datetime.now().strftime("%Y-%m-%d"))
      if not candidates:
        self.storage_warning = "Storage limit reached. Today's recording is protected; increase the allowance or archive footage."
        return
      oldest_recording = candidates[0]
      camera_dir = oldest_recording.parent.parent
      shutil.rmtree(oldest_recording)
      for category in ("event_images", "objects", "faces"):
        relative = category + "/" + oldest_recording.name
        if (camera_dir / relative).is_symlink() or (camera_dir / category).is_symlink(): continue
        folder = contained_path(camera_dir, relative)
        if folder.is_dir() and not folder.is_symlink(): shutil.rmtree(folder)
      self.storage_warning = None
      print(f"Deleted completed recording day: {oldest_recording}")

    def server_close(self):
        if hasattr(self, 'cleanup_stop_event'):
            self.cleanup_stop_event.set()
        if hasattr(self, 'cleanup_thread') and self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        if hasattr(self, 'clip_stop_event'):
            self.object_finder_stop_event.set()
        if hasattr(self, 'clip_thread') and self.object_finder_thread:
            self.object_finder_thread.join(timeout=5)

        super().server_close()

class GlobalSettings:
  def __init__(self, use_clip=False, use_face=False ,model_size="t", model_res=960, userID=None, key=None, use_qwen=False, qwen_size=2):
    self.use_clip = use_clip
    self.use_face = use_face
    self.model_size = model_size
    self.model_res = model_res
    self.userID = userID
    self.key= key
    self.use_qwen = use_qwen
    self.qwen_size = qwen_size

def secret_settings(settings):
    return GlobalSettings(
        use_clip=settings.use_clip,
        use_face=settings.use_face,
        model_size=settings.model_size,
        model_res=settings.model_res,
        userID=settings.userID is not None,
        key=settings.key is not None,
        use_qwen=settings.use_qwen,
        qwen_size=settings.qwen_size
    )

if __name__ == "__main__":
  jit_cache = {}
  yolo_jit_cache = {}
  alerts_on = {}
  multiprocessing.set_start_method("spawn", force=True)
  database = db()
  cams = database.run_get("links", None)
  classes = {"0","1","2","7"} # person, bike, car, truck, bird (14)

  
  from models.objects import ObjectFinder

  object_queue = []
  cam_name = next((arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("--cam_name=")), "my_camera")
  
  class_labels = fetch('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names').read_text().split("\n")
  color_dict = {label: tuple((((i+1) * 50) % 256, ((i+1) * 100) % 256, ((i+1) * 150) % 256)) for i, label in enumerate(class_labels)}
  cam = None

  global_settings = database.run_get("global_settings", "all")
  if global_settings == {}: # todo, use None?
    global_settings = GlobalSettings(use_qwen=os.environ.get('CLEARCAM_NATIVE') == '1')
    database.run_put("global_settings", "all", global_settings)

  model = make_detector(global_settings.model_size, int(global_settings.model_res))
  object_finder = ObjectFinder()
  cam = VideoCapture()

  if global_settings.use_clip: object_finder.init_clip()
  if global_settings.use_face: object_finder.init_face()

  local_descriptions.configure(global_settings.use_qwen, global_settings.qwen_size)
  local_descriptions.retry_saved(BASE_DIR / "cameras")

  try:
    bind_host = os.environ.get("CLEARCAM_BIND_HOST", "127.0.0.1")
    server = ThreadedHTTPServer((bind_host, int(os.environ.get('CLEARCAM_PORT', '8080'))), RequestHandlerClass=HLSRequestHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Serving at http://{bind_host}:{server.server_port}")
    if ready_file := os.environ.get('CLEARCAM_READY_FILE'):
      ready_path = Path(ready_file)
      temporary = ready_path.with_suffix('.tmp')
      temporary.write_text(json.dumps({'port': server.server_port, 'pid': os.getpid()}))
      temporary.replace(ready_path)
  except OSError as e:
    if e.errno == socket.errno.EADDRINUSE:
      raise SystemExit('ClearCam port is already in use; no camera workers were started.')
    else:
        raise
    
  restart_time = (0, 0)
  threading.Thread(
    target=schedule_daily_restart,
    args=(cam, restart_time),
    daemon=True
  ).start()
  def stop_engine(signum, frame):
    raise KeyboardInterrupt
  signal.signal(signal.SIGTERM, stop_engine)
  try:
    cam.start()
  except KeyboardInterrupt:
    print("Stopping local camera engine")
  finally:
    cam.stopping.set()
    with cam.restart_lock:
      for name in set(cam.proc) | set(cam.hls_proc):
        cam._safe_kill_process(cam.proc.get(name))
        cam._safe_kill_process(cam.hls_proc.get(name))
    if server:
      server.shutdown()
      server.server_close()
