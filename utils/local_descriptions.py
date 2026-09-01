"""Bounded local event descriptions; recording never waits for the model."""
import json
import queue
import threading
import logging
from pathlib import Path

PROMPT = "Describe only what is visibly happening in this camera image in one short sentence. Do not infer identity, intent, or events outside the image."


def frame_region(box, width, height):
    """Where a detection sits in the frame, in words a reader can picture."""
    x1, y1, x2, y2 = box
    across = ('left', 'centre', 'right')[min(2, int(((x1 + x2) / 2) / max(width, 1) * 3))]
    down = ('top', 'middle', 'bottom')[min(2, int(((y1 + y2) / 2) / max(height, 1) * 3))]
    if across == 'centre' and down == 'middle': return 'centre'
    if across == 'centre': return f'{down} centre'
    if down == 'middle': return across
    return f'{down} {across}'


def trigger_prompt(label, box, width, height):
    """Point the model at the detection that fired, not the prettiest thing in view.

    The event exists because one box crossed a threshold; a description that
    opens on scenery buries the reason the owner was alerted.
    """
    if not label: return PROMPT
    where = frame_region(box, width, height) if box is not None and len(box) == 4 else None
    located = f" in the {where} of the frame" if where else ""
    return (f"A {label} was detected{located} in this camera image. Write one short sentence that "
            f"starts with \"A {label}\" and says what it is visibly doing, then adds only essential "
            "surroundings. Do not infer identity or intent. Do not mention detection, boxes, or the frame.")

def write_trigger_crop(frame, box, event_path, margin=1.0, min_side=320):
    """Save the region around the triggering box (un-annotated) for description.

    A 2x crop carries the subject at far higher pixel density than the whole
    frame, and the model spends its vision budget on the thing that fired
    rather than on scenery. Returns None so callers fall back to the frame.
    """
    try:
        import cv2
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = map(float, box)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        hw = max((x2 - x1) * (1 + margin) / 2, min_side / 2)
        hh = max((y2 - y1) * (1 + margin) / 2, min_side / 2)
        a, b = int(max(0, cx - hw)), int(min(W, cx + hw))
        c, d = int(max(0, cy - hh)), int(min(H, cy + hh))
        if b - a < 64 or d - c < 64: return None
        target = Path(str(event_path)).with_suffix('.trigger.jpg')
        cv2.imwrite(str(target), frame[c:d, a:b], [cv2.IMWRITE_JPEG_QUALITY, 88])
        return target
    except Exception:
        return None


def default_model_factory():
    """MLX when its runtime is present (6x faster on Apple silicon), else tinygrad.

    CLEARCAM_DESCRIBER=tinygrad forces the old path for comparison.
    """
    import os
    if os.environ.get('CLEARCAM_DESCRIBER') != 'tinygrad':
        from utils import mlx_describer
        if mlx_describer.runtime_available():
            return mlx_describer.MlxProcess
    from utils.qwen_process import QwenProcess
    return QwenProcess


def read_description(path):
    try:
        return json.loads(Path(path).with_suffix('.description.json').read_text()).get('description')
    except (OSError, ValueError):
        return None

class LocalDescriptions:
    def __init__(self, model_factory=None):
        if model_factory is None:
            model_factory = default_model_factory()
        self.model_factory = model_factory
        self.enabled = False
        self.size = 2
        self.state = 'disabled'
        self.error = None
        self.failure = None
        self.jobs = queue.Queue(maxsize=32)
        self.last_backfill = 0.0
        self.model = None
        self.model_size = None
        threading.Thread(target=self._run, daemon=True, name='LocalQwen').start()

    def configure(self, enabled, size=2):
        if int(size) not in (2, 4, 8):
            raise ValueError('Description model size must be 2, 4, or 8')
        self.size, self.enabled = int(size), bool(enabled)
        self.state = 'waiting_for_event' if enabled else 'disabled'
        self.error = None
        self.failure = None

    def status(self):
        return dict(enabled=self.enabled, model=f'Qwen3-VL-{self.size}B', state=self.state, error=self.error, queued=self.jobs.qsize())

    def submit(self, image_path, camera_name, notify=False, prompt=None, image_override=None):
        if not self.enabled:
            return False
        try:
            self.jobs.put_nowait((Path(image_path), camera_name, (notify, prompt, image_override)))
            return True
        except queue.Full:
            return False

    def submit_summary(self, prompt, on_result):
        """Text-only generation through the same single-consumer queue.

        on_result(text or None) always runs — with the model's text when
        generation succeeds, or None so the caller can use its fallback.
        """
        if not self.enabled:
            on_result(None)
            return False
        try:
            self.jobs.put_nowait(('summary', prompt, on_result))
            return True
        except queue.Full:
            on_result(None)
            return False

    def backfill_if_idle(self, camera_root, every=300):
        """Describe anything that was dropped during a busy spell, once the queue is quiet."""
        import time
        if not self.enabled or not self.jobs.empty(): return 0
        if time.time() - self.last_backfill < every: return 0
        self.last_backfill = time.time()
        return self.retry_saved(camera_root)

    def retry_saved(self, camera_root):
        """Recover a bounded set of missing descriptions without replaying alerts."""
        if not self.enabled:
            return 0
        pending = sorted((p for p in Path(camera_root).glob('*/event_images/*/*.jpg') if not p.name.endswith('.trigger.jpg')),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        count = 0
        for path in pending:
            if read_description(path):
                continue
            if not self.submit(path, path.parents[2].name, notify=False):
                break
            count += 1
            if count == 8:
                break
        return count

    def _set_state(self, state):
        self.state = state

    def _run(self):
        while True:
            path, camera_name, notify = self.jobs.get()
            prompt, image_override = PROMPT, None
            if isinstance(notify, tuple):
                notify, override, image_override = (tuple(notify) + (None, None))[:3]
                prompt = override or PROMPT
            if path == 'summary':
                prompt, on_result = camera_name, notify
                try:
                    if self.model is None or self.model_size != self.size:
                        if self.model is not None and hasattr(self.model, 'close'):
                            self.model.close()
                        self.state = 'loading_model'
                        self.model = self.model_factory(size=f'{self.size}B', res=(448, 448))
                        self.model_size = self.size
                    self.state = 'describing'
                    text = self.model.generate(prompt=prompt, image_path=None, reset=True,
                                               max_tokens=220, quiet=True, on_state=self._set_state).strip()
                    self.state = 'ready'
                    on_result(text or None)
                except Exception:
                    logging.getLogger(__name__).exception('Local summary generation failed')
                    self.state = 'ready'
                    on_result(None)
                finally:
                    self.jobs.task_done()
                continue
            try:
                if not self.enabled or not path.exists():
                    continue
                if self.model is None or self.model_size != self.size:
                    if self.model is not None and hasattr(self.model, 'close'):
                        self.model.close()
                    self.state = 'loading_model'
                    self.model = self.model_factory(size=f'{self.size}B', res=(448, 448))
                    self.model_size = self.size
                self.state = 'describing'
                source = Path(image_override) if image_override and Path(image_override).exists() else path
                description = self.model.generate(prompt=prompt, image_path=source.resolve(), reset=True, max_tokens=96, quiet=True, on_state=self._set_state).strip()
                if not description:
                    raise ValueError('Model returned an empty description')
                if not self.enabled or not path.exists():
                    continue
                sidecar = path.with_suffix('.description.json')
                temporary = sidecar.with_suffix('.tmp')
                temporary.write_text(json.dumps(dict(description=description, model=f'Qwen3-VL-{self.model_size}B', generated=True)))
                temporary.replace(sidecar)
                if notify:
                    from utils.macos_notifications import send
                    send(f'AI description — {camera_name}', description)
                self.state, self.error, self.failure = 'ready', None, None
            except Exception as exc:
                logging.getLogger(__name__).exception('Local Qwen description failed')
                if self.model is not None and hasattr(self.model, 'close'):
                    self.model.close()
                self.model, self.model_size = None, None
                self.failure = exc
                self.state = 'error'
                self.error = f'Local description failed ({type(exc).__name__}). Recording is unaffected.'
            finally:
                self.jobs.task_done()
