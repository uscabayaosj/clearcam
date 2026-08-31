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
    where = frame_region(box, width, height) if box else None
    located = f" in the {where} of the frame" if where else ""
    return (f"A {label} was detected{located} and is outlined by a box in this camera image. "
            f"Begin your sentence with that {label} and what it is visibly doing, then add only "
            "essential surroundings. One short sentence. Do not infer identity or intent, and do "
            "not mention the box.")

def read_description(path):
    try:
        return json.loads(Path(path).with_suffix('.description.json').read_text()).get('description')
    except (OSError, ValueError):
        return None

class LocalDescriptions:
    def __init__(self, model_factory=None):
        if model_factory is None:
            from utils.qwen_process import QwenProcess
            model_factory = QwenProcess
        self.model_factory = model_factory
        self.enabled = False
        self.size = 2
        self.state = 'disabled'
        self.error = None
        self.failure = None
        self.jobs = queue.Queue(maxsize=8)
        self.model = None
        self.model_size = None
        threading.Thread(target=self._run, daemon=True, name='LocalQwen').start()

    def configure(self, enabled, size=2):
        if int(size) not in (2, 4):
            raise ValueError('Qwen model size must be 2 or 4')
        self.size, self.enabled = int(size), bool(enabled)
        self.state = 'waiting_for_event' if enabled else 'disabled'
        self.error = None
        self.failure = None

    def status(self):
        return dict(enabled=self.enabled, model=f'Qwen3-VL-{self.size}B', state=self.state, error=self.error, queued=self.jobs.qsize())

    def submit(self, image_path, camera_name, notify=False, prompt=None):
        if not self.enabled:
            return False
        try:
            self.jobs.put_nowait((Path(image_path), camera_name, (notify, prompt)))
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

    def retry_saved(self, camera_root):
        """Recover a bounded set of missing descriptions without replaying alerts."""
        if not self.enabled:
            return 0
        pending = sorted(Path(camera_root).glob('*/event_images/*/*.jpg'),
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
            prompt = PROMPT
            if isinstance(notify, tuple): notify, override = notify; prompt = override or PROMPT
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
                description = self.model.generate(prompt=prompt, image_path=path.resolve(), reset=True, max_tokens=96, quiet=True, on_state=self._set_state).strip()
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
