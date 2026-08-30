"""Bounded local event descriptions; recording never waits for the model."""
import json
import queue
import threading
import logging
from pathlib import Path

PROMPT = "Describe only what is visibly happening in this camera image in one short sentence. Do not infer identity, intent, or events outside the image."

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

    def submit(self, image_path, camera_name, notify=False):
        if not self.enabled:
            return False
        try:
            self.jobs.put_nowait((Path(image_path), camera_name, notify))
            return True
        except queue.Full:
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
                description = self.model.generate(prompt=PROMPT, image_path=path.resolve(), reset=True, max_tokens=96, quiet=True, on_state=self._set_state).strip()
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
