"""In-memory event journal for live-only mode (record_video=False).

Nothing here touches disk. Events (with their JPEG bytes already encoded)
live in a bounded deque for as long as they fit; older ones are simply
dropped. This is the live-only analogue of the event_images tree.
"""
import collections
import threading
import time


class LiveJournal:
    def __init__(self, maxlen=300):
        self._lock = threading.Lock()
        self._events = collections.deque(maxlen=maxlen)
        self._by_id = {}

    def add(self, event):
        """event is a dict; must contain 'id'. Returns the id."""
        event_id = str(event['id'])
        event['id'] = event_id
        with self._lock:
            self._events.appendleft(event)
            self._by_id[event_id] = event
            # Deque eviction on the left side happens automatically via maxlen,
            # but we must also drop the corresponding entry from the index.
            self._prune_index_locked()
        return event_id

    def _prune_index_locked(self):
        live_ids = {e['id'] for e in self._events}
        for stale in [eid for eid in self._by_id if eid not in live_ids]:
            del self._by_id[stale]

    def get(self, event_id):
        with self._lock:
            return self._by_id.get(str(event_id))

    def update(self, event_id, **fields):
        with self._lock:
            event = self._by_id.get(str(event_id))
            if event is None: return None
            event.update(fields)
            return event

    def list(self, start=0, count=100, cam=None, alerts_only=False):
        with self._lock:
            events = list(self._events)  # already newest-first
        if cam:
            events = [e for e in events if e.get('cam_name') == cam]
        if alerts_only:
            events = [e for e in events if e.get('is_notif')]
        return events[start:start + count]

    def all_since(self, ts):
        with self._lock:
            events = list(self._events)
        return [e for e in events if e.get('captured_at', 0) >= ts]
