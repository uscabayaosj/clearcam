"""Named household recognition on top of the local face-embedding models.

Everything stays on this Mac: enrolled members are embedding files in the
data directory, and matches are written as sidecar JSON next to each event.
"""
import json
import math
from pathlib import Path
import re
import time
import uuid

MATCH_THRESHOLD = 0.35  # cosine similarity floor for AdaFace ir50 embeddings
MAX_SAMPLES_PER_MEMBER = 8


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0: return 0.0
    return dot / (norm_a * norm_b)


def clean_name(name):
    name = re.sub(r'\s+', ' ', str(name or '')).strip()
    if not name or len(name) > 60: raise ValueError('Name must be 1-60 characters')
    return name


class HouseholdStore:
    def __init__(self, root):
        self.root = Path(root) / 'household'

    def _member_files(self):
        if not self.root.is_dir(): return []
        return sorted(self.root.glob('*.json'))

    def _read(self, path):
        try:
            data = json.loads(path.read_text())
            if not isinstance(data.get('name'), str) or not isinstance(data.get('embeddings'), list):
                return None
            return data
        except (OSError, ValueError):
            return None

    def list_members(self):
        members = []
        for path in self._member_files():
            data = self._read(path)
            if data:
                members.append(dict(id=path.stem, name=data['name'], samples=len(data['embeddings'])))
        return members

    def has_members(self):
        return any(self._read(path) for path in self._member_files())

    def add_sample(self, name, embedding):
        """Add one face sample; samples for an existing name merge into that member."""
        name = clean_name(name)
        embedding = [float(v) for v in embedding]
        for path in self._member_files():
            data = self._read(path)
            if data and data['name'].casefold() == name.casefold():
                data['embeddings'] = (data['embeddings'] + [embedding])[-MAX_SAMPLES_PER_MEMBER:]
                self._write(path, data)
                return path.stem
        member_id = uuid.uuid4().hex
        self.root.mkdir(parents=True, exist_ok=True)
        self._write(self.root / f'{member_id}.json', dict(name=name, embeddings=[embedding], created=time.time()))
        return member_id

    def _write(self, path, data):
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data))
        temporary.replace(path)

    def remove(self, member_id):
        if not re.fullmatch(r'[0-9a-f]{32}', str(member_id)): raise ValueError('Invalid member id')
        target = self.root / f'{member_id}.json'
        if not target.is_file(): return False
        target.unlink()
        return True

    def match(self, embedding, threshold=MATCH_THRESHOLD):
        """Best enrolled name for an embedding, or None below the threshold."""
        embedding = [float(v) for v in embedding]
        best_name, best_score = None, threshold
        for path in self._member_files():
            data = self._read(path)
            if not data: continue
            for sample in data['embeddings']:
                score = _cosine(embedding, sample)
                if score >= best_score:
                    best_name, best_score = data['name'], score
        return None if best_name is None else dict(name=best_name, score=round(best_score, 3))


def people_path(image_path):
    return Path(image_path).with_suffix('.people.json')


def read_people(image_path):
    try:
        data = json.loads(people_path(image_path).read_text())
        names = data.get('people')
        return names if isinstance(names, list) else None
    except (OSError, ValueError):
        return None


def write_people(image_path, names):
    target = people_path(image_path)
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(people=list(names), recognized_at=time.time())))
    temporary.replace(target)
