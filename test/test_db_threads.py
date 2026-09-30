"""The engine shares one SQLite connection across HTTP handler threads, camera
loops and counters. Concurrent use of a single connection corrupts its state
("returned NULL without setting an exception"); access must be serialised."""
import os
import tempfile
import threading
import unittest


class ConcurrentDbTest(unittest.TestCase):
  def test_many_threads_reading_and_writing_never_error(self):
    with tempfile.TemporaryDirectory() as tmp:
      from utils import db as dbmod
      old_path, old_conn, old_tables = dbmod.CACHEDB, dbmod._db_connection, set(dbmod._db_tables)
      dbmod.CACHEDB, dbmod._db_connection = os.path.join(tmp, 'test.db'), None
      dbmod._db_tables.clear()
      try:
        store = dbmod.db()
        store.run_put('links', 'cam0', 'x')
        errors = []

        def worker(n):
          try:
            for i in range(1500):
              store.run_put('analysis_prog', f'cam{n}', {'Tracking': i})
              store.run_get('analysis_prog', None)
              store.run_get('links', None)
          except Exception as e:   # noqa: BLE001 - any failure is the bug
            errors.append(repr(e))

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(16)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(errors[:3], [])
        self.assertEqual(len(store.run_get("analysis_prog", None)), 16)
      finally:
        if dbmod._db_connection is not None: dbmod._db_connection.close()
        dbmod.CACHEDB, dbmod._db_connection = old_path, old_conn
        dbmod._db_tables.clear(); dbmod._db_tables.update(old_tables)


if __name__ == '__main__':
  unittest.main()
