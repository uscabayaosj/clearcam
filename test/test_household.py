import tempfile
import unittest
from pathlib import Path

from utils.household import HouseholdStore, clean_name, read_people, write_people


class HouseholdTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = HouseholdStore(self.temp.name)

    def test_enroll_match_and_threshold(self):
        member = self.store.add_sample('Ulysses', [1.0, 0.0, 0.0])
        self.assertTrue(self.store.has_members())
        hit = self.store.match([0.99, 0.05, 0.0])
        self.assertEqual(hit['name'], 'Ulysses')
        self.assertGreater(hit['score'], 0.9)
        self.assertIsNone(self.store.match([0.0, 1.0, 0.0]))
        self.assertIsNone(self.store.match([0.0, 0.0, 0.0]))
        self.assertEqual(self.store.list_members(), [dict(id=member, name='Ulysses', samples=1)])

    def test_same_name_merges_and_samples_are_capped(self):
        first = self.store.add_sample('Ana', [1.0, 0.0])
        for index in range(12):
            merged = self.store.add_sample('ana', [0.0, 1.0 + index])
            self.assertEqual(merged, first)
        members = self.store.list_members()
        self.assertEqual(len(members), 1)
        self.assertLessEqual(members[0]['samples'], 8)

    def test_best_of_multiple_members_wins(self):
        self.store.add_sample('Ana', [1.0, 0.0])
        self.store.add_sample('Ben', [0.7, 0.7])
        self.assertEqual(self.store.match([0.72, 0.69])['name'], 'Ben')

    def test_remove_validates_id_and_deletes(self):
        member = self.store.add_sample('Ana', [1.0])
        with self.assertRaises(ValueError): self.store.remove('../escape')
        self.assertFalse(self.store.remove('0' * 32))
        self.assertTrue(self.store.remove(member))
        self.assertFalse(self.store.has_members())

    def test_clean_name_rejects_junk(self):
        self.assertEqual(clean_name('  Ana   Lee '), 'Ana Lee')
        with self.assertRaises(ValueError): clean_name('   ')
        with self.assertRaises(ValueError): clean_name('x' * 61)

    def test_people_sidecar_roundtrip_and_corruption(self):
        image = Path(self.temp.name) / 'event_notif.jpg'
        image.touch()
        self.assertIsNone(read_people(image))
        write_people(image, ['Ana'])
        self.assertEqual(read_people(image), ['Ana'])
        write_people(image, [])
        self.assertEqual(read_people(image), [])
        image.with_suffix('.people.json').write_text('not json')
        self.assertIsNone(read_people(image))

    def test_corrupt_member_file_is_ignored(self):
        self.store.add_sample('Ana', [1.0, 0.0])
        (self.store.root / 'broken.json').write_text('{"nope": true}')
        (self.store.root / 'invalid.json').write_text('garbage')
        self.assertEqual(len(self.store.list_members()), 1)
        self.assertEqual(self.store.match([1.0, 0.0])['name'], 'Ana')


if __name__ == '__main__':
    unittest.main()
