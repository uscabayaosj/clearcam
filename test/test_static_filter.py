from utils import static_filter as sf


def frames_with(post_every=True, n=10):
    frames = []
    for i in range(n):
        boxes = [('car', (0.1, 0.1, 0.3, 0.3))]                      # parked car, every frame
        if post_every: boxes.append(('person', (0.5, 0.5, 0.55, 0.7)))  # a post seen as a person
        if i == 3: boxes.append(('person', (0.2 + i * 0.01, 0.6, 0.25, 0.8)))  # one real passer-by
        frames.append(('cam', boxes))
    return frames


def test_stationary_person_dropped_parked_car_and_passer_by_kept():
    frames = frames_with()
    mask = sf.static_mask(frames)
    assert all(m[0] for m in mask)            # cars stay
    assert not any(m[1] for m in mask)        # the post is dropped everywhere
    assert mask[3][2] is True                 # the passer-by stays


def test_cameras_are_judged_separately():
    frames = frames_with(post_every=False) + [('other', [('person', (0.5, 0.5, 0.55, 0.7))])] * 2
    assert all(all(m) for m in sf.static_mask(frames))


def test_camera_of_parses_collected_names():
    assert sf.camera_of('own-test-camera-20260928-010203.jpg') == 'test-camera'
    assert sf.camera_of('plain.jpg') == 'plain.jpg'


def test_ignore_areas_parse_and_match():
    import pytest
    areas = sf.parse_ignore(['tapoc113=0.86,0.65,1,1:car'])
    bins = (0.88, 0.74, 0.98, 0.90)
    assert sf.ignored('tapoc113', 'car', bins, areas)
    assert not sf.ignored('tapoc113', 'person', bins, areas)     # class-restricted
    assert not sf.ignored('tapo360', 'car', bins, areas)          # per camera
    assert not sf.ignored('tapoc113', 'car', (0.1, 0.1, 0.3, 0.3), areas)
    with pytest.raises(ValueError):
        sf.parse_ignore(['tapoc113=0.9,0.5,0.8,1'])
