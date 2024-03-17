from toyllm.device import get_device


def test_device():
    device = get_device()
    assert device.type in ("cuda", "cpu")
