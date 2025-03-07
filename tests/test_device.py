from toyllm.device import get_device


def test_device() -> None:
    device = get_device()
    assert device.type in ("cuda", "cpu")
