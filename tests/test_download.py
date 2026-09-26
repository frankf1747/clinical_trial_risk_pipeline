import io
import zipfile

import pytest

from ctrisk.ingest import download as dl


class FakeResponse:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.body


def zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.txt", "x")
    return buf.getvalue()


def test_saves_zip_and_leaves_no_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(dl.requests, "get", lambda *a, **k: FakeResponse(zip_bytes()))
    out = dl.download("https://example.org/a.zip", tmp_path / "q1" / "a.zip")
    assert zipfile.is_zipfile(out)
    assert not list(tmp_path.rglob("*.part"))


def test_rejects_non_zip(tmp_path, monkeypatch):
    monkeypatch.setattr(dl.requests, "get", lambda *a, **k: FakeResponse(b"<html>blocked</html>"))
    with pytest.raises(ValueError, match="not return a zip"):
        dl.download("https://example.org/a.zip", tmp_path / "a.zip")
    assert not list(tmp_path.iterdir())


def test_skips_existing_file(tmp_path, monkeypatch):
    target = tmp_path / "a.zip"
    target.write_bytes(b"already here")
    monkeypatch.setattr(dl.requests, "get", lambda *a, **k: pytest.fail("should not download"))
    assert dl.download("https://example.org/a.zip", target) == target
