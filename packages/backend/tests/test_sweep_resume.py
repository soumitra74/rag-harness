import pytest

from app.config import Settings
import app.services.sweep as sweep


class FakeDatabase:
    def __init__(self, corpus_snapshot):
        self._corpus_snapshot = corpus_snapshot
        self.status_updates = []

    async def get_sweep(self, sweep_id):
        return {"id": sweep_id, "corpus_snapshot": self._corpus_snapshot}

    async def update_sweep_status(self, sweep_id, status, error_message=None):
        self.status_updates.append((sweep_id, status, error_message))


def _settings(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "doc1.pdf").write_bytes(b"x")
    return Settings(corpus_dir=str(corpus_dir))


@pytest.mark.asyncio
async def test_resume_sweep_resumes_loop_when_corpus_unchanged(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    resumed = []

    async def fake_run_sweep_loop(db, provider, chat_provider, judge, ner, settings_arg, sweep_id):
        resumed.append(sweep_id)

    monkeypatch.setattr(sweep, "run_sweep_loop", fake_run_sweep_loop)
    db = FakeDatabase(corpus_snapshot=["doc1.pdf"])

    await sweep.resume_sweep(db, object(), object(), object(), object(), settings, sweep_id=1)

    assert resumed == [1]
    assert db.status_updates == [(1, "running", None)]


@pytest.mark.asyncio
async def test_resume_sweep_raises_when_corpus_changed(tmp_path, monkeypatch):
    settings = _settings(tmp_path)

    async def fake_run_sweep_loop(*args, **kwargs):
        raise AssertionError("must not resume the loop when the corpus changed")

    monkeypatch.setattr(sweep, "run_sweep_loop", fake_run_sweep_loop)
    db = FakeDatabase(corpus_snapshot=["doc1.pdf", "doc2.pdf"])

    with pytest.raises(sweep.CorpusChangedError):
        await sweep.resume_sweep(db, object(), object(), object(), object(), settings, sweep_id=1)

    assert db.status_updates == []
