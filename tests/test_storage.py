"""The database is shared by the listener, the reminder checker and the web API at once."""
import threading

import config
import memory


def test_database_uses_wal(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    memory.init_db()
    db = memory._connect()
    assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    db.close()


def test_writers_at_once_do_not_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    memory.init_db()
    errors = []

    def write(n):
        try:
            for i in range(40):
                memory.add_message(f"s{n}", "user", f"message {i}")
        except Exception as e:  # "database is locked" would land here
            errors.append(e)

    threads = [threading.Thread(target=write, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
