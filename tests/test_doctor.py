"""Offline tests for doctor.py: no Ollama, GPU or network needed."""
import config
import doctor


def test_model_missing_is_a_fail_with_pull_fix():
    c = doctor.check_model(["llama3:latest"])
    assert c.status == "fail" and f"ollama pull {config.MODEL}" in c.fix


def test_model_present_is_ok():
    assert doctor.check_model([config.MODEL]).status == "ok"


def test_ollama_down():
    assert doctor.check_ollama(None).status == "fail"
    assert doctor.check_model(None).status == "fail"


def _fake_model(folder):
    m = folder / "manifests" / "registry.ollama.ai" / "library" / "qwen3"
    m.mkdir(parents=True)
    (m / "8b").write_text("{}")


def test_models_left_in_old_folder_is_caught(tmp_path, monkeypatch):
    """The exact problem from 15 Nov 2026: OLLAMA_MODELS set, models still on C:."""
    home, new = tmp_path / "home", tmp_path / "D_models"
    _fake_model(home / ".ollama" / "models")
    new.mkdir()
    monkeypatch.setattr(doctor.Path, "home", lambda: home)
    monkeypatch.setenv("OLLAMA_MODELS", str(new))
    c = doctor.check_model_folder()
    assert c.status == "fail" and "robocopy" in c.fix


def test_models_in_custom_folder_ok(tmp_path, monkeypatch):
    _fake_model(tmp_path)
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path))
    assert doctor.check_model_folder().status == "ok"


def test_summary_takes_the_worst_status():
    checks = [doctor.Check("a", "ok", ""), doctor.Check("b", "warn", ""), doctor.Check("c", "fail", "")]
    assert doctor.summary(checks)["status"] == "fail"
    assert doctor.summary(checks[:2])["status"] == "warn"
