"""The conversation simulation and the one-command runner, with a scripted model."""
import config
import tools
from evals import conversation_eval, simulate
from tests.test_brain import call, say, script


def test_conversation_turn_checks():
    turn = conversation_eval.t("What's 15 times 4?", tool="calculate", contains="60")
    assert conversation_eval.check_turn(turn, "That's 60, sir.", ["calculate"]) == []
    problems = conversation_eval.check_turn(turn, "I'm here, sir.", [])
    assert "expected calculate, used no tool" in problems and "style: no_presence_filler" in problems
    assert conversation_eval.check_turn(conversation_eval.t("Goats."), "", []) [0] == "empty reply"


def test_simulation_fakes_side_effects_and_restores_everything(monkeypatch):
    real_run_tool, real_db = tools.run_tool, config.DB_PATH
    monkeypatch.setattr(conversation_eval, "SCENARIOS", [
        ("volume", [conversation_eval.t("It's way too loud", tool="pc_control")]),
        ("approval: yes", [conversation_eval.t("Copy hello to my clipboard", tool="clipboard"),
                           conversation_eval.t("yes", starts="Done")]),
    ])
    script(monkeypatch,
           call("pc_control", action="volume_down"), say("Turned it down, sir."),
           call("clipboard", action="write", text="hello"), say("Shall I copy that, sir?"))
    out = conversation_eval.run(show=lambda *_: None)
    assert out["turns_passed"] == 3, out["rows"]
    assert out["rows"][0]["tools"] == ["pc_control"]
    assert out["rows"][2]["reply"].startswith("Done")  # approved in code, faked, nothing copied
    assert tools.run_tool is real_run_tool and config.DB_PATH == real_db


def test_simulate_quick_writes_a_summary(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(conversation_eval, "SCENARIOS", [("hi", [conversation_eval.t("Long day")])])
    script(monkeypatch, say("Then sit down, sir. The day is over."))
    monkeypatch.setattr("sys.argv", ["simulate.py", "--quick"])
    assert simulate.main() == 0
    saved = list((tmp_path / "evals").glob("simulation-*.txt"))
    assert saved and "Conversations: 1/1 turns" in saved[0].read_text()


def test_numbers_said_as_words_count():
    turn = conversation_eval.t("What's 15 times 4?", contains=("60", "sixty"))
    assert conversation_eval.check_turn(turn, "Sixty, sir.", []) == []
