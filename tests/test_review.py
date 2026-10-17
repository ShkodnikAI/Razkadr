"""Text review toolkit tests (naryad 020): fact extraction, multiset diff,
ingredient candidates, CLI text-review.

All checks are deterministic - no providers, no network. The tests pin the
gateway D invariant (numbers, times, temperatures survive from original to
output) as the tool's fail verdict, the "candidates are marked, not failed"
policy for words, and the n/a handling for episodes without a source text.
"""

import json

from razkadr.cli import EXIT_FILE_ERROR, EXIT_INVALID, EXIT_OK, main
from razkadr.review import (
    episode_verdict,
    extract_measures,
    extract_numbers,
    extract_temperatures,
    extract_times,
    fact_diff,
    ingredient_candidates,
    review_file,
    review_storyboard,
)


def storyboard_with_text(episode_id: str, mode: str, original: str, output: str, role: str = "steps") -> dict:
    return {
        "schema_version": "1.0",
        "project": {"id": "p", "title": "t", "preset": "recipe", "output_lang": "ru"},
        "sources": [{"id": "v1", "path": "video.mp4"}],
        "episodes": [
            {
                "id": episode_id,
                "role": role,
                "source_id": "v1",
                "t_start": 0.0,
                "t_end": 10.0,
                "frame": {"path": "frames/f.jpg", "t": 5.0},
                "text": {"mode": mode, "source_lang": "ru", "original": original, "output": output, "output_lang": "ru"},
            }
        ],
    }


# --- fact extraction ------------------------------------------------------------------


def test_extract_numbers_multiset_with_decimal_normalization():
    assert extract_numbers("2 яйца и 0,5 стакана, потом ещё 2") == {"2": 2, "0.5": 1}
    assert extract_numbers("без чисел") == {}


def test_extract_temperatures_scales_and_bare_gradus():
    assert extract_temperatures("при 350 °F и 180 °C") == {"350 °F": 1, "180 °C": 1}
    assert extract_temperatures("разогрейте до 200 градусов") == {"200 градус": 1}
    assert extract_temperatures("без температуры") == {}


def test_extract_times_normalizes_units():
    assert extract_times("15 минут, потом 1,5 часа и 30 сек") == {
        "15 min": 1,
        "1.5 h": 1,
        "30 sec": 1,
    }
    # word forms: "минуты" inside "в течение 3 минуты" and plural "2 часами"
    assert extract_times("3 минуты и 2 часами позже") == {"3 min": 1, "2 h": 1}


def test_extract_times_no_false_positive_inside_words():
    # "минус" is not "мин", "лимон" is not "л"
    assert extract_times("остынет за 5 минут - минус 10") == {"5 min": 1}
    assert extract_measures("сок 2 лимонов и 3 лимона") == {}


def test_extract_measures_mass_volume_units():
    assert extract_measures("200 г муки, 2 ст. л. масла и 0,5 л молока") == {
        "200 g": 1,
        "2 tbsp": 1,
        "0.5 l": 1,
    }
    assert extract_measures("1 cup rice and 2 cups flour") == {"1 cup": 1, "2 cup": 1}
    # "г" inside "градусов" is not a measure; "л" inside "лимонов" is not either
    assert extract_measures("до 200 градусов") == {}


# --- diff and verdicts ----------------------------------------------------------------


def test_fact_diff_multiset_counts():
    diff = fact_diff("2 яйца, 200 г муки, 15 минут", "яйца, мука, 15 минут")
    assert diff["numbers"]["lost"] == {"2": 1, "200": 1}
    assert diff["measures"]["lost"] == {"200 g": 1}
    assert diff["times"]["lost"] == {}
    assert not diff["temperatures"]["lost"]


def test_fact_diff_added_facts():
    diff = fact_diff("жарьте блины", "жарьте 5 минут")
    assert diff["times"]["added"] == {"5 min": 1}
    assert diff["numbers"]["added"] == {"5": 1}


def test_verdict_fail_on_lost_numbers():
    facts = fact_diff("Варите 200 г риса 15 минут", "Варите рис 15 минут")
    assert episode_verdict(facts, []) == "fail"


def test_verdict_review_on_candidates_or_added():
    facts = fact_diff("посыпьте зеленью", "посыпьте укропом")
    assert facts["numbers"]["lost"] == {}
    assert episode_verdict(facts, ["зеленью"]) == "review"
    facts = fact_diff("жарьте блины", "жарьте 5 минут")
    assert episode_verdict(facts, []) == "review"


def test_verdict_ok_when_everything_matches():
    facts = fact_diff("Варите 200 г риса 15 минут при 180 °C", "Рис 200 г: 15 минут, 180 °C")
    assert episode_verdict(facts, []) == "ok"


def test_ingredient_candidates_marks_missing_words():
    candidates = ingredient_candidates(
        "Для теста: мука, кефир, сода и соль",
        "Мука с содой",
    )
    # "сода" matches "содой" through the stem; "теста", "кефир", "соль" are lost
    assert "теста" in candidates and "кефир" in candidates and "соль" in candidates
    assert "сода" not in candidates
    # stop words and unit words never appear as candidates
    assert "для" not in candidates


def test_ingredient_candidates_empty_when_all_present():
    assert ingredient_candidates("мука и соль", "мука, соль") == []


# --- storyboard review ----------------------------------------------------------------


def test_review_storyboard_ok_and_na():
    storyboard = {
        "schema_version": "1.0",
        "project": {"id": "p", "title": "t", "preset": "recipe", "output_lang": "ru"},
        "sources": [{"id": "v1", "path": "video.mp4"}],
        "episodes": [
            {
                "id": "ep-ok", "role": "steps", "source_id": "v1", "t_start": 0, "t_end": 10,
                "frame": {"path": "f.jpg", "t": 5},
                "text": {"mode": "condensed", "source_lang": "ru",
                         "original": "Варите 200 г риса 15 минут", "output": "Рис 200 г, 15 минут",
                         "output_lang": "ru"},
            },
            {
                "id": "ep-desc", "role": "hero", "source_id": "v1", "t_start": 20, "t_end": 30,
                "frame": {"path": "f.jpg", "t": 25},
                "text": {"mode": "described", "source_lang": "ru", "original": "",
                         "output": "готовое блюдо", "output_lang": "ru"},
            },
            {
                "id": "ep-none", "role": "serving", "source_id": "v1", "t_start": 40, "t_end": 50,
                "frame": {"path": "f.jpg", "t": 45},
            },
        ],
    }
    report = review_storyboard(storyboard)
    assert report["summary"] == {"episodes": 3, "reviewable": 1, "ok": 1, "review": 0, "fail": 0}
    by_id = {e["id"]: e for e in report["episodes"]}
    assert by_id["ep-ok"]["status"] == "ok"
    assert by_id["ep-desc"]["status"] == "n/a"
    assert by_id["ep-none"]["status"] == "n/a"


def test_review_storyboard_fail_records_lost_facts():
    storyboard = storyboard_with_text(
        "ep-steps", "condensed",
        "Обжарьте 200 г лука 10 минут при 180 °C",
        "Обжарьте лук",
    )
    report = review_storyboard(storyboard)
    entry = report["episodes"][0]
    assert entry["status"] == "fail"
    assert entry["facts"]["numbers"]["lost"] == {"200": 1, "10": 1, "180": 1}
    assert entry["facts"]["times"]["lost"] == {"10 min": 1}
    assert entry["facts"]["temperatures"]["lost"] == {"180 °C": 1}


def test_review_storyboard_translation_mode_counts_output_only():
    # EN output compared against RU original: the facts that must survive are
    # the numbers; words differ by language - no ingredient candidates across
    # languages (they would be noise), verdict comes from facts
    storyboard = storyboard_with_text(
        "ep-steps", "condensed",
        "Выливаем 200 г теста, жарим 1,5 минуты",
        "Pour 200 g of batter, fry 1.5 minutes",
    )
    report = review_storyboard(storyboard)
    entry = report["episodes"][0]
    assert entry["facts"]["numbers"]["lost"] == {}
    assert entry["facts"]["measures"]["lost"] == {}
    assert entry["facts"]["times"]["lost"] == {}
    # Cyrillic words of the original are absent from the EN output - they are
    # candidates, and the episode is flagged for human review, not failed
    assert entry["ingredient_candidates"]
    assert entry["status"] == "review"


# --- CLI ------------------------------------------------------------------------------


def test_cli_text_review_ok_exit_0(tmp_path, capsys):
    path = tmp_path / "sb.json"
    path.write_text(json.dumps(storyboard_with_text(
        "ep-steps", "condensed", "Варите 200 г риса 15 минут", "Рис 200 г, 15 минут",
    ), ensure_ascii=False), encoding="utf-8")
    out_path = tmp_path / "review.json"
    assert main(["text-review", str(path), "--out", str(out_path)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "ok 1" in out and "1/1 episodes reviewable" in out
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["summary"]["ok"] == 1


def test_cli_text_review_fail_exit_1(tmp_path, capsys):
    path = tmp_path / "sb.json"
    path.write_text(json.dumps(storyboard_with_text(
        "ep-steps", "condensed", "Варите 200 г риса 15 минут", "Варите рис",
    ), ensure_ascii=False), encoding="utf-8")
    assert main(["text-review", str(path)]) == EXIT_INVALID
    assert "lost" in capsys.readouterr().out


def test_cli_text_review_missing_file_exit_2(tmp_path, capsys):
    assert main(["text-review", str(tmp_path / "nope.json")]) == EXIT_FILE_ERROR
    assert "file not found" in capsys.readouterr().err


def test_cli_text_review_not_a_storyboard_exit_2(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"something": "else"}), encoding="utf-8")
    assert main(["text-review", str(path)]) == EXIT_FILE_ERROR
    assert "not a Storyboard" in capsys.readouterr().err


def test_review_file_on_examples_storyboard():
    """The shipped valid example reviews cleanly: e2 (condensed) keeps its facts."""
    report, _ = review_file("examples/storyboard.valid.json")
    by_id = {e["id"]: e for e in report["episodes"]}
    assert by_id["e1"]["status"] == "n/a"  # described, no original
    assert by_id["e2"]["status"] in ("ok", "review")
    assert by_id["e2"]["facts"]["numbers"]["lost"] == {}
