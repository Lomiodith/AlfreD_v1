import os
import sys

import pytest

import file_opener


@pytest.fixture
def files(tmp_path, monkeypatch):
    for name in ["alfred_tray.py", "README.md", "notes/readme.txt", "budget 2026.xlsx"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("text")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "budget.js").write_text("skipped")
    monkeypatch.setattr(file_opener, "FILE_SEARCH_DIRS", [str(tmp_path)])
    opened = []
    monkeypatch.setattr(
        file_opener, "open_path", lambda p: opened.append(p) or {"opened": p}
    )
    return tmp_path, opened


@pytest.mark.parametrize(
    "spoken, expected",
    [
        ("alfred tray", "alfred_tray.py"),
        ("Alfred_Tray.py", "alfred_tray.py"),
        ("budget", "budget 2026.xlsx"),
    ],
)
def test_spoken_name_opens_the_single_match(files, spoken, expected):
    root, opened = files

    assert file_opener.open_file(spoken) == {"opened": str(root / expected)}


def test_exact_name_beats_partial_matches(files):
    root, opened = files

    file_opener.open_file("README.md")

    assert opened == [str(root / "README.md")]


def test_several_equally_good_matches_are_returned_to_ask(files):
    root, opened = files

    result = file_opener.open_file("readme")

    assert opened == []
    assert set(result["several_matches"]) == {
        str(root / "README.md"),
        str(root / "notes" / "readme.txt"),
    }


def test_full_path_opens_directly_and_unknown_names_report_an_error(files):
    root, opened = files

    file_opener.open_file(f'"{root / "README.md"}"')

    assert opened == [str(root / "README.md")]
    assert "error" in file_opener.open_file("nothing like this")


def test_a_script_whose_default_action_runs_it_goes_to_the_editor(
    tmp_path, monkeypatch
):
    script, program = tmp_path / "setup.bat", tmp_path / "tool.exe"
    script.write_text("echo hi")
    program.write_bytes(b"MZ\x00\x00")
    launched = []
    monkeypatch.setattr(file_opener, "_would_run", lambda p: True)
    monkeypatch.setattr(file_opener.shutil, "which", lambda name: "code")
    monkeypatch.setattr(file_opener, "_launch", launched.append)

    assert "VS Code" in file_opener.open_path(str(script))["with"]
    assert launched == [["code", str(script)]]
    assert "error" in file_opener.open_path(str(program))


@pytest.mark.skipif(sys.platform != "win32", reason="reads the Windows registry")
def test_windows_association_decides_what_would_run(tmp_path):
    folder = tmp_path / "folder"
    folder.mkdir()

    assert file_opener._would_run(str(tmp_path / "x.bat"))
    assert not file_opener._would_run(str(folder))
    assert not os.path.exists(tmp_path / "x.bat")  # only the registry was read
