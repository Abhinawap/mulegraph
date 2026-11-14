"""gather_changelog.py moves each fragment under ## Unreleased, highest issue first (#89)."""

import runpy
from pathlib import Path

import pytest

gather = runpy.run_path(str(Path(__file__).parents[1] / "scripts/gather_changelog.py"))["gather"]


def test_fragments_land_under_unreleased_and_are_removed(tmp_path):
    log = tmp_path / "changelog.md"
    log.write_text("# Changelog\n\n## Unreleased\n\n### old (#1)\n- a\n\n## v1.0\n")
    frags = tmp_path / "changelog.d"
    frags.mkdir()
    (frags / "9.md").write_text("### nine (#9)\n- b\n")
    (frags / "10.md").write_text("### ten (#10)\n- c\n\n")
    gather(log, frags)
    assert log.read_text() == (
        "# Changelog\n\n## Unreleased\n\n### ten (#10)\n- c\n\n### nine (#9)\n- b\n\n"
        "### old (#1)\n- a\n\n## v1.0\n"
    )
    assert list(frags.iterdir()) == []


def test_a_fragment_not_named_by_its_issue_is_refused(tmp_path):
    log = tmp_path / "changelog.md"
    log.write_text("## Unreleased\n\n")
    (tmp_path / "changelog.d").mkdir()
    (tmp_path / "changelog.d/fix-x.md").write_text("### x\n")
    with pytest.raises(SystemExit, match="issue number"):
        gather(log, tmp_path / "changelog.d")
    assert (tmp_path / "changelog.d/fix-x.md").exists()
