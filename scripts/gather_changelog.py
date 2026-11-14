"""Move docs/changelog.d/<issue>.md entries under ## Unreleased, highest issue first (#89)."""

from pathlib import Path

ROOT = Path(__file__).parents[1]
HEADING = "## Unreleased\n\n"


def gather(changelog: Path, fragments: Path) -> None:
    """Insert every fragment below the Unreleased heading, then delete the fragments (#89)."""
    files = list(fragments.glob("*.md"))
    if bad := [f.name for f in files if not f.stem.isdigit()]:
        raise SystemExit(f"{bad}: name each fragment by its issue number (89.md) to order it.")
    files.sort(key=lambda f: int(f.stem), reverse=True)
    text = changelog.read_text()
    if HEADING not in text:
        raise SystemExit(f"{changelog} has no '## Unreleased' heading to gather under.")
    entries = "".join(f.read_text().rstrip("\n") + "\n\n" for f in files)
    changelog.write_text(text.replace(HEADING, HEADING + entries, 1))
    for f in files:
        f.unlink()


if __name__ == "__main__":
    gather(ROOT / "docs/changelog.md", ROOT / "docs/changelog.d")
