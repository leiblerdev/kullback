"""Every relative link in the docs resolves to a file, a directory or a heading anchor.

Usage: python scripts/check_doc_links.py [root]. Reads docs/**/*.md, README.md and AGENTS.md, prints one
line per broken link and exits 1 when there is one. Links inside code fences and web links are skipped.
"""

import re
import sys
from pathlib import Path

LINK_RE = re.compile(r"\]\(([^)\s]+)\)|href=\"([^\"]+)\"")
HEADING_RE = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")


def slug(heading):
    """The anchor GitHub gives a heading: lower case, punctuation dropped, spaces to hyphens."""
    text = re.sub(r"[`*_\[\]()]", "", heading.strip().lower())
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def prose_lines(path):
    """The lines of a Markdown file outside code fences."""
    inside = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            inside = not inside
        elif not inside:
            yield line


def anchors(path):
    return {slug(match.group(1)) for line in prose_lines(path) if (match := HEADING_RE.match(line))}


def broken_links(root):
    root = Path(root)
    pages = sorted(root.glob("docs/**/*.md")) + [root / name for name in ("README.md", "AGENTS.md")]
    for page in filter(Path.is_file, pages):
        for line in prose_lines(page):
            for match in LINK_RE.finditer(line):
                target = match.group(1) or match.group(2)
                if re.match(r"[a-z]+:", target):
                    continue
                file_part, _, anchor = target.partition("#")
                resolved = (page.parent / file_part).resolve() if file_part else page
                if not resolved.exists():
                    yield f"{page.relative_to(root)}: {target} (no such file)"
                elif anchor and resolved.suffix == ".md" and anchor not in anchors(resolved):
                    yield f"{page.relative_to(root)}: {target} (no such heading)"


def main(argv):
    broken = list(broken_links(argv[1] if len(argv) > 1 else Path(__file__).resolve().parent.parent))
    print("\n".join(broken) or "ok doc links")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
