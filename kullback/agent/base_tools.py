"""The base tools every extension may have: read, write, edit, grep, find, ls, inspect, web_search, bash.

Ported from tau_coding's coding tools (read, write, edit, bash: the line slicing and its
continuation hint, the exact-match edit, the head and tail truncation, the shell's captured output)
and bound to a root directory, because an extension of this core owns a directory and nothing
above it. grep, find and ls are ours: tau reaches them through the shell, and an extension whose
shell is an allowlist needs them as tools. inspect is ours too: the shape of a large JSON or JSONL
file (keys, row counts, columns, samples) in one call, where the allowlist has no JSON reader.

Two levels of boundary, and no third. Cannot: the tool is not registered (`only` decides what an
extension gets, so the Examiner has no write and no bash, and the user has no file tools at all).
Should not: what the prompt says. Between them sits the one mechanical rule the tools enforce
themselves, because a path is an argument and not a registration: every path is relative to the
root, and an absolute path or a step above the root is refused with the rule named in the result.
The shell is the one tool that is many tools, so `bash` takes an `Allowlist`: the first word of
every pipeline segment must be on it, and a segment matching a refused pattern never runs. There is
no host shell behind it: each stage runs as its own program with argv from the tokens, joined only
by pipes, so an allowed program is also refused the arguments that would start another (`SPAWNING`).
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import re
import shlex
import signal
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from kullback.agent.tools import AgentTool

__all__ = [
    "Allowlist",
    "CommandRefused",
    "MAX_OUTPUT_BYTES",
    "MAX_OUTPUT_LINES",
    "PathRefused",
    "base_tools",
    "register_base_tools",
    "resolve_in_root",
]

# tau_coding's limits: a result is cut at whichever comes first.
MAX_OUTPUT_LINES = 2_000
MAX_OUTPUT_BYTES = 50 * 1024
DEFAULT_BASH_TIMEOUT_S = 120.0
GREP_MAX_MATCHES = 200
FIND_MAX_PATHS = 500

INSPECT_MAX_CHARS = 4_000
INSPECT_ROW_CHARS = 200
INSPECT_MAX_ROWS = 5
INSPECT_MAX_FIELDS = 60
# The input cap: a file at most this size is parsed whole, one over it never is. 8 MiB clears every
# file a Builder reads to learn the corpus with room to spare (on the smoke 8 retail workdir the world
# is 0.5 MB, the largest call log 1.8 MB, the run digest 4.7 MB, the Builder session 5 MB), while the
# files that grow with trace volume (the event bus, the raw corpus, the exam's run store, 32 MB to
# 660 MB there) are past it, and parsing one of those whole could exhaust memory or stall the loop.
INSPECT_MAX_BYTES = 8 * 1024 * 1024
# Over the input cap a JSONL file is summarised from its head: at most this many lines, and never more
# than INSPECT_MAX_BYTES read. A thousand rows is enough to see every key, result type and error class
# a log carries, and the summary says the counts are over those lines only.
INSPECT_SAMPLE_LINES = 1_000

TOOL_NAMES = ("read", "write", "edit", "grep", "find", "ls", "inspect", "web_search", "bash")


class PathRefused(ValueError):
    """A path argument the root fence does not allow; the message names the rule."""


class CommandRefused(ValueError):
    """A shell command the allowlist does not allow; the message names the rule."""


class NotConfigured(RuntimeError):
    """A tool whose backing service this workdir has not been given."""


# --- the root fence ----------------------------------------------------------


def resolve_in_root(root: Path, path: str) -> Path:
    """The file `path` names under `root`, or a refusal naming the rule it broke.

    Relative to the root, always: an absolute path, a step above the root, and a link that leads
    out of it are all the same mistake, which is asking for a file this extension does not own.
    An empty path reads as the root itself.
    """
    text = (path or "").strip() or "."
    candidate = Path(text)
    if candidate.is_absolute():
        raise PathRefused(
            f"rule root: {text!r} is an absolute path; paths are relative to this agent's root directory"
        )
    if any(part == ".." for part in candidate.parts):
        raise PathRefused(f"rule root: {text!r} steps above this agent's root directory with '..'")
    resolved = (root / candidate).resolve()
    base = root.resolve()
    if resolved != base and base not in resolved.parents:
        raise PathRefused(f"rule root: {text!r} resolves outside this agent's root directory")
    return resolved


def _relative(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:  # pragma: no cover - resolve_in_root has already fenced the path
        return str(path)


# --- truncation (tau_coding's head and tail) ---------------------------------


class Truncation(BaseModel):
    """What a result had to leave out, so the model reads how much it is not seeing."""

    model_config = ConfigDict(extra="forbid")

    truncated: bool = False
    truncated_by: Optional[str] = None
    total_lines: int = 0
    total_bytes: int = 0
    output_lines: int = 0
    output_bytes: int = 0
    # A line that alone exceeds the byte limit is shown cut inside the line, not dropped.
    line_cut: bool = False


def truncate_head(content: str, max_lines: int = MAX_OUTPUT_LINES,
                  max_bytes: int = MAX_OUTPUT_BYTES) -> tuple[str, Truncation]:
    """The first lines that fit; what a read of a file returns.

    A first line that alone exceeds `max_bytes` is shown cut to its head and counted as one
    shown line (`line_cut`), so a file of long lines is still readable one line at a time.
    """
    lines = content.split("\n")
    total_bytes = len(content.encode())
    if len(lines) <= max_lines and total_bytes <= max_bytes:
        return content, Truncation(total_lines=len(lines), total_bytes=total_bytes,
                                   output_lines=len(lines), output_bytes=total_bytes)
    kept: list[str] = []
    size = 0
    by = "lines"
    for index, line in enumerate(lines[:max_lines]):
        cost = len(line.encode()) + (1 if index else 0)
        if size + cost > max_bytes:
            by = "bytes"
            if not kept:
                kept.append(line.encode()[:max_bytes].decode(errors="ignore"))
            break
        kept.append(line)
        size += cost
    output = "\n".join(kept)
    return output, Truncation(truncated=True, truncated_by=by, total_lines=len(lines),
                              total_bytes=total_bytes, output_lines=len(kept),
                              output_bytes=len(output.encode()),
                              line_cut=by == "bytes" and len(kept) == 1 and kept[0] != lines[0])


def truncate_tail(content: str, max_lines: int = MAX_OUTPUT_LINES,
                  max_bytes: int = MAX_OUTPUT_BYTES) -> tuple[str, Truncation]:
    """The last lines that fit; what a command's output returns, because its end is its answer."""
    lines = content.split("\n")
    total_bytes = len(content.encode())
    if len(lines) <= max_lines and total_bytes <= max_bytes:
        return content, Truncation(total_lines=len(lines), total_bytes=total_bytes,
                                   output_lines=len(lines), output_bytes=total_bytes)
    kept: list[str] = []
    size = 0
    by = "lines"
    cut = False
    for line in reversed(lines):
        if len(kept) >= max_lines:
            by = "lines"
            break
        cost = len(line.encode()) + (1 if kept else 0)
        if size + cost > max_bytes:
            by = "bytes"
            if not kept:
                clipped = line.encode()[-max_bytes:].decode(errors="ignore")
                kept.insert(0, clipped)
                size = len(clipped.encode())
                cut = True
            break
        kept.insert(0, line)
        size += cost
    output = "\n".join(kept)
    return output, Truncation(truncated=True, truncated_by=by, total_lines=len(lines),
                              total_bytes=total_bytes, output_lines=len(kept),
                              output_bytes=len(output.encode()), line_cut=cut)


# --- the shell allowlist -----------------------------------------------------

DEFAULT_REFUSED_PATTERNS = (
    r"(^|\s)rm(\s|$)",
    r"(^|\s)mv(\s|$)",
    r"(^|\s)git(\s|$)",
    r"(^|\s)curl(\s|$)",
    r"(^|\s)wget(\s|$)",
    r"(^|\s)pip(\s|$)",
    r"(^|\s)uv(\s|$)",
    r"(^|\s)sudo(\s|$)",
    r"(^|\s)chmod(\s|$)",
    r"(^|\s)/",
    r"\.\.",
)

_OPERATORS = {"|", "||", "&&", ";", "&", "\n", "(", ")", "{", "}"}
_PUNCTUATION = set("();<>|&\n")

# The arguments by which an allowed program would start another program, write where it was not
# asked to, or run code it read from a file nobody checked: a pattern per primitive, matched against
# every argument after the program's name, and the primitive named in the refusal.
_SED_ADDRESS = r"(\d+|\$|/[^/]*/)?(,(\d+|\$|/[^/]*/))?\s*!?\s*"
SPAWNING: dict[str, tuple[tuple[str, str], ...]] = {
    "awk": (("system(", r"system\s*\("), ("getline", r"getline"), ("a pipe", r"\|"),
            ("a program file", r"^(-f|--file)")),
    "find": (("-exec", r"^-exec$"), ("-execdir", r"^-execdir$"), ("-ok", r"^-ok$"), ("-okdir", r"^-okdir$"),
             ("-delete", r"^-delete$"), ("-fprint", r"^-fprint0?$"), ("-fprintf", r"^-fprintf$"),
             ("-fls", r"^-fls$")),
    "sed": (("the e command", rf"(^|[;{{}}\n])\s*{_SED_ADDRESS}e(\s|;|}}|$)"),
            ("the w command", rf"(^|[;{{}}\n])\s*{_SED_ADDRESS}[wW](\s|$)"),
            ("the e or w flag", r"(^|[;{}\n\s])s(.)(?:\\.|(?!\2).)*\2(?:\\.|(?!\2).)*\2[0-9gpiImM]*[ewW]"),
            ("-i", r"^(-[a-zA-Z]*i|--in-place)"), ("a script file", r"^(-[a-zA-Z]*f|--file)")),
    "sort": (("-o", r"^(-[a-zA-Z]*o|--output)"), ("--compress-program", r"^--compress-program")),
}
SPAWNING["gawk"] = SPAWNING["awk"] + (("@load", r"@load"),)


@dataclass(frozen=True)
class Allowlist:
    """What `bash` may run: the commands by name, and the patterns no segment may match.

    The shell is the one tool that is many tools, so the check is per pipeline segment: `find . |
    head` is two commands and both must be allowed. A refused pattern is the second half of the
    same rule, for what a command name alone does not say: a path out of the root, a step above it,
    a network fetch, a package install, anything that changes the repository.
    """

    commands: frozenset[str] = frozenset()
    refused_patterns: tuple[str, ...] = DEFAULT_REFUSED_PATTERNS

    @classmethod
    def of(cls, commands: Iterable[str], refused_patterns: Optional[Sequence[str]] = None) -> "Allowlist":
        return cls(
            commands=frozenset(commands),
            refused_patterns=tuple(DEFAULT_REFUSED_PATTERNS if refused_patterns is None else refused_patterns),
        )

    @staticmethod
    def tokens(command: str) -> list[str]:
        """The command's words and operators, as a POSIX shell would split them."""
        try:
            lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
            lexer.whitespace_split = True
            return list(lexer)
        except ValueError as exc:
            raise CommandRefused(f"rule shell: the command does not parse as a shell command ({exc})") from exc

    def segments(self, command: str) -> list[list[str]]:
        """The command's pipeline segments as token lists; a shell it cannot parse is one segment."""
        segments: list[list[str]] = [[]]
        for token in self.tokens(command):
            if token in _OPERATORS:
                segments.append([])
                continue
            segments[-1].append(token)
        return [segment for segment in segments if segment]

    def check(self, command: str) -> None:
        """Raise `CommandRefused` naming the rule, or return with the command cleared to run."""
        if not command.strip():
            raise CommandRefused("rule shell: the command is empty")
        segments = self.segments(command)
        if not segments:
            raise CommandRefused("rule shell: the command is empty")
        for segment in segments:
            text = " ".join(segment)
            for pattern in self.refused_patterns:
                if re.search(pattern, text):
                    raise CommandRefused(
                        f"rule refused_pattern: the segment {text!r} matches {pattern!r} and is not run"
                    )
            argv = _argv(segment)
            name = argv[0]
            if name not in self.commands:
                allowed = ", ".join(sorted(self.commands)) or "nothing"
                raise CommandRefused(
                    f"rule allowlist: {name!r} is not one of the commands this agent may run ({allowed})"
                )
            for primitive, pattern in SPAWNING.get(name, ()):
                if any(re.search(pattern, argument) for argument in argv[1:]):
                    raise CommandRefused(
                        f"rule spawning: {name} may not use {primitive} here, it would start another program "
                        "or write outside what the command shows"
                    )
        self.stages(command)

    def stages(self, command: str) -> list[list[str]]:
        """The argv of each pipeline stage; any operator other than `|` is refused."""
        stages: list[list[str]] = [[]]
        for token in self.tokens(command):
            if token == "|":
                stages.append([])
            elif token and set(token) <= _PUNCTUATION:
                raise CommandRefused(f"rule shell: only pipes join commands here, not {token!r}")
            else:
                stages[-1].append(token)
        if any(not stage for stage in stages):
            raise CommandRefused("rule shell: a pipe needs a command on both sides")
        return [_argv(stage) for stage in stages]


def _argv(segment: list[str]) -> list[str]:
    """The segment from its program's name on: leading `NAME=value` words are dropped, not exported."""
    start = next((index for index, token in enumerate(segment) if "=" not in token.split("/")[0]), 0)
    return segment[start:]


DEFAULT_ALLOWLIST = Allowlist.of(
    ("ls", "cat", "head", "tail", "wc", "grep", "find", "sort", "uniq", "cut", "sed", "awk", "echo", "diff")
)


# --- the tools ---------------------------------------------------------------


class ReadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Path to the file, relative to your root directory.")
    offset: Optional[int] = Field(default=None, description="First line to read, 1 indexed.")
    limit: Optional[int] = Field(default=None, description="How many lines to read.")


class ReadResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    text: str
    lines: int
    note: str = ""
    truncation: Truncation = Field(default_factory=Truncation)


class WriteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Path to the file, relative to your root directory.")
    content: str = Field(description="The whole content of the file.")


class WriteResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    characters: int
    created: bool


class EditArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Path to the file, relative to your root directory.")
    old: str = Field(description="The exact text to replace; it must occur once unless replace_all.")
    new: str = Field(description="What stands in its place.")
    replace_all: bool = Field(default=False, description="Replace every occurrence instead of the one.")


class EditResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    replacements: int
    first_changed_line: Optional[int] = None


class GrepArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str = Field(description="A regular expression.")
    path: str = Field(default=".", description="File or directory to search, relative to your root.")
    glob: Optional[str] = Field(default=None, description="Only files whose name matches this glob.")


class GrepMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    line: int
    text: str


class GrepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str
    matches: list[GrepMatch] = Field(default_factory=list)
    files_searched: int = 0
    truncated: bool = False


class FindArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    glob: str = Field(description="A glob such as '**/*.py', matched against paths under `path`.")
    path: str = Field(default=".", description="Directory to search, relative to your root.")


class FindResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: list[str] = Field(default_factory=list)
    truncated: bool = False


class LsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(default=".", description="Directory to list, relative to your root.")


class LsEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: str
    size: int = 0


class LsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    entries: list[LsEntry] = Field(default_factory=list)


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="What to search the web for.")


class SearchHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = ""
    url: str = ""
    snippet: str = ""


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    hits: list[SearchHit] = Field(default_factory=list)


class InspectArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="A .json or .jsonl file, relative to your root.")
    key: Optional[str] = Field(default=None, description="A dotted path into a JSON document, such as a table name.")
    rows: int = Field(default=2, description=f"Sample rows to show, 0 to {INSPECT_MAX_ROWS}.")


class InspectResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    summary: str
    truncated: bool = False


class BashArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str = Field(description="The command to run, in your root directory.")
    timeout: Optional[float] = Field(default=None, description="Seconds to wait before it is killed.")


class BashResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str
    output: str
    exit_code: Optional[int] = None
    timed_out: bool = False
    duration_seconds: float = 0.0
    truncation: Truncation = Field(default_factory=Truncation)


@dataclass
class BaseToolConfig:
    """What the base tools are bound to: one root, one allowlist, one timeout."""

    root: Path
    allowlist: Allowlist = field(default_factory=lambda: DEFAULT_ALLOWLIST)
    timeout_s: float = DEFAULT_BASH_TIMEOUT_S


# How many same-stem siblings a missing-file refusal names (F35).
SIBLINGS_SHOWN = 5


def _siblings_note(root: Path, path: Path) -> str:
    """The files beside a missing one that share its stem, so a wrong extension costs one turn (F35)."""
    if not path.parent.is_dir():
        return ""
    names = sorted(_relative(root, p) for p in path.parent.iterdir() if p.is_file() and p.stem == path.stem)
    return f"; the directory holds {', '.join(names[:SIBLINGS_SHOWN])}" if names else ""


def _read_tool(config: BaseToolConfig) -> AgentTool:
    async def execute(args: ReadArgs) -> ReadResult:
        path = resolve_in_root(config.root, args.path)
        if not path.exists():
            raise PathRefused(f"no file {args.path!r} under this agent's root directory"
                              + _siblings_note(config.root, path))
        if path.is_dir():
            raise PathRefused(f"{args.path!r} is a directory; ls lists one and read reads a file")
        if args.offset is not None and args.offset < 0:
            raise ValueError("offset must be at least 0")
        if args.limit is not None and args.limit < 1:
            raise ValueError("limit must be at least 1")
        text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        lines = text.split("\n")
        start = 0 if not args.offset else args.offset - 1
        if start >= len(lines):
            # Past the end answers what is there rather than failing the call (F31).
            return ReadResult(path=_relative(config.root, path), text="", lines=len(lines),
                              note=f"the file has {len(lines)} lines; nothing at offset {args.offset}")
        end = min(start + args.limit, len(lines)) if args.limit is not None else len(lines)
        selected = "\n".join(lines[start:end])
        output, truncation = truncate_head(selected)
        note = ""
        shown = truncation.output_lines if truncation.truncated else end - start
        if truncation.line_cut:
            note = (f"line {start + 1} is longer than {truncation.output_bytes} bytes and was cut there; "
                    f"read again with offset={start + 2} for the next line")
        elif truncation.truncated:
            note = (f"showing lines {start + 1} to {start + shown} of {len(lines)}; "
                    f"read again with offset={start + shown + 1} to continue")
        elif end < len(lines):
            note = (f"{len(lines) - end} more lines in the file; "
                    f"read again with offset={end + 1} to continue")
        return ReadResult(path=_relative(config.root, path), text=output, lines=len(lines),
                          note=note, truncation=truncation)

    def render(result: ReadResult) -> str:
        return f"{result.text}\n[{result.note}]" if result.note else result.text

    return AgentTool(
        "read",
        "Read a file under your root directory. Slice a large one with offset and limit.",
        ReadArgs,
        ReadResult,
        execute,
        render=render,
        prompt_snippet="Read a file",
    )


def _write_tool(config: BaseToolConfig) -> AgentTool:
    async def execute(args: WriteArgs) -> WriteResult:
        path = resolve_in_root(config.root, args.path)
        if path.is_dir():
            raise PathRefused(f"{args.path!r} is a directory")
        created = not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args.content, encoding="utf-8")
        return WriteResult(path=_relative(config.root, path), characters=len(args.content), created=created)

    def render(result: WriteResult) -> str:
        what = "wrote a new" if result.created else "overwrote"
        return f"{what} file {result.path}, {result.characters} characters"

    return AgentTool(
        "write",
        "Write a whole file under your root directory, creating it and its parents when needed.",
        WriteArgs,
        WriteResult,
        execute,
        render=render,
        prompt_snippet="Write a whole file",
    )


def _edit_tool(config: BaseToolConfig) -> AgentTool:
    async def execute(args: EditArgs) -> EditResult:
        path = resolve_in_root(config.root, args.path)
        if not path.is_file():
            raise PathRefused(f"no file {args.path!r} to edit under this agent's root directory")
        if not args.old:
            raise ValueError("old must not be empty: edit replaces exact text, it does not insert")
        content = path.read_text(encoding="utf-8")
        occurrences = content.count(args.old)
        if occurrences == 0:
            raise ValueError(f"the text to replace does not occur in {args.path}; read it and match it exactly")
        if occurrences > 1 and not args.replace_all:
            raise ValueError(
                f"the text to replace occurs {occurrences} times in {args.path}; "
                "give more context so it is unique, or set replace_all"
            )
        if args.old == args.new:
            raise ValueError("old and new are the same text, so the edit would change nothing")
        updated = content.replace(args.old, args.new) if args.replace_all else content.replace(args.old, args.new, 1)
        path.write_text(updated, encoding="utf-8")
        before = content.split("\n")
        after = updated.split("\n")
        changed = next((i + 1 for i, (a, b) in enumerate(zip(before, after, strict=False)) if a != b), None)
        if changed is None and len(before) != len(after):
            changed = min(len(before), len(after)) + 1
        return EditResult(
            path=_relative(config.root, path),
            replacements=occurrences if args.replace_all else 1,
            first_changed_line=changed,
        )

    def render(result: EditResult) -> str:
        where = f" from line {result.first_changed_line}" if result.first_changed_line else ""
        return f"replaced {result.replacements} occurrence(s) in {result.path}{where}"

    return AgentTool(
        "edit",
        "Replace exact text in one file. The text must occur once unless replace_all is set.",
        EditArgs,
        EditResult,
        execute,
        render=render,
        prompt_snippet="Edit a file by exact replacement",
    )


def _files_under(root: Path, base: Path, glob: Optional[str] = None) -> list[Path]:
    if base.is_file():
        return [base]
    found = [p for p in sorted(base.rglob("*")) if p.is_file()]
    if glob:
        found = [p for p in found if fnmatch.fnmatch(p.name, glob) or fnmatch.fnmatch(_relative(root, p), glob)]
    return found


def _grep_tool(config: BaseToolConfig) -> AgentTool:
    async def execute(args: GrepArgs) -> GrepResult:
        base = resolve_in_root(config.root, args.path)
        if not base.exists():
            raise PathRefused(f"no path {args.path!r} under this agent's root directory")
        try:
            expression = re.compile(args.pattern)
        except re.error as exc:
            raise ValueError(f"{args.pattern!r} is not a regular expression: {exc}") from exc
        matches: list[GrepMatch] = []
        files = _files_under(config.root, base, args.glob)
        truncated = False
        for file in files:
            try:
                text = file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for number, line in enumerate(text.split("\n"), start=1):
                if expression.search(line):
                    if len(matches) >= GREP_MAX_MATCHES:
                        truncated = True
                        break
                    matches.append(GrepMatch(path=_relative(config.root, file), line=number, text=line[:500]))
            if truncated:
                break
        return GrepResult(pattern=args.pattern, matches=matches, files_searched=len(files), truncated=truncated)

    def render(result: GrepResult) -> str:
        if not result.matches:
            return f"no match for {result.pattern!r} in {result.files_searched} file(s)"
        lines = [f"{m.path}:{m.line}: {m.text}" for m in result.matches]
        if result.truncated:
            lines.append(f"[the first {len(result.matches)} matches; narrow the pattern or the path for the rest]")
        return "\n".join(lines)

    return AgentTool(
        "grep",
        "Search files under your root directory for a regular expression.",
        GrepArgs,
        GrepResult,
        execute,
        render=render,
        prompt_snippet="Search file contents",
    )


def _find_tool(config: BaseToolConfig) -> AgentTool:
    async def execute(args: FindArgs) -> FindResult:
        base = resolve_in_root(config.root, args.path)
        if not base.is_dir():
            raise PathRefused(f"{args.path!r} is not a directory under this agent's root directory")
        found = [_relative(config.root, p) for p in sorted(base.glob(args.glob))]
        return FindResult(paths=found[:FIND_MAX_PATHS], truncated=len(found) > FIND_MAX_PATHS)

    def render(result: FindResult) -> str:
        if not result.paths:
            return "no path matched"
        text = "\n".join(result.paths)
        return text + ("\n[the first paths only; narrow the glob for the rest]" if result.truncated else "")

    return AgentTool(
        "find",
        "List the paths under your root directory that match a glob, such as '**/*.py'.",
        FindArgs,
        FindResult,
        execute,
        render=render,
        prompt_snippet="Find files by name",
    )


def _ls_tool(config: BaseToolConfig) -> AgentTool:
    async def execute(args: LsArgs) -> LsResult:
        base = resolve_in_root(config.root, args.path)
        if not base.exists():
            raise PathRefused(f"no path {args.path!r} under this agent's root directory")
        if base.is_file():
            return LsResult(path=_relative(config.root, base),
                            entries=[LsEntry(name=base.name, kind="file", size=base.stat().st_size)])
        entries = [
            LsEntry(name=child.name, kind="directory" if child.is_dir() else "file",
                    size=0 if child.is_dir() else child.stat().st_size)
            for child in sorted(base.iterdir())
        ]
        return LsResult(path=_relative(config.root, base), entries=entries)

    def render(result: LsResult) -> str:
        if not result.entries:
            return f"{result.path} is empty"
        return "\n".join(
            f"{e.name}/" if e.kind == "directory" else f"{e.name} ({e.size} bytes)" for e in result.entries
        )

    return AgentTool(
        "ls",
        "List a directory under your root directory.",
        LsArgs,
        LsResult,
        execute,
        render=render,
        prompt_snippet="List a directory",
    )


# --- inspect: the shape of a JSON document or a JSONL file ---------------------


def _kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    return "object"


def _table_rows(value: Any) -> Optional[list[tuple[Optional[str], dict]]]:
    """The rows of a table (an object of row objects, or a list of objects), else None.

    An object of objects is a table only when its rows share their columns: the cells filled are
    at least half of rows times columns. An object of differently shaped objects is a document.
    """
    if isinstance(value, dict) and value and all(isinstance(row, dict) for row in value.values()):
        columns = set().union(*value.values())
        filled = sum(len(row) for row in value.values())
        if columns and filled * 2 < len(value) * len(columns):
            return None
        return [(str(key), row) for key, row in value.items()]
    if isinstance(value, list) and value and all(isinstance(row, dict) for row in value):
        return [(None, row) for row in value]
    return None


def _size(value: Any) -> str:
    if _table_rows(value) is not None:
        return f"table of {len(value)} rows"
    if isinstance(value, dict):
        return f"{len(value)} keys"
    if isinstance(value, list):
        return f"{len(value)} items"
    if isinstance(value, str):
        return f"{len(value)} chars"
    return _compact(value)


def _compact(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    return text if len(text) <= INSPECT_ROW_CHARS else text[:INSPECT_ROW_CHARS] + "..."


def _fields(rows: Iterable[dict], total: int) -> list[str]:
    """Each key the rows carry, with its value types and how many of the `total` rows carry it."""
    counts: Counter[str] = Counter()
    kinds: dict[str, Counter[str]] = {}
    for row in rows:
        for key, value in row.items():
            counts[key] += 1
            kinds.setdefault(key, Counter())[_kind(value)] += 1
    lines = [f"  {key}: {'|'.join(kind for kind, _ in kinds[key].most_common())} ({count} of {total} rows)"
             for key, count in list(counts.items())[:INSPECT_MAX_FIELDS]]
    if len(counts) > INSPECT_MAX_FIELDS:
        lines.append(f"  [and {len(counts) - INSPECT_MAX_FIELDS} more columns]")
    return lines


def _describe(value: Any, rows: int) -> list[str]:
    table = _table_rows(value)
    if table is not None:
        lines = [f"table: {len(table)} rows ({'object keyed by row id' if isinstance(value, dict) else 'list'})",
                 "columns:"]
        lines += _fields((row for _, row in table), len(table))
        if rows:
            lines.append(f"sample rows ({min(rows, len(table))}):")
            lines += [f"  {key}: {_compact(row)}" if key is not None else f"  {_compact(row)}"
                      for key, row in table[:rows]]
        return lines
    if isinstance(value, dict):
        lines = [f"object with {len(value)} keys:"] + [
            f"  {key}: {_kind(item)}, {_size(item)}" for key, item in list(value.items())[:INSPECT_MAX_FIELDS]]
        if len(value) > INSPECT_MAX_FIELDS:
            lines.append(f"  [and {len(value) - INSPECT_MAX_FIELDS} more keys]")
        return lines
    if isinstance(value, list):
        kinds = Counter(_kind(item) for item in value)
        lines = [f"list of {len(value)} items: " + ", ".join(f"{kind} {count}" for kind, count in kinds.most_common())]
        return lines + [f"  {_compact(item)}" for item in value[:rows]]
    return [f"{_kind(value)}: {_compact(value)}"]


def _at_key(document: Any, key: str) -> Any:
    value = document
    walked: list[str] = []
    for part in key.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.lstrip("-").isdigit() and -len(value) <= int(part) < len(value):
            value = value[int(part)]
        else:
            where = ".".join(walked) or "the document"
            options = (", ".join(list(value)[:20]) if isinstance(value, dict)
                       else f"indexes 0 to {len(value) - 1}" if isinstance(value, list) else "nothing below it")
            raise KeyError(f"no {part!r} under {where}; it holds {options}")
        walked.append(part)
    return value


def _shape(value: Any) -> str:
    """A generic name for a value's shape: its type, and its keys when it is an object."""
    if isinstance(value, dict):
        keys = sorted(value)
        return "object{" + ",".join(keys[:8]) + (",..." if len(keys) > 8 else "") + "}"
    if isinstance(value, list):
        inner = sorted({_kind(item) for item in value})
        return f"list[{'|'.join(inner)}]" if inner else "list[]"
    return _kind(value)


def _error_class(value: Any) -> str:
    """The class of an error value: a type or name field of an object, or the head of a message."""
    if isinstance(value, dict):
        for field_name in ("type", "class", "name", "code", "kind"):
            if isinstance(value.get(field_name), (str, int)):
                return str(value[field_name])
        return _shape(value)
    if isinstance(value, str):
        head = value.split(":", 1)[0].strip()
        return head[:60] or "(empty)"
    return _kind(value)


def _counted(label: str, counts: Counter[str]) -> list[str]:
    return [f"{label} ({sum(counts.values())} lines):"] + [
        f"  {name}: {count}" for name, count in counts.most_common()]


def _describe_lines(text: str, rows: int, of_bytes: Optional[int] = None) -> list[str]:
    """A JSONL summary; with `of_bytes`, `text` is the head of a file that size and the header says so."""
    records: list[Any] = []
    broken = 0
    lines = [line for line in text.split("\n") if line.strip()]
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            broken += 1
    objects = [record for record in records if isinstance(record, dict)]
    head = f"jsonl: {len(lines)} lines"
    if of_bytes is not None:
        head = (f"jsonl: the file is {of_bytes} bytes, over the {INSPECT_MAX_BYTES} byte inspect cap, so every "
                f"count below is over its first {len(lines)} lines only")
    out = [head + (f", {broken} not JSON" if broken else "")]
    if len(objects) < len(records):
        out.append(f"{len(records) - len(objects)} lines are not objects")
    if objects:
        out += ["keys:"] + _fields(objects, len(objects))
    results = Counter(_shape(record["result"]) for record in objects if "result" in record)
    if results:
        out += _counted("result types", results)
    errors = Counter(_error_class(record["error"]) for record in objects if record.get("error") is not None)
    if errors:
        out += _counted("error classes", errors)
    if rows and records:
        out.append(f"sample lines ({min(rows, len(records))}):")
        out += [f"  {_compact(record)}" for record in records[:rows]]
    return out


def _capped(lines: list[str]) -> tuple[str, bool]:
    kept: list[str] = []
    size = 0
    for index, line in enumerate(lines):
        if size + len(line) + 1 > INSPECT_MAX_CHARS:
            kept.append(f"[and {len(lines) - index} more lines; pass key to look inside one value]")
            return "\n".join(kept), True
        kept.append(line)
        size += len(line) + 1
    return "\n".join(kept), False


def _head_lines(path: Path) -> str:
    """The first INSPECT_SAMPLE_LINES lines of a file, reading no more than INSPECT_MAX_BYTES in all.

    A line the byte budget cuts short is dropped rather than parsed as a broken row.
    """
    kept: list[str] = []
    budget = INSPECT_MAX_BYTES
    with path.open("rb") as handle:
        while len(kept) < INSPECT_SAMPLE_LINES and budget > 0:
            raw = handle.readline(budget)
            if not raw:
                break
            budget -= len(raw)
            if not raw.endswith(b"\n") and budget <= 0:
                break
            kept.append(raw.decode("utf-8", errors="replace").rstrip("\r\n"))
    return "\n".join(kept)


def _first_line_is_json(text: str) -> bool:
    first = next((line for line in text.split("\n") if line.strip()), None)
    if first is None:
        return False
    try:
        json.loads(first)
    except json.JSONDecodeError:
        return False
    return True


def _inspect_large(path: Path, shown: str, size: int, args: InspectArgs) -> list[str]:
    """A file over INSPECT_MAX_BYTES: JSONL from a bounded head, one JSON document refused.

    Whether it is JSONL is read off its head: a first line that parses on its own is a row. A
    document spread over lines, or one line longer than the whole budget, never does.
    """
    head = _head_lines(path)
    if not _first_line_is_json(head):
        raise ValueError(f"{shown} is {size} bytes, over the {INSPECT_MAX_BYTES} byte inspect cap, and is one "
                         "JSON document, so inspect does not parse it; read it with offset and limit, or grep "
                         "it for the key you are after")
    if args.key:
        raise ValueError(f"{shown} is not one JSON document, so key does not apply; inspect it without key")
    return _describe_lines(head, args.rows, of_bytes=size)


def _inspect_tool(config: BaseToolConfig) -> AgentTool:
    async def execute(args: InspectArgs) -> InspectResult:
        path = resolve_in_root(config.root, args.path)
        if not path.is_file():
            raise PathRefused(f"no file {args.path!r} under this agent's root directory"
                              + _siblings_note(config.root, path))
        if not 0 <= args.rows <= INSPECT_MAX_ROWS:
            raise ValueError(f"rows must be 0 to {INSPECT_MAX_ROWS}")
        size = path.stat().st_size
        if size > INSPECT_MAX_BYTES:
            summary, truncated = _capped(_inspect_large(path, args.path, size, args))
            return InspectResult(path=_relative(config.root, path), summary=summary, truncated=truncated)
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            document = json.loads(text)
        except json.JSONDecodeError:
            if args.key:
                raise ValueError(f"{args.path} is not one JSON document, so key does not apply; "
                                 "inspect it without key") from None
            lines = _describe_lines(text, args.rows)
        else:
            value = _at_key(document, args.key) if args.key else document
            where = f"{args.key}: " if args.key else "document: "
            lines = _describe(value, args.rows)
            lines[0] = where + lines[0]
        summary, truncated = _capped(lines)
        return InspectResult(path=_relative(config.root, path), summary=summary, truncated=truncated)

    def render(result: InspectResult) -> str:
        return result.summary

    return AgentTool(
        "inspect",
        "Summarise a JSON or JSONL file under your root directory: keys, sizes, table rows, columns and "
        "their types, sample rows, and for JSONL the result types and error classes. key looks inside one value.",
        InspectArgs,
        InspectResult,
        execute,
        render=render,
        prompt_snippet="Summarise the shape of a JSON or JSONL file",
    )


def _search_backend() -> Any:
    """The provider layer's web search, when it has one. `kullback.ai` owns which service that is."""
    from kullback import ai  # noqa: PLC0415 - imported here so the tool module stays cheap

    return getattr(ai, "web_search", None)


def _web_search_tool(config: BaseToolConfig) -> AgentTool:
    async def execute(args: SearchArgs) -> SearchResult:
        backend = _search_backend()
        if backend is None:
            raise NotConfigured(
                "web_search is not configured in this workdir: the provider layer has no search "
                "service, so nothing can be looked up. Work from what is in your root directory, "
                "and say in your answer what you could not look up."
            )
        hits = await asyncio.to_thread(backend, args.query)
        return SearchResult(query=args.query, hits=[SearchHit(**dict(hit)) for hit in hits])

    def render(result: SearchResult) -> str:
        if not result.hits:
            return f"no result for {result.query!r}"
        return "\n".join(f"{hit.title} ({hit.url})\n{hit.snippet}" for hit in result.hits)

    return AgentTool(
        "web_search",
        "Search the web for a query, when this workdir has a search service.",
        SearchArgs,
        SearchResult,
        execute,
        render=render,
        prompt_snippet="Search the web",
    )


def _bash_tool(config: BaseToolConfig) -> AgentTool:
    async def execute(args: BashArgs) -> BashResult:
        config.allowlist.check(args.command)
        timeout = args.timeout if args.timeout is not None else config.timeout_s
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0")
        started = time.monotonic()
        raw, exit_code, timed_out = await _run_pipeline(config.allowlist.stages(args.command), config.root, timeout)
        output, truncation = truncate_tail(raw.decode(errors="replace"))
        if timed_out:
            output = f"{output}\n[the command was killed after {timeout:g} seconds]".strip()
        return BashResult(
            command=args.command,
            output=output or "(no output)",
            exit_code=exit_code,
            timed_out=timed_out,
            duration_seconds=round(time.monotonic() - started, 3),
            truncation=truncation,
        )

    def render(result: BashResult) -> str:
        text = result.output
        if result.truncation.truncated:
            first = result.truncation.total_lines - result.truncation.output_lines + 1
            text += (f"\n[the last lines only: {first} to {result.truncation.total_lines} "
                     f"of {result.truncation.total_lines}]")
        if result.exit_code not in (0, None):
            text += f"\n[exit code {result.exit_code}]"
        return text

    return AgentTool(
        "bash",
        "Run a command in your root directory. Only the commands on your allowlist run.",
        BashArgs,
        BashResult,
        execute,
        render=render,
        prompt_snippet="Run an allowed shell command",
    )


_FACTORIES = {
    "read": _read_tool,
    "write": _write_tool,
    "edit": _edit_tool,
    "grep": _grep_tool,
    "find": _find_tool,
    "ls": _ls_tool,
    "inspect": _inspect_tool,
    "web_search": _web_search_tool,
    "bash": _bash_tool,
}


def base_tools(root: Any, allowlist: Optional[Allowlist] = None, only: Optional[Sequence[str]] = None,
               timeout_s: float = DEFAULT_BASH_TIMEOUT_S) -> list[AgentTool]:
    """The base tools bound to `root`; `only` names the subset this extension gets."""
    config = BaseToolConfig(root=Path(root), allowlist=allowlist or DEFAULT_ALLOWLIST, timeout_s=timeout_s)
    names = list(only) if only is not None else list(TOOL_NAMES)
    unknown = [name for name in names if name not in _FACTORIES]
    if unknown:
        raise ValueError(f"no base tool named {', '.join(unknown)}; the base tools are {', '.join(TOOL_NAMES)}")
    return [_FACTORIES[name](config) for name in names]


def register_base_tools(api: Any, root: Any, allowlist: Optional[Allowlist] = None,
                        only: Optional[Sequence[str]] = None,
                        timeout_s: float = DEFAULT_BASH_TIMEOUT_S) -> list[AgentTool]:
    """Register the subset on anything with `register_tool`: an ExtensionAPI or a harness."""
    tools = base_tools(root, allowlist=allowlist, only=only, timeout_s=timeout_s)
    for tool in tools:
        api.register_tool(tool)
    return tools


def _pipeline_env() -> dict[str, str]:
    """PATH and LANG from the host and nothing else: no credential in the environment reaches a stage."""
    return {"PATH": os.environ.get("PATH", os.defpath), "LANG": os.environ.get("LANG", "C.UTF-8")}


async def _run_pipeline(stages: list[list[str]], root: Path,
                        timeout: float) -> tuple[bytes, Optional[int], bool]:
    """Run each stage as its own program, stdout of one into stdin of the next, with no shell between.

    Every stage's stderr and the last stage's stdout land in one pipe, read as the output. The exit
    code is the last non-zero one of any stage, as with pipefail, except a stage the next one stopped
    reading from (SIGPIPE), which is how `find | head` ends and not a failure.
    """
    env = _pipeline_env()
    out_read, out_write = os.pipe()
    processes: list[Any] = []
    stdin: int = asyncio.subprocess.DEVNULL
    failure: Optional[str] = None
    try:
        for index, argv in enumerate(stages):
            last = index == len(stages) - 1
            next_stdin, stdout = (asyncio.subprocess.DEVNULL, out_write) if last else os.pipe()
            try:
                processes.append(await asyncio.create_subprocess_exec(
                    *argv, cwd=str(root), env=env, stdin=stdin, stdout=stdout, stderr=out_write,
                    start_new_session=os.name == "posix",
                ))
            except OSError as exc:
                failure = f"{argv[0]}: {exc.strerror or exc}"
            finally:
                if stdin != asyncio.subprocess.DEVNULL:
                    os.close(stdin)
                if not last:
                    os.close(stdout)
                stdin = next_stdin
            if failure is not None:
                break
    finally:
        if stdin != asyncio.subprocess.DEVNULL:
            os.close(stdin)
        os.close(out_write)
    reader = asyncio.ensure_future(asyncio.to_thread(_read_all, out_read))
    timed_out = False
    try:
        raw = await asyncio.wait_for(asyncio.shield(reader), timeout=timeout)
        for process in processes:
            await process.wait()
    except asyncio.TimeoutError:
        timed_out = True
        for process in processes:
            _kill(process)
        for process in processes:
            with _suppress_process_errors():
                await process.wait()
        raw = await reader
    if failure is not None:
        return raw + f"{failure}\n".encode(), 127, timed_out
    return raw, _pipeline_exit_code(processes), timed_out


def _read_all(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    try:
        while chunk := os.read(descriptor, 65536):
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _pipeline_exit_code(processes: list[Any]) -> Optional[int]:
    codes = [process.returncode for process in processes]
    if not codes or any(code is None for code in codes):
        return None
    failing = [code for index, code in enumerate(codes)
               if code != 0 and not (index < len(codes) - 1 and code == -signal.SIGPIPE)]
    return failing[-1] if failing else 0


def _kill(process: Any) -> None:
    """Kill the whole process group on posix, so a pipeline's children go with the shell."""
    with _suppress_process_errors():
        if os.name == "posix":
            os.killpg(os.getpgid(process.pid), 9)
        else:  # pragma: no cover - the harness runs on posix
            process.kill()


class _suppress_process_errors:
    def __enter__(self) -> None:
        return None

    def __exit__(self, kind, value, traceback) -> bool:
        return kind is not None and issubclass(kind, (ProcessLookupError, OSError))
