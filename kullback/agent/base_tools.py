"""The base tools every extension may have: read, write, edit, grep, find, ls, web_search, bash.

Ported from tau_coding's coding tools (read, write, edit, bash: the line slicing and its
continuation hint, the exact-match edit, the head and tail truncation, the shell's captured output)
and bound to a root directory, because an extension of this core owns a directory and nothing
above it. grep, find and ls are ours: tau reaches them through the shell, and an extension whose
shell is an allowlist needs them as tools.

Two levels of boundary, and no third. Cannot: the tool is not registered (`only` decides what an
extension gets, so the Examiner has no write and no bash, and the user has no file tools at all).
Should not: what the prompt says. Between them sits the one mechanical rule the tools enforce
themselves, because a path is an argument and not a registration: every path is relative to the
root, and an absolute path or a step above the root is refused with the rule named in the result.
The shell is the one tool that is many tools, so `bash` takes an `Allowlist`: the first word of
every pipeline segment must be on it, and a segment matching a refused pattern never runs.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import shlex
import time
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

TOOL_NAMES = ("read", "write", "edit", "grep", "find", "ls", "web_search", "bash")


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

    def segments(self, command: str) -> list[list[str]]:
        """The command's pipeline segments as token lists; a shell it cannot parse is one segment."""
        try:
            lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
            lexer.whitespace_split = True
            tokens = list(lexer)
        except ValueError as exc:
            raise CommandRefused(f"rule shell: the command does not parse as a shell command ({exc})") from exc
        segments: list[list[str]] = [[]]
        for token in tokens:
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
            name = next((token for token in segment if "=" not in token.split("/")[0]), segment[0])
            if name not in self.commands:
                allowed = ", ".join(sorted(self.commands)) or "nothing"
                raise CommandRefused(
                    f"rule allowlist: {name!r} is not one of the commands this agent may run ({allowed})"
                )


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


def _read_tool(config: BaseToolConfig) -> AgentTool:
    async def execute(args: ReadArgs) -> ReadResult:
        path = resolve_in_root(config.root, args.path)
        if not path.exists():
            raise PathRefused(f"no file {args.path!r} under this agent's root directory")
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
        process = await asyncio.create_subprocess_shell(
            args.command,
            cwd=str(config.root),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        timed_out = False
        try:
            raw, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            timed_out = True
            raw = b""
            _kill(process)
            with _suppress_process_errors():
                await process.wait()
        output, truncation = truncate_tail(raw.decode(errors="replace"))
        if timed_out:
            output = f"{output}\n[the command was killed after {timeout:g} seconds]".strip()
        return BashResult(
            command=args.command,
            output=output or "(no output)",
            exit_code=process.returncode,
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
