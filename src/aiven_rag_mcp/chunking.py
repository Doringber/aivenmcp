"""Structure-aware chunking for Markdown and reStructuredText.

Documentation answers usually live in a single section ("what does parameter X
do", "step 3 of the how-to"), so we split on headings first and only fall back
to paragraph/line/word splitting for sections that exceed the token budget.
Small sibling sections (for example one section per config parameter) are packed
together so we do not create hundreds of near-empty chunks.

Every chunk gets a breadcrumb ``Title > Section > Subsection`` that is embedded
together with the text, which gives short sections the context they lack.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Protocol

from aiven_rag_mcp.models import Chunk, DocFormat

BREADCRUMB_SEPARATOR = " > "
_PARAGRAPH_SPLIT = re.compile(r"\n[ \t]*\n")
_MD_ATX = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_MD_FENCE = re.compile(r"^[ \t]{0,3}(```+|~~~+)")
_MD_SETEXT = re.compile(r"^[ \t]{0,3}(=+|-+)[ \t]*$")
_MD_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_FRONT_MATTER_TITLE = re.compile(r"^title:\s*[\"']?(.+?)[\"']?\s*$", re.MULTILINE)
_RST_ADORNMENT_CHARS = set("=-`:'\"~^_*+#<>.")
_RST_LABEL = re.compile(r"^\.\.\s+_[^:]+:\s*$")
_INLINE_MARKUP = re.compile(r"[`*]+")
_RST_ROLE = re.compile(r":[a-z:+-]+:`([^`<]*?)(?:\s*<[^>]*>)?`")


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


@dataclass(frozen=True)
class Section:
    path: tuple[str, ...]
    heading_line: str
    body: str

    @property
    def text(self) -> str:
        parts = [self.heading_line, self.body.strip()]
        return "\n\n".join(p for p in parts if p).strip()


@dataclass(frozen=True)
class ParsedDocument:
    title: Optional[str]
    sections: list[Section]


def clean_heading(raw: str) -> str:
    text = _RST_ROLE.sub(r"\1", raw)
    text = _INLINE_MARKUP.sub("", text)
    return " ".join(text.split())


def parse_document(text: str, fmt: DocFormat) -> ParsedDocument:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    match fmt:
        case "markdown":
            return _parse_markdown(text)
        case "rst":
            return _parse_rst(text)
        case _:
            return ParsedDocument(title=None, sections=[Section((), "", text)])


class _SectionBuilder:
    """Accumulates body lines and maintains the heading stack."""

    def __init__(self) -> None:
        self.sections: list[Section] = []
        self._stack: list[tuple[int, str]] = []
        self._heading_line = ""
        self._lines: list[str] = []

    def add_line(self, line: str) -> None:
        self._lines.append(line)

    def pop_last_line(self) -> None:
        if self._lines:
            self._lines.pop()

    def start_section(self, level: int, title: str, heading_line: str) -> None:
        self._flush()
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        self._stack.append((level, title))
        self._heading_line = heading_line

    def finish(self) -> list[Section]:
        self._flush()
        return self.sections

    def _flush(self) -> None:
        body = "\n".join(line.rstrip() for line in self._lines).strip("\n")
        body = re.sub(r"\n{3,}", "\n\n", body)
        if body.strip() or self._heading_line:
            path = tuple(title for _, title in self._stack)
            self.sections.append(Section(path, self._heading_line, body))
        self._lines = []
        self._heading_line = ""


def _parse_markdown(text: str) -> ParsedDocument:
    title: Optional[str] = None
    front = _MD_FRONT_MATTER.match(text)
    if front:
        title_match = _FRONT_MATTER_TITLE.search(front.group(1))
        if title_match:
            title = clean_heading(title_match.group(1))
        text = text[front.end() :]

    builder = _SectionBuilder()
    lines = text.split("\n")
    in_fence: Optional[str] = None
    first_heading_level: Optional[int] = None
    i = 0
    while i < len(lines):
        line = lines[i]
        fence = _MD_FENCE.match(line)
        if fence:
            marker = fence.group(1)[0] * 3
            if in_fence is None:
                in_fence = marker
            elif marker == in_fence:
                in_fence = None
            builder.add_line(line)
            i += 1
            continue
        if in_fence is None:
            heading = _match_markdown_heading(lines, i)
            if heading is not None:
                level, raw_title, consumed = heading
                heading_title = clean_heading(raw_title)
                if first_heading_level is None:
                    first_heading_level = level
                    if title is None and level == 1:
                        title = heading_title
                builder.start_section(level, heading_title, "#" * level + " " + heading_title)
                i += consumed
                continue
        builder.add_line(line)
        i += 1
    return ParsedDocument(title=title, sections=builder.finish())


def _match_markdown_heading(lines: Sequence[str], i: int) -> Optional[tuple[int, str, int]]:
    line = lines[i]
    atx = _MD_ATX.match(line)
    if atx:
        return len(atx.group(1)), atx.group(2), 1
    if i + 1 < len(lines) and line.strip() and not line.startswith((" ", "\t", "|", "-", "*", ">")):
        setext = _MD_SETEXT.match(lines[i + 1])
        if setext and (i == 0 or not lines[i - 1].strip()):
            return (1 if setext.group(1)[0] == "=" else 2), line.strip(), 2
    return None


def _is_rst_adornment(line: str) -> bool:
    stripped = line.rstrip()
    return (
        len(stripped) >= 3
        and stripped[0] in _RST_ADORNMENT_CHARS
        and stripped == stripped[0] * len(stripped)
    )


def _parse_rst(text: str) -> ParsedDocument:
    builder = _SectionBuilder()
    lines = text.split("\n")
    styles: list[tuple[str, bool]] = []
    title: Optional[str] = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if _RST_LABEL.match(line):
            i += 1
            continue
        is_heading = (
            i + 1 < len(lines)
            and line.strip()
            and not line[0].isspace()
            and not _is_rst_adornment(line)
            and _is_rst_adornment(lines[i + 1])
            and len(lines[i + 1].rstrip()) >= len(line.rstrip()) - 2
        )
        if is_heading:
            char = lines[i + 1][0]
            overline = i > 0 and _is_rst_adornment(lines[i - 1]) and lines[i - 1][0] == char
            if overline:
                builder.pop_last_line()
            style = (char, overline)
            if style not in styles:
                styles.append(style)
            level = styles.index(style)
            heading_title = clean_heading(line)
            if title is None and level == 0:
                title = heading_title
            builder.start_section(level + 1, heading_title, heading_title)
            i += 2
            continue
        builder.add_line(line)
        i += 1
    return ParsedDocument(title=title, sections=builder.finish())


def breadcrumb(title: str, path: Sequence[str]) -> str:
    parts = list(path)
    if parts and title and parts[0] == title:
        parts = parts[1:]
    return BREADCRUMB_SEPARATOR.join([p for p in [title, *parts] if p])


def _common_prefix(paths: Sequence[tuple[str, ...]]) -> tuple[str, ...]:
    prefix = paths[0]
    for path in paths[1:]:
        n = 0
        while n < min(len(prefix), len(path)) and prefix[n] == path[n]:
            n += 1
        prefix = prefix[:n]
    return prefix


class Chunker:
    """Turns a parsed document into chunks that fit the embedding model."""

    def __init__(self, counter: TokenCounter, max_tokens: int, overlap_tokens: int) -> None:
        if overlap_tokens >= max_tokens // 2:
            raise ValueError("overlap_tokens must be smaller than half of max_tokens")
        self._counter = counter
        self._max = max_tokens
        self._overlap = overlap_tokens

    def chunk(self, text: str, fmt: DocFormat, title: str) -> list[Chunk]:
        parsed = parse_document(text, fmt)
        out: list[tuple[str, str, int]] = []
        group: list[tuple[Section, int]] = []

        def flush() -> None:
            if not group:
                return
            if not any(s.body.strip() for s, _ in group):
                # A heading with no body (e.g. a page title right before its first section)
                # is already carried by the breadcrumbs of the sections below it.
                group.clear()
                return
            heading = breadcrumb(title, _common_prefix([s.path for s, _ in group]))
            content = "\n\n".join(s.text for s, _ in group)
            tokens = sum(t for _, t in group) + self._counter.count(heading)
            out.append((heading, content, tokens))
            group.clear()

        for section in parsed.sections:
            section_text = section.text
            if not section_text:
                continue
            tokens = self._counter.count(section_text)
            if group and self._fits_group(group, section, tokens, title):
                group.append((section, tokens))
                continue
            flush()
            heading = breadcrumb(title, section.path)
            budget = self._max - self._counter.count(heading)
            if tokens <= budget:
                group.append((section, tokens))
            else:
                for piece, piece_tokens in self._split(section_text, budget):
                    out.append((heading, piece, piece_tokens + self._counter.count(heading)))
        flush()
        return [
            Chunk(ordinal=i, heading=h, content=c, token_count=t) for i, (h, c, t) in enumerate(out)
        ]

    def _fits_group(
        self, group: list[tuple[Section, int]], section: Section, tokens: int, title: str
    ) -> bool:
        anchor = group[0][0].path
        if section.path[: max(len(anchor) - 1, 0)] != anchor[: max(len(anchor) - 1, 0)]:
            return False
        heading = breadcrumb(title, _common_prefix([*(s.path for s, _ in group), section.path]))
        used = sum(t for _, t in group) + len(group)
        return used + tokens + self._counter.count(heading) <= self._max

    def _split(self, text: str, budget: int) -> list[tuple[str, int]]:
        units: list[tuple[str, int]] = []
        for paragraph in _PARAGRAPH_SPLIT.split(text):
            paragraph = paragraph.strip("\n")
            if not paragraph.strip():
                continue
            tokens = self._counter.count(paragraph)
            if tokens <= budget:
                units.append((paragraph, tokens))
            else:
                units.extend(self._split_oversized(paragraph, budget))

        pieces: list[tuple[str, int]] = []
        current: list[tuple[str, int]] = []
        for unit in units:
            used = sum(t for _, t in current) + len(current)
            if current and used + unit[1] > budget:
                pieces.append(self._join(current))
                current = self._overlap_tail(current, budget - unit[1])
            current.append(unit)
        if current:
            pieces.append(self._join(current))
        return pieces

    def _overlap_tail(self, units: list[tuple[str, int]], room: int) -> list[tuple[str, int]]:
        tail: list[tuple[str, int]] = []
        total = 0
        for unit in reversed(units):
            if total + unit[1] > min(self._overlap, room):
                break
            tail.insert(0, unit)
            total += unit[1] + 1
        return tail

    def _split_oversized(self, paragraph: str, budget: int) -> list[tuple[str, int]]:
        """Split a paragraph that exceeds the budget on line breaks, then on words."""
        out: list[tuple[str, int]] = []
        buffer: list[str] = []

        def flush() -> None:
            if buffer:
                text = "\n".join(buffer)
                out.append((text, self._counter.count(text)))
                buffer.clear()

        for line in paragraph.split("\n"):
            if self._counter.count(line) > budget:
                flush()
                out.extend(self._split_words(line, budget))
                continue
            if buffer and self._counter.count("\n".join([*buffer, line])) > budget:
                flush()
            buffer.append(line)
        flush()
        return out

    def _split_words(self, line: str, budget: int) -> list[tuple[str, int]]:
        words = line.split()
        out: list[tuple[str, int]] = []
        start = 0
        while start < len(words):
            end = len(words)
            while end - start > 1 and self._counter.count(" ".join(words[start:end])) > budget:
                end = start + max(1, (end - start) * 3 // 4)
            piece = " ".join(words[start:end])
            out.append((piece, self._counter.count(piece)))
            start = end
        return out

    @staticmethod
    def _join(units: list[tuple[str, int]]) -> tuple[str, int]:
        return "\n\n".join(u for u, _ in units), sum(t for _, t in units) + len(units) - 1
