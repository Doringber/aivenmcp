"""Chunking decides what the retriever can find, so its invariants are tested directly."""

from __future__ import annotations

import re

import pytest

from aiven_rag_mcp.chunking import Chunker, breadcrumb, parse_document
from aiven_rag_mcp.models import DocFormat
from tests.support.corpus import KAFKA_FRAGMENT, KAFKA_PAGE, PG_LIMITS, README
from tests.support.fakes import WordCounter

RST_NESTED = """\
=========
Top Title
=========

Intro paragraph.

Section A
---------

Text of A.

Sub A1
~~~~~~

Text of A1.

Section B
---------

Text of B.
"""

MARKDOWN_WITH_CODE = """\
# Guide

Intro.

```bash
# this is a shell comment, not a heading
echo hi
```

## Step 1

Do the first thing.

Setext heading
--------------

Body under setext.
"""


def make_chunker(max_tokens: int = 60, overlap: int = 8) -> Chunker:
    return Chunker(WordCounter(), max_tokens=max_tokens, overlap_tokens=overlap)


def words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text))


class TestParsing:
    def test_rst_levels_follow_first_appearance_including_overline_titles(self) -> None:
        parsed = parse_document(RST_NESTED, "rst")
        assert parsed.title == "Top Title"
        paths = [s.path for s in parsed.sections]
        assert ("Top Title", "Section A", "Sub A1") in paths
        assert ("Top Title", "Section B") in paths

    def test_rst_inline_markup_is_removed_from_headings(self) -> None:
        parsed = parse_document(KAFKA_FRAGMENT, "rst")
        assert [s.path[-1] for s in parsed.sections] == [
            "auto_create_topics_enable",
            "min_insync_replicas",
        ]

    def test_markdown_ignores_hash_lines_inside_code_fences(self) -> None:
        parsed = parse_document(MARKDOWN_WITH_CODE, "markdown")
        headings = [s.path[-1] for s in parsed.sections if s.path]
        assert "this is a shell comment, not a heading" not in headings
        assert headings == ["Guide", "Step 1", "Setext heading"]

    def test_markdown_front_matter_title_wins(self) -> None:
        parsed = parse_document(README, "markdown")
        assert parsed.title == "Contributing guide"
        assert "title:" not in "".join(s.text for s in parsed.sections)

    def test_simple_table_borders_are_not_headings(self) -> None:
        table = "Title\n=====\n\n=====  =====\nA      B\n=====  =====\n1      2\n=====  =====\n"
        parsed = parse_document(table, "rst")
        assert [s.path for s in parsed.sections] == [("Title",)]


class TestChunker:
    @pytest.mark.parametrize(
        ("text", "fmt"),
        [
            (RST_NESTED, "rst"),
            (MARKDOWN_WITH_CODE, "markdown"),
            (KAFKA_PAGE + KAFKA_FRAGMENT, "rst"),
            (PG_LIMITS * 20, "rst"),
            ("word " * 1000, "text"),
            ("\n".join(f"line {i} " + "x " * 30 for i in range(40)), "markdown"),
        ],
    )
    def test_chunks_fit_budget_and_lose_no_content(self, text: str, fmt: DocFormat) -> None:
        chunker = make_chunker()
        chunks = chunker.chunk(text, fmt, "Doc")
        assert chunks
        counter = WordCounter()
        for chunk in chunks:
            assert counter.count(chunk.heading) + counter.count(chunk.content) <= 60 + 2
        heading_words = {w for line in text.splitlines() for w in re.findall(r"\w+", line)}
        assert heading_words <= words(" ".join(c.heading + " " + c.content for c in chunks))

    def test_small_sibling_sections_are_packed_into_one_chunk(self) -> None:
        chunks = make_chunker(max_tokens=200).chunk(KAFKA_FRAGMENT, "rst", "Kafka params")
        assert len(chunks) == 1
        assert "auto_create_topics_enable" in chunks[0].content
        assert "min_insync_replicas" in chunks[0].content
        assert chunks[0].heading == "Kafka params"

    def test_small_document_becomes_a_single_chunk(self) -> None:
        chunks = make_chunker(max_tokens=200).chunk(RST_NESTED, "rst", "Top Title")
        assert len(chunks) == 1

    def test_subsection_is_not_merged_with_the_next_branch(self) -> None:
        text = (
            "Title\n=====\n\nAlpha\n-----\n\n"
            + "atext " * 30
            + "\n\nAlpha one\n~~~~~~~~~\n\n"
            + "aonetext " * 30
            + "\n\nBravo\n-----\n\nbtext once\n"
        )
        chunks = make_chunker(max_tokens=50).chunk(text, "rst", "Title")
        assert not any("aonetext" in c.content and "btext" in c.content for c in chunks)
        assert chunks[-1].heading == "Title > Bravo"

    def test_heading_only_sections_do_not_become_chunks(self) -> None:
        text = "Title\n=====\n\nSection\n-------\n\n" + "body " * 80
        chunks = make_chunker(max_tokens=60).chunk(text, "rst", "Title")
        assert all(c.content.strip() != "Title" for c in chunks)

    def test_continuation_chunks_keep_section_breadcrumb_and_overlap(self) -> None:
        paragraphs = "\n\n".join(f"Paragraph {i} " + "filler " * 15 for i in range(10))
        text = f"Title\n=====\n\nLong section\n------------\n\n{paragraphs}\n"
        chunks = make_chunker(max_tokens=60, overlap=20).chunk(text, "rst", "Title")
        assert len(chunks) > 2
        assert all(c.heading == "Title > Long section" for c in chunks[1:])
        first_last = chunks[0].content.split("\n\n")[-1]
        assert chunks[1].content.startswith(first_last)

    def test_ordinals_are_sequential(self) -> None:
        chunks = make_chunker().chunk(PG_LIMITS * 10, "rst", "PG")
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))

    def test_embedding_text_contains_breadcrumb(self) -> None:
        chunk = make_chunker().chunk(RST_NESTED, "rst", "Top Title")[-1]
        assert chunk.embedding_text.startswith(chunk.heading)

    def test_overlap_must_be_smaller_than_half_the_budget(self) -> None:
        with pytest.raises(ValueError):
            Chunker(WordCounter(), max_tokens=100, overlap_tokens=60)


@pytest.mark.parametrize(
    ("title", "path", "expected"),
    [
        ("Doc", (), "Doc"),
        ("Doc", ("Doc", "A"), "Doc > A"),
        ("Doc", ("A", "B"), "Doc > A > B"),
        ("", ("A",), "A"),
    ],
)
def test_breadcrumb(title: str, path: tuple[str, ...], expected: str) -> None:
    assert breadcrumb(title, path) == expected
