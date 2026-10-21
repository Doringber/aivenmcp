"""Loaders decide what ends up in the index and under which identity (URI, title, URL)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest

from aiven_rag_mcp.config import Settings
from aiven_rag_mcp.errors import InvalidInputError, SourceNotFoundError
from aiven_rag_mcp.loaders import (
    DocumentLoader,
    _github_blob_to_raw,
    _parse_git_url,
    expand_rst_includes,
    html_to_markdown,
)


@pytest.fixture
def loader() -> DocumentLoader:
    return DocumentLoader(Settings())


def test_directory_walk_expands_includes_and_skips_rendered_fragments(
    loader: DocumentLoader, corpus: Path
) -> None:
    loaded = loader.from_path(str(corpus))
    documents = {Path(d.uri).relative_to(corpus).as_posix(): d for d in loaded.documents}

    assert set(documents) == {
        "README.md",
        "docs/products/kafka/reference/advanced-params.rst",
        "docs/products/postgresql/reference/pg-connection-limits.rst",
    }
    kafka = documents["docs/products/kafka/reference/advanced-params.rst"]
    assert "auto_create_topics_enable" in kafka.text
    assert kafka.metadata["includes"] == ["includes/config-kafka.rst"]
    assert kafka.title == "Advanced parameters for Aiven for Apache Kafka"
    assert loaded.complete_collection is True


def test_includes_cannot_escape_the_root(tmp_path: Path) -> None:
    (tmp_path / "outside.rst").write_text("secret\n")
    root = tmp_path / "root"
    root.mkdir()
    page = root / "page.rst"
    page.write_text(".. include:: ../outside.rst\n")
    text, included = expand_rst_includes(page.read_text(), page, root)
    assert "secret" not in text
    assert included == []


def test_single_file_keeps_its_title_and_format(loader: DocumentLoader, corpus: Path) -> None:
    loaded = loader.from_path(str(corpus / "README.md"))
    (document,) = list(loaded.documents)
    assert document.title == "Contributing guide"
    assert document.format == "markdown"
    assert loaded.complete_collection is False


def test_text_input_gets_a_stable_content_addressed_uri(loader: DocumentLoader) -> None:
    first = next(loader.from_text("# Note\n\nhello").documents)
    second = next(loader.from_text("# Note\n\nhello").documents)
    assert first.uri == second.uri
    assert first.uri.startswith("text://")
    assert first.title == "Note"


def test_text_input_with_source_id_uses_it(loader: DocumentLoader) -> None:
    document = next(loader.from_text("hello", source_id="notes/1", title="T").documents)
    assert (document.uri, document.title) == ("notes/1", "T")


def test_empty_text_is_rejected(loader: DocumentLoader) -> None:
    with pytest.raises(InvalidInputError):
        loader.from_text("   ")


def test_missing_path_is_reported(loader: DocumentLoader, tmp_path: Path) -> None:
    with pytest.raises(SourceNotFoundError):
        loader.from_path(str(tmp_path / "nope"))


def test_allowed_roots_are_enforced(corpus: Path, tmp_path: Path) -> None:
    restricted = DocumentLoader(Settings(ingest_allowed_paths=[corpus / "docs"]))
    restricted.from_path(str(corpus / "docs"))
    with pytest.raises(InvalidInputError, match="outside the allowed"):
        restricted.from_path(str(corpus / "README.md"))


def test_allowed_roots_parse_from_colon_separated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_INGEST_ALLOWED_PATHS", "/data:/srv/docs")
    assert Settings().ingest_allowed_paths == [Path("/data"), Path("/srv/docs")]


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://github.com/aiven/devportal", ("github.com/aiven/devportal", None, None)),
        ("https://github.com/aiven/devportal.git", ("github.com/aiven/devportal", None, None)),
        (
            "https://github.com/aiven/devportal/tree/main/docs/products",
            ("github.com/aiven/devportal", "main", "docs/products"),
        ),
        ("https://gitlab.com/group/project.git", ("gitlab.com/group/project", None, None)),
        ("https://github.com/aiven/devportal/blob/main/README.md", None),
        ("https://docs.aiven.io/docs/products/postgresql", None),
    ],
)
def test_git_url_detection(
    url: str, expected: Optional[tuple[str, Optional[str], Optional[str]]]
) -> None:
    repo = _parse_git_url(url)
    assert (None if repo is None else (repo.name, repo.ref, repo.subpath)) == expected


def test_github_blob_urls_are_fetched_raw() -> None:
    assert (
        _github_blob_to_raw("https://github.com/aiven/devportal/blob/main/docs/index.rst")
        == "https://raw.githubusercontent.com/aiven/devportal/main/docs/index.rst"
    )


def test_html_conversion_keeps_headings_and_drops_chrome() -> None:
    html = """<html><head><title>Page</title><style>p{}</style></head><body>
    <nav>Menu Home About</nav>
    <h1>Main heading</h1><p>First <b>para</b>.</p>
    <h2>Sub</h2><ul><li>one</li><li>two</li></ul>
    <pre>code  line</pre><script>alert(1)</script></body></html>"""
    title, text = html_to_markdown(html)
    assert title == "Page"
    assert "# Main heading" in text
    assert "## Sub" in text
    assert "- one" in text
    assert "code  line" in text
    assert "Menu" not in text
    assert "alert" not in text
