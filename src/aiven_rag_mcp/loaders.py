"""Turn ingest inputs (raw text, local path, URL, git repository) into documents."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlparse

import httpx

from aiven_rag_mcp.chunking import parse_document
from aiven_rag_mcp.config import Settings
from aiven_rag_mcp.errors import FetchError, InvalidInputError, SourceNotFoundError
from aiven_rag_mcp.models import DocFormat, RawDocument

logger = logging.getLogger(__name__)

EXTENSION_FORMATS: dict[str, DocFormat] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".mdx": "markdown",
    ".rst": "rst",
    ".txt": "text",
}
SKIP_DIRS = frozenset(
    {".git", ".github", "node_modules", "_build", "build", ".venv", "venv", "__pycache__", ".tox"}
)
GIT_CLONE_TIMEOUT_SECONDS = 600
MAX_INCLUDE_DEPTH = 3
_GITHUB_HOSTS = frozenset({"github.com", "www.github.com"})
_RST_INCLUDE = re.compile(
    r"^(?P<indent>[ \t]*)\.\.[ \t]+include::[ \t]+(?P<target>\S+)[ \t]*$", re.MULTILINE
)


@dataclass
class LoadedSource:
    """Documents from one ingest request. ``documents`` is consumed lazily."""

    collection: str
    documents: Iterator[RawDocument]
    complete_collection: bool
    """True when the request enumerates a whole collection, so absent sources can be pruned."""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _decode(data: bytes) -> Optional[str]:
    if b"\x00" in data[:8192]:
        return None
    return data.decode("utf-8", errors="replace")


def _title_for(text: str, fmt: DocFormat, fallback: str) -> str:
    parsed_title = parse_document(text, fmt).title
    return parsed_title or fallback


def _resolve_include(target: str, including_file: Path, root: Path) -> Optional[Path]:
    """Resolve an RST include target the way Sphinx does: '/x' is relative to the docs root."""
    base = root / target.lstrip("/") if target.startswith("/") else including_file.parent / target
    candidate = base.resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        return None
    return candidate if candidate.is_file() else None


def expand_rst_includes(
    text: str, path: Path, root: Path, depth: int = 0
) -> tuple[str, list[Path]]:
    """Inline ``.. include::`` directives so included content inherits the page's headings.

    Pages such as "Advanced parameters for Aiven for Apache Kafka" consist of a single include
    of a generated fragment; without expansion the fragment has no title at all.
    """
    included: list[Path] = []

    def replace(match: re.Match[str]) -> str:
        target = _resolve_include(match.group("target"), path, root)
        if target is None or depth >= MAX_INCLUDE_DEPTH:
            return match.group(0)
        body = _decode(target.read_bytes()) or ""
        body, nested = expand_rst_includes(body, target, root, depth + 1)
        included.extend([target, *nested])
        indent = match.group("indent")
        return "\n".join(indent + line if line else line for line in body.split("\n"))

    return _RST_INCLUDE.sub(replace, text), included


def _include_targets(files: list[Path], root: Path) -> set[Path]:
    targets: set[Path] = set()
    for file_path in files:
        if file_path.suffix.lower() != ".rst":
            continue
        text = _decode(file_path.read_bytes()) or ""
        for match in _RST_INCLUDE.finditer(text):
            target = _resolve_include(match.group("target"), file_path, root)
            if target is not None:
                targets.add(target)
    return targets


class DocumentLoader:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def from_text(
        self,
        text: str,
        *,
        title: Optional[str] = None,
        source_id: Optional[str] = None,
        collection: Optional[str] = None,
        fmt: DocFormat = "markdown",
    ) -> LoadedSource:
        if not text.strip():
            raise InvalidInputError("'text' is empty.")
        uri = source_id or f"text://{_sha(text)[:16]}"
        resolved_title = title or _title_for(text, fmt, uri)
        document = RawDocument(
            uri=uri,
            collection=collection or "inline",
            kind="text",
            title=resolved_title,
            text=text,
            format=fmt,
        )
        return LoadedSource(document.collection, iter([document]), complete_collection=False)

    def from_path(
        self, raw_path: str, *, collection: Optional[str] = None, title: Optional[str] = None
    ) -> LoadedSource:
        path = Path(raw_path).expanduser().resolve()
        self._check_allowed(path)
        if not path.exists():
            raise SourceNotFoundError(f"Path does not exist inside the server: {path}")
        if path.is_file():
            document = self._read_file(
                path, root=path.parent, collection=collection or str(path.parent)
            )
            if document is None:
                raise InvalidInputError(f"Unsupported or binary file: {path}")
            if title:
                document.title = title
            return LoadedSource(document.collection, iter([document]), complete_collection=False)
        resolved_collection = collection or str(path)
        documents = self._walk(path, path, resolved_collection, web_base=None)
        return LoadedSource(resolved_collection, documents, complete_collection=True)

    @contextmanager
    def from_url(
        self, url: str, *, collection: Optional[str] = None, title: Optional[str] = None
    ) -> Iterator[LoadedSource]:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise InvalidInputError("Only http(s) URLs are supported.")
        repo = _parse_git_url(url)
        if repo is not None:
            with self._clone(repo) as (checkout, ref):
                root = checkout / repo.subpath if repo.subpath else checkout
                if not root.exists():
                    raise SourceNotFoundError(f"'{repo.subpath}' does not exist in {repo.name}")
                resolved_collection = collection or repo.name
                web_base = (
                    f"https://github.com/{repo.owner}/{repo.repo}/blob/{ref}/"
                    if repo.owner
                    else None
                )
                documents = self._walk(
                    root, checkout, resolved_collection, web_base=web_base, uri_prefix=repo.name
                )
                yield LoadedSource(resolved_collection, documents, complete_collection=True)
            return
        document = self._fetch(_github_blob_to_raw(url), original_url=url)
        if title:
            document.title = title
        if collection:
            document.collection = collection
        yield LoadedSource(document.collection, iter([document]), complete_collection=False)

    def _check_allowed(self, path: Path) -> None:
        roots = [r.expanduser().resolve() for r in self._settings.ingest_allowed_paths]
        if roots and not any(path == root or root in path.parents for root in roots):
            allowed = ", ".join(str(r) for r in roots)
            raise InvalidInputError(f"Path {path} is outside the allowed ingest roots: {allowed}")

    def _walk(
        self,
        directory: Path,
        root: Path,
        collection: str,
        *,
        web_base: Optional[str],
        uri_prefix: Optional[str] = None,
    ) -> Iterator[RawDocument]:
        files: list[Path] = []
        for current, dirnames, filenames in os.walk(directory):
            dirnames[:] = sorted(
                d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")
            )
            files.extend(
                Path(current) / name
                for name in sorted(filenames)
                if Path(name).suffix.lower() in EXTENSION_FORMATS
            )
        # Fragments are indexed as part of the page that includes them, not twice.
        rendered_elsewhere = _include_targets(files, root)
        for file_path in files:
            if file_path.resolve() in rendered_elsewhere:
                continue
            document = self._read_file(
                file_path,
                root=root,
                collection=collection,
                web_base=web_base,
                uri_prefix=uri_prefix,
                expand_includes=True,
            )
            if document is not None:
                yield document

    def _read_file(
        self,
        path: Path,
        *,
        root: Path,
        collection: str,
        web_base: Optional[str] = None,
        uri_prefix: Optional[str] = None,
        expand_includes: bool = False,
    ) -> Optional[RawDocument]:
        fmt = EXTENSION_FORMATS.get(path.suffix.lower())
        if fmt is None:
            return None
        if path.stat().st_size > self._settings.max_download_bytes:
            logger.warning(
                "Skipping %s: larger than %d bytes", path, self._settings.max_download_bytes
            )
            return None
        text = _decode(path.read_bytes())
        if text is None:
            return None
        metadata: dict[str, object] = {}
        if expand_includes and fmt == "rst":
            text, included = expand_rst_includes(text, path, root)
            if included:
                resolved_root = root.resolve()
                metadata["includes"] = sorted(
                    {p.relative_to(resolved_root).as_posix() for p in included}
                )
        relative = path.relative_to(root).as_posix()
        metadata["path"] = relative
        uri = f"{uri_prefix}/{relative}" if uri_prefix else str(path)
        return RawDocument(
            uri=uri,
            collection=collection,
            kind="git" if uri_prefix else "file",
            title=_title_for(text, fmt, relative.rsplit(".", 1)[0]),
            text=text,
            format=fmt,
            web_url=f"{web_base}{relative}" if web_base else None,
            metadata=metadata,
        )

    def _fetch(self, url: str, *, original_url: str) -> RawDocument:
        limit = self._settings.max_download_bytes
        try:
            with (
                httpx.Client(
                    follow_redirects=True,
                    timeout=self._settings.http_timeout_seconds,
                    headers={"User-Agent": "aiven-rag-mcp/0.1"},
                ) as client,
                client.stream("GET", url) as response,
            ):
                response.raise_for_status()
                content_type = (
                    response.headers.get("content-type", "").split(";")[0].strip().lower()
                )
                body = bytearray()
                for block in response.iter_bytes():
                    body.extend(block)
                    if len(body) > limit:
                        raise FetchError(f"{original_url} is larger than {limit} bytes")
        except httpx.HTTPStatusError as exc:
            raise FetchError(
                f"GET {original_url} returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise FetchError(f"GET {original_url} failed: {exc}") from exc

        text = _decode(bytes(body))
        if text is None:
            raise InvalidInputError(f"{original_url} returned binary content ({content_type})")
        detected = _format_for_url(url, content_type)
        page_title: Optional[str] = None
        fmt: DocFormat
        if detected == "html":
            page_title, text = html_to_markdown(text)
            fmt = "markdown"
        else:
            fmt = detected
        return RawDocument(
            uri=original_url,
            collection=urlparse(original_url).netloc,
            kind="url",
            title=page_title or _title_for(text, fmt, original_url),
            text=text,
            format=fmt,
            web_url=original_url,
            metadata={"content_type": content_type},
        )

    @contextmanager
    def _clone(self, repo: _GitRepo) -> Iterator[tuple[Path, str]]:
        with tempfile.TemporaryDirectory(prefix="rag-git-") as tmp:
            command = ["git", "clone", "--depth", "1", "--quiet"]
            if repo.ref:
                command += ["--branch", repo.ref]
            command += [repo.clone_url, tmp]
            logger.info("Cloning %s", repo.clone_url)
            try:
                subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=GIT_CLONE_TIMEOUT_SECONDS,
                    env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
                )
                ref = (
                    repo.ref
                    or subprocess.run(
                        ["git", "-C", tmp, "rev-parse", "--abbrev-ref", "HEAD"],
                        check=True,
                        capture_output=True,
                        text=True,
                    ).stdout.strip()
                )
            except FileNotFoundError as exc:
                raise FetchError("git is not installed in the server environment") from exc
            except subprocess.TimeoutExpired as exc:
                raise FetchError(f"Cloning {repo.clone_url} timed out") from exc
            except subprocess.CalledProcessError as exc:
                raise FetchError(f"Cloning {repo.clone_url} failed: {exc.stderr.strip()}") from exc
            yield Path(tmp), ref


@dataclass(frozen=True)
class _GitRepo:
    clone_url: str
    name: str
    owner: Optional[str]
    repo: Optional[str]
    ref: Optional[str]
    subpath: Optional[str]


def _parse_git_url(url: str) -> Optional[_GitRepo]:
    parsed = urlparse(url)
    segments = [s for s in parsed.path.split("/") if s]
    if parsed.netloc.lower() in _GITHUB_HOSTS and len(segments) >= 2:
        owner, repo = segments[0], segments[1].removesuffix(".git")
        name = f"github.com/{owner}/{repo}"
        clone_url = f"https://github.com/{owner}/{repo}.git"
        if len(segments) == 2:
            return _GitRepo(clone_url, name, owner, repo, None, None)
        if segments[2] == "tree" and len(segments) >= 4:
            subpath = "/".join(segments[4:]) or None
            return _GitRepo(clone_url, name, owner, repo, segments[3], subpath)
        return None
    if parsed.path.endswith(".git"):
        name = f"{parsed.netloc}{parsed.path.removesuffix('.git')}"
        return _GitRepo(url, name, None, None, None, None)
    return None


def _github_blob_to_raw(url: str) -> str:
    parsed = urlparse(url)
    segments = [s for s in parsed.path.split("/") if s]
    if parsed.netloc.lower() in _GITHUB_HOSTS and len(segments) >= 5 and segments[2] == "blob":
        owner, repo, ref, path = segments[0], segments[1], segments[3], "/".join(segments[4:])
        return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
    return url


def _format_for_url(url: str, content_type: str) -> DocFormat | Literal["html"]:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in EXTENSION_FORMATS:
        return EXTENSION_FORMATS[suffix]
    match content_type:
        case "text/html" | "application/xhtml+xml":
            return "html"
        case "text/markdown" | "text/x-markdown":
            return "markdown"
        case "text/x-rst":
            return "rst"
        case "text/plain" | "":
            return "text"
        case _:
            raise InvalidInputError(f"Unsupported content type '{content_type}' for {url}")


class _HtmlToMarkdown(HTMLParser):
    _SKIP = frozenset(
        {"script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "form"}
    )
    _BLOCK = frozenset(
        {"p", "div", "section", "article", "main", "table", "tr", "ul", "ol", "dl", "br", "hr"}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title: Optional[str] = None
        self._skip_depth = 0
        self._in_title = False
        self._in_pre = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif re.fullmatch(r"h[1-6]", tag):
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag == "pre":
            self._in_pre = True
            self.parts.append("\n\n```\n")
        elif tag in {"td", "th"}:
            self.parts.append(" | ")
        elif tag in self._BLOCK:
            self.parts.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag == "title":
            self._in_title = False
        elif tag == "pre":
            self._in_pre = False
            self.parts.append("\n```\n\n")
        elif re.fullmatch(r"h[1-6]", tag) or tag in self._BLOCK:
            self.parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title = (self.title or "") + data.strip()
        elif self._skip_depth == 0:
            self.parts.append(data if self._in_pre else re.sub(r"\s+", " ", data))


def html_to_markdown(html: str) -> tuple[Optional[str], str]:
    """Minimal HTML to Markdown conversion that keeps headings for the chunker."""
    parser = _HtmlToMarkdown()
    parser.feed(html)
    text = "".join(parser.parts)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return parser.title or None, text
