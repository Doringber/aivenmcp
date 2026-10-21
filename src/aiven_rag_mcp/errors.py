"""Exception hierarchy. Every error raised on purpose derives from ``RagError``."""

from __future__ import annotations


class RagError(Exception):
    """Base class for errors that should be reported to the MCP client as-is."""


class ConfigurationError(RagError):
    """The server is misconfigured (bad env var, model/database mismatch...)."""


class InvalidInputError(RagError):
    """The caller passed arguments that cannot be processed."""


class SourceNotFoundError(RagError):
    """A file, directory, URL or stored source does not exist."""


class FetchError(RagError):
    """Downloading a URL or cloning a repository failed."""


class StorageError(RagError):
    """The database is unreachable or rejected an operation."""
