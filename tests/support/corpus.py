"""A miniature docs repository shaped like aiven/devportal (RST pages, includes, Markdown)."""

from __future__ import annotations

from pathlib import Path

KAFKA_PAGE = """\
Advanced parameters for Aiven for Apache Kafka
==============================================

Below you can find a summary of every configuration option:

.. include:: /includes/config-kafka.rst
"""

KAFKA_FRAGMENT = """\
``auto_create_topics_enable``
-----------------------------
*boolean*

**auto.create.topics.enable** Enable auto creation of topics

``min_insync_replicas``
-----------------------
*integer*

**min.insync.replicas** Minimum number of replicas that must acknowledge a write.
"""

PG_LIMITS = """\
Connection limits per plan for Aiven for PostgreSQL
===================================================

The ``max_connections`` setting varies according to the service plan.

Hobbyist plans allow 25 connections. Startup-8 plans allow 200 connections.
"""

README = """\
---
title: Contributing guide
---

# Contributing

Run the linter before opening a pull request.
"""


def write_corpus(root: Path) -> Path:
    files = {
        "docs/products/kafka/reference/advanced-params.rst": KAFKA_PAGE,
        "includes/config-kafka.rst": KAFKA_FRAGMENT,
        "docs/products/postgresql/reference/pg-connection-limits.rst": PG_LIMITS,
        "README.md": README,
        "node_modules/pkg/README.md": "# Should be skipped\n",
        ".github/PULL_REQUEST_TEMPLATE.md": "# Should be skipped\n",
        "images/logo.png": "not text",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root
