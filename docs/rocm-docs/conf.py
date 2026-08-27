# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Configuration file for the Sphinx documentation builder.
# pylint: disable=invalid-name

import logging


class _SuppressKnownUnreleasedRocmDocsCoreWarnings(logging.Filter):
    """rocm-cli isn't registered in rocm-docs-core's shared projects.yaml
    catalog yet, so `external_projects_current_project` can never resolve
    until that's added upstream. Separately, the "rocm-ai" flavor used here
    was only added to rocm-docs-core after its last release, so the
    published package doesn't recognize it yet and falls back to "rocm".
    Both are purely informational until rocm-docs-core catches up, and
    would otherwise permanently block a `-W` build."""

    _IGNORED_SUBSTRINGS = (
        "not found in projects",
        'Unsupported theme flavor "rocm-ai"',
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(s in message for s in self._IGNORED_SUBSTRINGS)


_suppress_known_unreleased_rocm_docs_core_warnings = (
    _SuppressKnownUnreleasedRocmDocsCoreWarnings()
)
for _logger_name in ("sphinx.rocm_docs.projects", "sphinx.rocm_docs.theme"):
    logging.getLogger(_logger_name).addFilter(
        _suppress_known_unreleased_rocm_docs_core_warnings
    )

# -- Project information ---------------------------------------------------

project = "ROCm CLI"
author = "Advanced Micro Devices, Inc."
# pylint: disable=redefined-builtin
copyright = "Copyright (c) Advanced Micro Devices, Inc. All rights reserved."
# pylint: enable=redefined-builtin

# -- General configuration -------------------------------------------------

html_theme = "rocm_docs_theme"
html_theme_options = {
    "flavor": "rocm-ai",
    "use_repository_button": True,
    "use_download_button": True,
    # Pin these instead of letting rocm-docs-core infer them from the local
    # git branch: CI's pull_request checkout leaves a detached HEAD on the
    # synthetic merge ref, which resolves to an empty repository_url and
    # crashes sphinx_book_theme's repository button.
    "repository_url": "https://github.com/ROCm/rocm-cli",
    "repository_branch": "main",
}
html_title = "ROCm CLI documentation"
# myst.header: `:start-after:` on included README sections strips the heading
# line itself, so the first heading in the chunk renders one level below its
# expected depth; docutils re-normalizes this in the rendered output, so it's
# cosmetic here.
suppress_warnings = ["etoc.toctree", "myst.header"]
external_toc_path = "./sphinx/_toc.yml"
external_projects_current_project = "rocm-cli"
# No page here cross-references another ROCm project via intersphinx, and the
# default ("all") tries to fetch inventories for every project in
# rocm_docs' bundled projects.yaml — several of which 404 or have moved on
# rocm.docs.amd.com, which would make a `-W` (warnings-as-errors) CI build
# flaky against failures this repo can't fix.
external_projects = []
extensions = [
    "rocm_docs",
    "sphinx_design",
]
exclude_patterns = []

# -- Sphinx setup ----------------------------------------------------------


def setup(app):
    """Sphinx setup"""
    return {"parallel_read_safe": True, "parallel_write_safe": True}
