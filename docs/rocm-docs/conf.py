# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Configuration file for the Sphinx documentation builder.
# pylint: disable=invalid-name

import logging


class _SuppressUnregisteredProjectWarning(logging.Filter):
    """rocm-cli isn't registered in rocm-docs-core's shared projects.yaml
    catalog yet, so `external_projects_current_project` can never resolve
    until that's added upstream. rocm_docs.projects treats an unresolved
    current project as None everywhere it's used (doxygen_html, theme
    version banner), so the warning is purely informational, not a defect
    here, and would otherwise permanently block a `-W` build."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "not found in projects" not in record.getMessage()


logging.getLogger("sphinx.rocm_docs.projects").addFilter(
    _SuppressUnregisteredProjectWarning()
)

# -- Project information ---------------------------------------------------

project = "ROCm CLI"
author = "Advanced Micro Devices, Inc."
# pylint: disable=redefined-builtin
copyright = "Copyright (c) Advanced Micro Devices, Inc. All rights reserved."
# pylint: enable=redefined-builtin

# -- General configuration -------------------------------------------------

html_theme = "rocm_docs_theme"
html_theme_options = {"flavor": "rocm"}
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
