.. meta::
   :description: ROCm CLI documentation.
   :keywords: ROCm CLI, ROCm, AMD, GPU, local AI

=======================
ROCm CLI documentation
=======================

ROCm CLI is a command-line tool for setting up and running local AI on AMD
GPUs, with a full-screen terminal user interface (TUI) dashboard for GPU
telemetry, model serving, and chat.

It ships as a prebuilt bundle for Linux and Windows (x86_64) that contains the
``rocm`` command and the ``rocmd`` background daemon. It needs no Python, Rust,
or existing ROCm install, and includes inference engine adapters for Lemonade
and vLLM.

.. important::

   **Tech Preview:** This software is provided as-is, without warranty or
   guarantee of stability. APIs, commands, and behavior might change without
   notice. Intended for experimentation and early feedback only.

The ROCm CLI public repository is located at
`ROCm/rocm-cli <https://github.com/ROCm/rocm-cli>`_.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Install

      * :doc:`Installing ROCm CLI <install/installation>`

   .. grid-item-card:: Getting started

      * :doc:`Getting started with ROCm CLI <getting-started>`
      * :doc:`See ROCm CLI in action <demos>`

   .. grid-item-card:: Use ROCm CLI

      * :doc:`Command reference <commands>`
      * :doc:`vLLM adapter <engines/vllm>`
