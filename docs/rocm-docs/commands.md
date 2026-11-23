<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Command reference

This page describes each `rocm` command, its options, and what it does. For a
short list of the commands and what each is for, see
[Getting started](getting-started.md).

```{include} ../../README.md
:start-after: "## Commands"
:end-before: "`install driver` previews or installs Linux AMD driver support."
```

<!-- The paragraph below is a deliberate copy of the README "Driver
     installation" opener, with the WSL link retargeted to this site. The
     includes around it are anchored on its first sentence and on the end of its
     last line; edit it and the README together. -->
`install driver` previews or installs Linux AMD driver support. On bare metal,
it prints a non-mutating plan and changes nothing unless you pass `--dkms`,
which plans the DKMS install; add `--yes` to apply it. On WSL2 there is no
DKMS flow: the command plans the ROCDXG install instead, and `--yes` applies it
without `--dkms` (see [Install ROCDXG in WSL](install/wsl.md#install-rocdxg-in-wsl)).

```{include} ../../README.md
:start-after: "install-rocdxg-in-wsl))."
:end-before: "and a chat tab backed by any configured provider."
```

and a chat tab backed by any configured provider. See
[Interactive interfaces](getting-started.md#interactive-interfaces) for the
tab breakdown.

```{include} ../../README.md
:start-after: "for the tab breakdown."
:end-before: "## Contributing"
```
