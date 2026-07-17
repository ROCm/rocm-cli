# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# A first-use ROCm CLI journey. `cargo xtask demos` supplies the isolated mock
# service. The --model below must match the mock's model (xtask DEFAULT_MODEL,
# overridable with ROCM_DEMO_MODEL) or `rocm chat` finds no ready service.
Output docs/media/cli.gif

Set Shell "bash"
Set FontSize 18
Set Width 1200
Set Height 700
Set Padding 20
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms

Type "rocm examine"
Sleep 500ms
Enter
Sleep 4s

Ctrl+L
Type "rocm engines list"
Sleep 400ms
Enter
Sleep 3s

Type "rocm model"
Sleep 400ms
Enter
Sleep 5s

Ctrl+L
Type "rocm services list"
Sleep 400ms
Enter
Sleep 4s

Type "rocm chat --provider local --model Qwen/Qwen3.5-0.6B --prompt 'What can ROCm CLI help me do?'"
Sleep 400ms
Enter
Sleep 6s
