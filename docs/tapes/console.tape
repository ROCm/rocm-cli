# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Tour the real Console with deterministic telemetry and an offline chat agent.
Output docs/media/console.gif

Set Shell "bash"
Set FontSize 16
Set Width 1440
Set Height 900
Set Padding 16
Set Theme "Catppuccin Mocha"
Set TypingSpeed 45ms

Type "rocm dash --demo --chat-mock"
Sleep 500ms
Enter
Sleep 6s

# Home → ROCm → Serving → Observe.
Type "2"
Sleep 3s
Type "3"
Sleep 3s
Type "4"
Sleep 5s

# Chat is pre-consented in mock mode; Enter focuses its input.
Type "5"
Sleep 3s
Enter
Type "Which GPU needs attention?"
Sleep 500ms
Enter
Sleep 5s

# Leave insert mode, show the global help, then quit.
Escape
Type "?"
Sleep 4s
Type "?"
Sleep 1s
Type "q"
Sleep 2s
