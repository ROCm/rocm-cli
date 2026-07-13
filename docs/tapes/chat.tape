# rocm chat (one-shot, against a mock server) — regenerate with:
#   docs/tapes/render.sh docs/tapes/chat.tape   (see docs/demos.md)
#
# Pure command tape: the mock server + isolated env + PATH are provided by
# render.sh BEFORE vhs starts, so nothing here blocks the shell while VHS types.
Output docs/media/chat.gif

Set Shell "bash"
Set FontSize 18
Set Width 1050
Set Height 420
Set Padding 20
Set Theme "Catppuccin Mocha"
Set TypingSpeed 45ms

Type "rocm chat --provider local --model Qwen/Qwen3.5-0.6B --prompt 'What GPU am I running?'"
Sleep 500ms
Enter
Sleep 6s
