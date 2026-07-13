# rocm services list (against a mock server) — regenerate with:
#   docs/tapes/render.sh docs/tapes/services.tape   (see docs/demos.md)
#
# Pure command tape: the mock server + isolated env + PATH are provided by
# render.sh BEFORE vhs starts.
Output docs/media/services.gif

Set Shell "bash"
Set FontSize 18
Set Width 1000
Set Height 480
Set Padding 20
Set Theme "Catppuccin Mocha"
Set TypingSpeed 60ms

Type "rocm services list"
Sleep 500ms
Enter
Sleep 4s
