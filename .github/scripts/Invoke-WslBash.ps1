# Runs a bash script inside the pinned WSL guest.
#
# Piping script text to `wsl.exe` from Windows PowerShell 5.1 is unreliable:
# - Piping a *string* prepends a UTF-8 BOM to the native process's stdin, even
#   with an explicit no-BOM UTF8Encoding assigned to $OutputEncoding.
# - Piping a *byte[]* doesn't write raw bytes either -- PowerShell enumerates
#   the array and stringifies each element, so bash sees "112 10 111 ..."
#   (each byte's decimal value) instead of the script text.
# Writing the script to a file with .NET's WriteAllText (true no-BOM control)
# and executing that file by path sidesteps both: no pipe, no marshaling.
param(
    [Parameter(Mandatory)][string]$Script,
    [string]$Distro = 'Ubuntu-24.04',
    [string]$User = 'root',
    [switch]$PipeFail
)

$tmp = [System.IO.Path]::GetTempFileName()
try {
    [System.IO.File]::WriteAllText($tmp, $Script, [System.Text.UTF8Encoding]::new($false))
    $wslPath = (wsl -d $Distro -u $User -- wslpath $tmp).Trim()
    if ($PipeFail) {
        wsl -d $Distro -u $User -- bash -eo pipefail $wslPath
    } else {
        wsl -d $Distro -u $User -- bash $wslPath
    }
} finally {
    Remove-Item $tmp -ErrorAction SilentlyContinue
}
