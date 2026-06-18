param(
    [Parameter(Position = 0)]
    [string] $Channel = $env:ROCM_CLI_CHANNEL,

    [string] $Repo = $env:ROCM_CLI_GITHUB_REPO,

    [string] $InstallDir = $env:ROCM_CLI_INSTALL_DIR,

    [string] $DownloadBase = $env:ROCM_CLI_DOWNLOAD_BASE,

    [string] $ArchiveExtension = $env:ROCM_CLI_ARCHIVE_EXTENSION,

    [string] $SigningPublicKeyPath = $env:ROCM_CLI_SIGNING_PUBLIC_KEY_PATH,

    [string] $SigningPublicKeyPem = $env:ROCM_CLI_SIGNING_PUBLIC_KEY_PEM,

    [switch] $RequireSignature,

    [switch] $NoPathUpdate
)

$ErrorActionPreference = "Stop"

function Fail {
    param([string] $Message)
    Write-Error "rocm-cli installer: $Message"
    exit 1
}

function Confirm-Command {
    param([string] $Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Fail "missing required command: $Name"
    }
}

function Test-Truthy {
    param([string] $Value)
    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $false
    }
    $normalized = $Value.Trim().ToLowerInvariant()
    $normalized -in @("1", "true", "yes", "on")
}

function Resolve-InstallerPath {
    param([string] $Path)
    $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
}

function Test-PathUnder {
    param(
        [string] $Child,
        [string] $Root
    )
    $rootFull = [System.IO.Path]::GetFullPath($Root).TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
    $childFull = [System.IO.Path]::GetFullPath($Child)
    if ($childFull.Equals($rootFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }
    $rootPrefix = $rootFull + [System.IO.Path]::DirectorySeparatorChar
    $childFull.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)
}

function Convert-FileUriToPath {
    param([string] $UriText)
    try {
        $uri = [System.Uri] $UriText
        if ($uri.IsFile) {
            return $uri.LocalPath
        }
    } catch {
        return $null
    }
    return $null
}

function Resolve-SigningPublicKey {
    param(
        [string] $KeyPath,
        [string] $KeyPem,
        [string] $TempRoot
    )

    if (-not [string]::IsNullOrWhiteSpace($KeyPath)) {
        return Resolve-InstallerPath $KeyPath
    }

    if (-not [string]::IsNullOrWhiteSpace($KeyPem)) {
        $path = Join-Path $TempRoot "rocm-cli-signing-public-key.pem"
        Set-Content -LiteralPath $path -Value $KeyPem -Encoding ascii
        return $path
    }

    return ""
}

function Save-File {
    param(
        [string] $Url,
        [string] $OutputPath,
        [string] $FailureMessage = ""
    )
    if ([string]::IsNullOrWhiteSpace($FailureMessage)) {
        $FailureMessage = "failed to download $Url"
    }

    $localPath = Convert-FileUriToPath $Url
    if ($localPath) {
        try {
            Copy-Item -LiteralPath $localPath -Destination $OutputPath -Force -ErrorAction Stop
        } catch {
            Remove-Item -LiteralPath $OutputPath -Force -ErrorAction SilentlyContinue
            Fail "${FailureMessage}: $($_.Exception.Message)"
        }
        return
    }

    $parameters = @{
        Uri = $Url
        OutFile = $OutputPath
        ErrorAction = "Stop"
    }
    if ($PSVersionTable.PSVersion.Major -lt 6) {
        $parameters.UseBasicParsing = $true
    }
    try {
        Invoke-WebRequest @parameters
    } catch {
        Remove-Item -LiteralPath $OutputPath -Force -ErrorAction SilentlyContinue
        Fail "${FailureMessage}: $($_.Exception.Message)"
    }
}

function Confirm-ArchiveSignature {
    param(
        [string] $ArchivePath,
        [string] $SignaturePath,
        [string] $PublicKeyPath
    )

    Confirm-Command openssl
    & openssl dgst -sha256 -verify $PublicKeyPath -signature $SignaturePath $ArchivePath | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Fail "signature verification failed"
    }
}

function Get-ExpectedSha256 {
    param([string] $Path)
    $text = Get-Content -LiteralPath $Path -Raw
    $match = [regex]::Match($text, "(?i)\b[0-9a-f]{64}\b")
    if (-not $match.Success) {
        Fail "checksum file did not contain a sha256 digest"
    }
    $match.Value.ToLowerInvariant()
}

function Get-RocmConfigDir {
    if (-not [string]::IsNullOrWhiteSpace($env:ROCM_CLI_CONFIG_DIR)) {
        return Resolve-InstallerPath $env:ROCM_CLI_CONFIG_DIR
    }
    $homeDir = if (-not [string]::IsNullOrWhiteSpace($env:USERPROFILE)) {
        $env:USERPROFILE
    } elseif (-not [string]::IsNullOrWhiteSpace($env:HOME)) {
        $env:HOME
    } else {
        Fail "unable to determine the user home directory for rocm-cli config"
    }
    Join-Path $homeDir ".rocm"
}

function Write-MinimalConfigIfMissing {
    $configDir = Get-RocmConfigDir
    $configPath = Join-Path $configDir "config.json"
    if (Test-Path -LiteralPath $configPath -PathType Leaf) {
        Write-Host "config: existing $configPath"
        return
    }

    New-Item -ItemType Directory -Force -Path $configDir | Out-Null
    $json = @'
{
  "default_engine": "pytorch",
  "telemetry": {
    "mode": "local"
  },
  "permissions": {
    "mode": "ask"
  },
  "setup": {
    "completed": false
  }
}
'@
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($configPath, $json + [Environment]::NewLine, $utf8NoBom)
    Write-Host "config: created $configPath"
}

function Expand-RocmArchive {
    param(
        [string] $ArchivePath,
        [string] $ExtractDir
    )

    if ($ArchivePath.EndsWith(".zip", [System.StringComparison]::OrdinalIgnoreCase)) {
        Expand-Archive -LiteralPath $ArchivePath -DestinationPath $ExtractDir -Force
        return
    }

    if ($ArchivePath.EndsWith(".tar.gz", [System.StringComparison]::OrdinalIgnoreCase) -or
        $ArchivePath.EndsWith(".tgz", [System.StringComparison]::OrdinalIgnoreCase)) {
        Confirm-Command tar
        & tar -xzf $ArchivePath -C $ExtractDir
        if ($LASTEXITCODE -ne 0) {
            Fail "failed to extract archive with tar"
        }
        return
    }

    Fail "unsupported archive extension for $ArchivePath"
}

function Find-BundleDir {
    param([string] $ExtractDir)

    $candidates = @((Get-Item -LiteralPath $ExtractDir))
    $candidates += Get-ChildItem -LiteralPath $ExtractDir -Directory
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath (Join-Path $candidate.FullName "bin\rocm.exe")) {
            return $candidate.FullName
        }
    }

    Fail "unable to locate extracted bundle directory"
}

function Test-PathInList {
    param(
        [string] $PathList,
        [string] $Path
    )
    if ([string]::IsNullOrWhiteSpace($PathList)) {
        return $false
    }

    $target = [System.IO.Path]::GetFullPath($Path).TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
    foreach ($entry in $PathList -split ";") {
        if ([string]::IsNullOrWhiteSpace($entry)) {
            continue
        }
        try {
            $entryFull = [System.IO.Path]::GetFullPath($entry).TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
        } catch {
            continue
        }
        if ($entryFull.Equals($target, [System.StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }
    return $false
}

function Add-PathListEntry {
    param(
        [string] $PathList,
        [string] $Path
    )

    if (Test-PathInList $PathList $Path) {
        return $PathList
    }

    if ([string]::IsNullOrWhiteSpace($PathList)) {
        return $Path
    }

    return "$Path;$PathList"
}

function Add-InstallDirToUserPath {
    param([string] $Path)

    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $userPathAlreadyConfigured = Test-PathInList $userPath $Path
    if (-not $userPathAlreadyConfigured) {
        $newUserPath = Add-PathListEntry $userPath $Path
        [Environment]::SetEnvironmentVariable("Path", $newUserPath, "User")
    }

    $processPathAlreadyConfigured = Test-PathInList $env:Path $Path
    if (-not $processPathAlreadyConfigured) {
        $env:Path = Add-PathListEntry $env:Path $Path
    }

    if ($userPathAlreadyConfigured) {
        Write-Host "user PATH already configured:"
        Write-Host "  path: $Path"
    } else {
        Write-Host "user PATH updated:"
        Write-Host "  added: $Path"
        Write-Host "  new PowerShell windows can run: rocm"
    }

    if ($processPathAlreadyConfigured) {
        Write-Host "installer PATH already configured:"
        Write-Host "  path: $Path"
    } else {
        Write-Host "installer PATH updated:"
        Write-Host "  this PowerShell process can run: rocm"
    }
}

if ([string]::IsNullOrWhiteSpace($Repo)) {
    $Repo = "powderluv/rocm-cli"
}

if ([string]::IsNullOrWhiteSpace($Channel)) {
    $Channel = "release"
}

if ([string]::IsNullOrWhiteSpace($ArchiveExtension)) {
    $ArchiveExtension = "zip"
}
$ArchiveExtension = $ArchiveExtension.TrimStart(".")

if ([string]::IsNullOrWhiteSpace($InstallDir)) {
    $homeDir = if (-not [string]::IsNullOrWhiteSpace($env:USERPROFILE)) {
        $env:USERPROFILE
    } else {
        [Environment]::GetFolderPath("UserProfile")
    }
    $InstallDir = Join-Path $homeDir ".local\bin"
}
$InstallDir = Resolve-InstallerPath $InstallDir

$architecture = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
switch ($architecture) {
    "X64" { $platformArch = "amd64" }
    default { Fail "unsupported architecture: $architecture (installer currently supports Windows x86_64 only)" }
}

$platformOs = "windows"

switch ($Channel) {
    "nightly" {
        $assetBase = "rocm-cli-nightly-$platformOs-$platformArch.$ArchiveExtension"
        $releasePath = "releases/download/nightly"
    }
    "release" {
        $assetBase = "rocm-cli-$platformOs-$platformArch.$ArchiveExtension"
        $releasePath = "releases/latest/download"
    }
    default {
        $assetBase = "rocm-cli-$platformOs-$platformArch.$ArchiveExtension"
        $releasePath = "releases/download/$Channel"
    }
}

if ([string]::IsNullOrWhiteSpace($DownloadBase)) {
    $DownloadBase = "https://github.com/$Repo/$releasePath"
}
$DownloadBase = $DownloadBase.TrimEnd("/")
$archiveUrl = "$DownloadBase/$assetBase"
$shaUrl = "$archiveUrl.sha256"
$sigUrl = "$archiveUrl.sig"

Confirm-Command Expand-Archive
Confirm-Command Get-FileHash

if ($PSVersionTable.PSEdition -eq "Desktop") {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 -bor [Net.ServicePointManager]::SecurityProtocol
}

$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("rocm-cli-install-" + [System.Guid]::NewGuid().ToString("N"))
$archivePath = Join-Path $tempRoot $assetBase
$shaPath = "$archivePath.sha256"
$sigPath = "$archivePath.sig"
$extractDir = Join-Path $tempRoot "extract"
$manifestPath = Join-Path $InstallDir ".rocm-cli-manifest"
$signatureRequired = $RequireSignature -or (Test-Truthy $env:ROCM_CLI_REQUIRE_SIGNATURE)

try {
    New-Item -ItemType Directory -Force -Path $tempRoot, $extractDir | Out-Null
    $signingPublicKey = Resolve-SigningPublicKey $SigningPublicKeyPath $SigningPublicKeyPem $tempRoot

    Write-Host "rocm-cli installer"
    Write-Host "  repo: $Repo"
    Write-Host "  channel: $Channel"
    Write-Host "  install_dir: $InstallDir"
    Write-Host "  download: $archiveUrl"

    Save-File $archiveUrl $archivePath
    Save-File $shaUrl $shaPath

    $expected = Get-ExpectedSha256 $shaPath
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash.ToLowerInvariant()
    if ($expected -ne $actual) {
        Fail "checksum verification failed"
    }

    if ($signatureRequired -or -not [string]::IsNullOrWhiteSpace($signingPublicKey)) {
        if ([string]::IsNullOrWhiteSpace($signingPublicKey)) {
            Fail "signature verification requires ROCM_CLI_SIGNING_PUBLIC_KEY_PATH, ROCM_CLI_SIGNING_PUBLIC_KEY_PEM, -SigningPublicKeyPath, or -SigningPublicKeyPem"
        }
        Save-File $sigUrl $sigPath "required signature sidecar is missing or unavailable"
        Confirm-ArchiveSignature $archivePath $sigPath $signingPublicKey
        Write-Host "signature verified"
    }

    Expand-RocmArchive $archivePath $extractDir
    $bundleDir = Find-BundleDir $extractDir
    $bundleBin = Join-Path $bundleDir "bin"

    # First-party engines are built into rocm.exe and run in-process; the
    # standalone rocm-engine-*.exe binaries are an external plugin fallback.
    foreach ($required in @("rocm.exe", "rocmd.exe")) {
        if (-not (Test-Path -LiteralPath (Join-Path $bundleBin $required) -PathType Leaf)) {
            Fail "bundle did not contain bin/$required"
        }
    }

    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    Write-MinimalConfigIfMissing

    if (Test-Path -LiteralPath $manifestPath -PathType Leaf) {
        Write-Host "removing previous rocm-cli install"
        foreach ($installedPath in Get-Content -LiteralPath $manifestPath) {
            if ([string]::IsNullOrWhiteSpace($installedPath)) {
                continue
            }
            $installedPath = $installedPath.Trim()
            if (Test-PathUnder $installedPath $InstallDir) {
                Remove-Item -LiteralPath $installedPath -Force -ErrorAction SilentlyContinue
            } else {
                Write-Warning "skipping manifest entry outside install dir: $installedPath"
            }
        }
        Remove-Item -LiteralPath $manifestPath -Force -ErrorAction SilentlyContinue
    }

    $manifestEntries = New-Object System.Collections.Generic.List[string]
    foreach ($binPath in Get-ChildItem -LiteralPath $bundleBin -File) {
        $targetPath = Join-Path $InstallDir $binPath.Name
        Remove-Item -LiteralPath $targetPath -Force -ErrorAction SilentlyContinue
        Copy-Item -LiteralPath $binPath.FullName -Destination $targetPath -Force
        $manifestEntries.Add($targetPath)
    }

    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllLines($manifestPath, [string[]] $manifestEntries, $utf8NoBom)

    Write-Host "installed:"
    foreach ($entry in $manifestEntries) {
        Write-Host "  $entry"
    }

    $updateUserPath = -not $NoPathUpdate
    if ($env:ROCM_CLI_UPDATE_USER_PATH -eq "0" -or $env:ROCM_CLI_UPDATE_SHELL_PATH -eq "0") {
        $updateUserPath = $false
    }

    if ($updateUserPath) {
        Add-InstallDirToUserPath $InstallDir
    } elseif (-not (Test-PathInList $env:Path $InstallDir)) {
        Write-Host "note: $InstallDir is not on PATH"
        Write-Host "  add it to the user PATH or run:"
        Write-Host "  `$env:Path = `"$InstallDir;`$env:Path`""
    }

    Write-Host "run:"
    if (Test-PathInList $env:Path $InstallDir) {
        Write-Host "  rocm doctor"
    } else {
        $rocmExe = Join-Path $InstallDir "rocm.exe"
        Write-Host "  & `"$rocmExe`" doctor"
    }
} finally {
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
