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

# Pinned production release signing public keys (trust roots). These stay empty
# until the repository owner publishes production keys (see docs/release-trust.md,
# "Remaining Owner Step"). While empty, release installs keep the opt-in behavior:
# a signature is verified only when a key is supplied via the parameters/env vars
# or ROCM_CLI_REQUIRE_SIGNATURE=1. Once populated, release-channel installs verify
# signatures by default with these keys as trust roots. Two slots support
# zero-downtime key rotation: a release signed with either the current or the
# pre-staged next key verifies, so the next key is trusted before its first use.
$PinnedReleasePublicKeyCurrent = @"
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAxuyScR/BzV+kuXqWAHtE
+9xiPCWURUYnsio9MOrf2Xe01mBngP7qPcF13+5nrfT3EnuxOn5rSCYwjOndlS+c
KzOw6GZXJD/ZqeojnbXxxsxlftAQHHEke1WCtga5ZEFxOauTeB5nTV/IbjMAl2Xc
M4PaudpFFH/6j/E3gongDmt0hWdpMLbaCcd3i1vMTEsaHZooNoAbJ/dIAHR/dDNM
pScZAZoy0LL3Afhn5Hiv71trfbfnnboVSdhnCoMmisl6/sK55zR7VM8hWDTTowl3
ultUtiz4emTfXDCb2RptOgoydBA+mu9z6O4eVF8S5dVr/S834SK6dD2fWHNnT0dc
JwIDAQAB
-----END PUBLIC KEY-----
"@
$PinnedReleasePublicKeyNext = ""

function Test-HasPinnedReleaseKey {
    return (-not [string]::IsNullOrWhiteSpace($PinnedReleasePublicKeyCurrent)) `
        -or (-not [string]::IsNullOrWhiteSpace($PinnedReleasePublicKeyNext))
}

# Return the candidate signing public keys as an array of file paths. An explicit
# parameter/env-provided key wins (an escape hatch for private mirrors); otherwise
# the pinned production trust roots are used.
function Resolve-SigningPublicKey {
    param(
        [string] $KeyPath,
        [string] $KeyPem,
        [string] $TempRoot
    )

    if (-not [string]::IsNullOrWhiteSpace($KeyPath)) {
        return @((Resolve-InstallerPath $KeyPath))
    }

    if (-not [string]::IsNullOrWhiteSpace($KeyPem)) {
        $path = Join-Path $TempRoot "rocm-cli-signing-public-key.pem"
        Set-Content -LiteralPath $path -Value $KeyPem -Encoding ascii
        return @($path)
    }

    $paths = @()
    $index = 0
    foreach ($pinned in @($PinnedReleasePublicKeyCurrent, $PinnedReleasePublicKeyNext)) {
        if (-not [string]::IsNullOrWhiteSpace($pinned)) {
            $index++
            $path = Join-Path $TempRoot "rocm-cli-pinned-key-$index.pem"
            Set-Content -LiteralPath $path -Value $pinned -Encoding ascii
            $paths += $path
        }
    }
    return $paths
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

function Stop-SigningKeyParsing {
    param([string] $Message)
    throw [System.Security.Cryptography.CryptographicException]::new($Message)
}

# Decode the base64 body of a PEM block (the DER bytes) regardless of the
# header label. openssl and `cargo xtask keygen` emit SubjectPublicKeyInfo
# ("BEGIN PUBLIC KEY") PEM; this strips the armor and returns the raw DER.
function Convert-PemToDer {
    param([string] $Pem)

    $lines = $Pem -split "`r?`n"
    $body = New-Object System.Text.StringBuilder
    $inBody = $false
    foreach ($line in $lines) {
        $trimmed = $line.Trim()
        if ($trimmed -like "-----BEGIN *-----") {
            $inBody = $true
            continue
        }
        if ($trimmed -like "-----END *-----") {
            $inBody = $false
            continue
        }
        if ($inBody -and -not [string]::IsNullOrWhiteSpace($trimmed)) {
            [void] $body.Append($trimmed)
        }
    }
    if ($body.Length -eq 0) {
        Stop-SigningKeyParsing "signing public key is not a valid PEM block"
    }
    try {
        return [System.Convert]::FromBase64String($body.ToString())
    } catch {
        Stop-SigningKeyParsing "signing public key PEM body is not valid base64: $($_.Exception.Message)"
    }
}

# Minimal DER reader: parse (tag, length, value) triplets. Windows PowerShell 5.1
# runs on .NET Framework, which lacks RSA.ImportSubjectPublicKeyInfo (.NET Core
# 3.0+ only), so the SubjectPublicKeyInfo is decoded by hand into RSAParameters.
# Only the small subset of DER needed for an RSA SPKI is handled.
function Read-DerTlv {
    param(
        [byte[]] $Bytes,
        [ref] $Offset
    )

    $start = $Offset.Value
    if ($start + 2 -gt $Bytes.Length) {
        Stop-SigningKeyParsing "signing public key DER is truncated"
    }
    $tag = $Bytes[$start]
    $index = $start + 1
    $lengthByte = $Bytes[$index]
    $index++
    if ($lengthByte -lt 0x80) {
        $length = [int] $lengthByte
    } else {
        $byteCount = $lengthByte -band 0x7f
        if ($byteCount -eq 0 -or $byteCount -gt 4) {
            Stop-SigningKeyParsing "signing public key DER has an unsupported length encoding"
        }
        $length = 0
        for ($i = 0; $i -lt $byteCount; $i++) {
            if ($index -ge $Bytes.Length) {
                Stop-SigningKeyParsing "signing public key DER is truncated"
            }
            $length = ($length -shl 8) -bor [int] $Bytes[$index]
            $index++
        }
        # A 4-byte length with the high bit set decodes to a negative Int32; reject
        # it rather than letting the bounds check below pass and then blowing up in
        # the byte[] allocation with an unhandled OverflowException.
        if ($length -lt 0) {
            Stop-SigningKeyParsing "signing public key DER has an unsupported length encoding"
        }
    }
    $valueStart = $index
    # Compare without adding (`$valueStart + $length` can overflow Int32 for a
    # crafted multi-GB length and wrap negative, silently passing the check).
    if ($length -gt $Bytes.Length - $valueStart) {
        Stop-SigningKeyParsing "signing public key DER is truncated"
    }
    $value = New-Object byte[] $length
    if ($length -gt 0) {
        [System.Array]::Copy($Bytes, $valueStart, $value, 0, $length)
    }
    $Offset.Value = $valueStart + $length
    return [PSCustomObject]@{
        Tag   = $tag
        Value = $value
    }
}

# Convert a DER INTEGER value to an unsigned big-endian byte array, dropping the
# leading 0x00 sign byte that DER inserts when the high bit is set.
function ConvertFrom-DerInteger {
    param([byte[]] $Value)

    $bytes = $Value
    if ($bytes.Length -gt 1 -and $bytes[0] -eq 0) {
        $trimmed = New-Object byte[] ($bytes.Length - 1)
        [System.Array]::Copy($bytes, 1, $trimmed, 0, $trimmed.Length)
        $bytes = $trimmed
    }
    return $bytes
}

# Build an RSA verifier from a SubjectPublicKeyInfo ("BEGIN PUBLIC KEY") PEM.
# Parses the SPKI DER into modulus/exponent and imports them via RSAParameters,
# which works on both Windows PowerShell 5.1 (.NET Framework) and PowerShell 7+
# (.NET). No external process (openssl) is involved.
function Resolve-RsaVerifier {
    param([string] $PublicKeyPem)

    $der = Convert-PemToDer $PublicKeyPem

    $offset = 0
    $spki = Read-DerTlv $der ([ref] $offset)
    if ($spki.Tag -ne 0x30) {
        Stop-SigningKeyParsing "signing public key is not a SubjectPublicKeyInfo structure"
    }

    # Inside the outer SEQUENCE: AlgorithmIdentifier (SEQUENCE) then the
    # public key BIT STRING.
    $inner = 0
    $algorithm = Read-DerTlv $spki.Value ([ref] $inner)
    if ($algorithm.Tag -ne 0x30) {
        Stop-SigningKeyParsing "signing public key algorithm identifier is malformed"
    }
    $subjectPublicKey = Read-DerTlv $spki.Value ([ref] $inner)
    if ($subjectPublicKey.Tag -ne 0x03) {
        Stop-SigningKeyParsing "signing public key bit string is malformed"
    }

    # BIT STRING: first byte is the count of unused bits (0 for a key), the rest
    # is the DER-encoded RSAPublicKey.
    $bitString = $subjectPublicKey.Value
    if ($bitString.Length -lt 1 -or $bitString[0] -ne 0) {
        Stop-SigningKeyParsing "signing public key bit string padding is unsupported"
    }
    $rsaKeyDer = New-Object byte[] ($bitString.Length - 1)
    [System.Array]::Copy($bitString, 1, $rsaKeyDer, 0, $rsaKeyDer.Length)

    # RSAPublicKey ::= SEQUENCE { modulus INTEGER, publicExponent INTEGER }
    $rsaOffset = 0
    $rsaSequence = Read-DerTlv $rsaKeyDer ([ref] $rsaOffset)
    if ($rsaSequence.Tag -ne 0x30) {
        Stop-SigningKeyParsing "signing public key RSA structure is malformed"
    }
    $rsaInner = 0
    $modulus = Read-DerTlv $rsaSequence.Value ([ref] $rsaInner)
    $exponent = Read-DerTlv $rsaSequence.Value ([ref] $rsaInner)
    if ($modulus.Tag -ne 0x02 -or $exponent.Tag -ne 0x02) {
        Stop-SigningKeyParsing "signing public key modulus or exponent is malformed"
    }

    $parameters = New-Object System.Security.Cryptography.RSAParameters
    $parameters.Modulus = ConvertFrom-DerInteger $modulus.Value
    $parameters.Exponent = ConvertFrom-DerInteger $exponent.Value

    $rsa = [System.Security.Cryptography.RSA]::Create()
    try {
        $rsa.ImportParameters($parameters)
    } catch [System.Security.Cryptography.CryptographicException] {
        $rsa.Dispose()
        Stop-SigningKeyParsing "signing public key could not be imported: $($_.Exception.Message)"
    }
    return $rsa
}

function Confirm-ArchiveSignature {
    param(
        [string] $ArchivePath,
        [string] $SignaturePath,
        [string[]] $PublicKeyPaths
    )

    # The `.sig` sidecar is the raw RSASSA-PKCS#1 v1.5 over SHA-256 signature
    # bytes (as produced by `cargo xtask sign` / `openssl dgst -sha256 -sign`),
    # not a PKCS#7/CMS or Authenticode container. certutil/certreq only verify
    # certificate-backed CMS/Authenticode blobs and cannot check a raw detached
    # signature against a bare SubjectPublicKeyInfo public key, so verification
    # uses .NET RSA directly. This keeps the installer free of any openssl
    # runtime dependency while supporting Windows PowerShell 5.1.
    $archiveBytes = [System.IO.File]::ReadAllBytes($ArchivePath)
    $signatureBytes = [System.IO.File]::ReadAllBytes($SignaturePath)
    $hashAlgorithm = [System.Security.Cryptography.HashAlgorithmName]::SHA256
    $padding = [System.Security.Cryptography.RSASignaturePadding]::Pkcs1

    foreach ($key in $PublicKeyPaths) {
        $rsa = $null
        try {
            $pem = Get-Content -LiteralPath $key -Raw
            $rsa = Resolve-RsaVerifier $pem
            if ($rsa.VerifyData($archiveBytes, $signatureBytes, $hashAlgorithm, $padding)) {
                return
            }
        } catch [System.Security.Cryptography.CryptographicException] {
            continue
        } finally {
            if ($null -ne $rsa) {
                $rsa.Dispose()
            }
        }
    }
    Fail "signature verification failed"
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
  "default_engine": "lemonade",
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
    $Repo = "ROCm/rocm-cli"
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
# Production default: once release trust roots are pinned, release-channel installs
# always verify. An unset or 0 ROCM_CLI_REQUIRE_SIGNATURE does not lower this
# floor; pass -SigningPublicKeyPath/-Pem (or the env vars) to point at an
# alternate key (e.g. a private mirror) instead.
if ($Channel -eq "release" -and (Test-HasPinnedReleaseKey)) {
    $signatureRequired = $true
}

try {
    New-Item -ItemType Directory -Force -Path $tempRoot, $extractDir | Out-Null
    $signingPublicKeys = @(Resolve-SigningPublicKey $SigningPublicKeyPath $SigningPublicKeyPem $tempRoot)

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

    if ($signatureRequired -or ($signingPublicKeys.Count -gt 0)) {
        if ($signingPublicKeys.Count -eq 0) {
            Fail "signature verification requires ROCM_CLI_SIGNING_PUBLIC_KEY_PATH, ROCM_CLI_SIGNING_PUBLIC_KEY_PEM, -SigningPublicKeyPath, or -SigningPublicKeyPem"
        }
        Save-File $sigUrl $sigPath "required signature sidecar is missing or unavailable"
        Confirm-ArchiveSignature $archivePath $sigPath $signingPublicKeys
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
        Write-Host "  rocm examine"
    } else {
        $rocmExe = Join-Path $InstallDir "rocm.exe"
        Write-Host "  & `"$rocmExe`" examine"
    }
} finally {
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
