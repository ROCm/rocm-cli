param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string] $DistName,

    [Parameter(Position = 1)]
    [string] $OutputDir = "dist",

    [Parameter(Position = 2)]
    [string] $TargetTriple = ""
)

$ErrorActionPreference = "Stop"

function Fail {
    param([string] $Message)
    Write-Error $Message
    exit 1
}

function Resolve-RepoPath {
    param([string] $Path)
    $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
}

function Resolve-CargoHome {
    if (-not [string]::IsNullOrWhiteSpace($env:CARGO_HOME)) {
        return Resolve-RepoPath $env:CARGO_HOME
    }
    if ([string]::IsNullOrWhiteSpace($env:USERPROFILE)) {
        Fail "unable to determine CARGO_HOME; set CARGO_HOME or USERPROFILE"
    }
    Resolve-RepoPath (Join-Path $env:USERPROFILE ".cargo")
}

function Resolve-CommandPath {
    param(
        [string] $Name,
        [string] $FallbackPath = ""
    )

    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    if (-not [string]::IsNullOrWhiteSpace($FallbackPath) -and (Test-Path -LiteralPath $FallbackPath -PathType Leaf)) {
        return $FallbackPath
    }
    Fail "missing required command: $Name"
}

function Resolve-OptionalSigningPrivateKey {
    param([string] $TempDir)

    if (-not [string]::IsNullOrWhiteSpace($env:ROCM_CLI_SIGNING_PRIVATE_KEY_PATH)) {
        return Resolve-RepoPath $env:ROCM_CLI_SIGNING_PRIVATE_KEY_PATH
    }

    if (-not [string]::IsNullOrWhiteSpace($env:ROCM_CLI_SIGNING_PRIVATE_KEY_PEM)) {
        $path = Join-Path $TempDir "rocm-cli-signing-private-key.pem"
        Set-Content -LiteralPath $path -Value $env:ROCM_CLI_SIGNING_PRIVATE_KEY_PEM -Encoding ascii
        return $path
    }

    return ""
}

function Test-Truthy {
    param([string] $Value)
    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $false
    }
    $normalized = $Value.Trim().ToLowerInvariant()
    $normalized -in @("1", "true", "yes", "on")
}

function Write-ArchiveSignature {
    param(
        [string] $ArchivePath,
        [string] $PrivateKeyPath
    )

    if ([string]::IsNullOrWhiteSpace($PrivateKeyPath)) {
        return
    }
    $openssl = Resolve-CommandPath "openssl"
    & $openssl dgst -sha256 -sign $PrivateKeyPath -out "$ArchivePath.sig" $ArchivePath
    if ($LASTEXITCODE -ne 0) {
        Fail "failed to sign $ArchivePath"
    }
}

function Copy-RequiredFile {
    param(
        [string] $Source,
        [string] $Destination
    )
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        Fail "required file not found: $Source"
    }
    Copy-Item -LiteralPath $Source -Destination $Destination -Force
}

$repoRoot = Resolve-RepoPath (Join-Path $PSScriptRoot "..")
$outputRoot = Resolve-RepoPath (Join-Path $repoRoot $OutputDir)
$rootDir = Join-Path $outputRoot $DistName
$binaryDir = if ([string]::IsNullOrWhiteSpace($TargetTriple)) {
    Join-Path $repoRoot "target\release"
} else {
    Join-Path $repoRoot "target\$TargetTriple\release"
}
$archivePath = Join-Path $outputRoot "$DistName.zip"
$cargoHome = Resolve-CargoHome
$cargoExe = Resolve-CommandPath "cargo" (Join-Path $cargoHome "bin\cargo.exe")
$codexBuildTargetDir = Join-Path $cargoHome "target\rocm-cli-codex"

New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
Remove-Item -LiteralPath $rootDir -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $archivePath -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath "$archivePath.sha256" -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath "$archivePath.sig" -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path (Join-Path $rootDir "bin") | Out-Null

$codexTarget = Join-Path $binaryDir "rocm-codex.exe"
if (-not (Test-Path -LiteralPath $codexTarget -PathType Leaf)) {
    $codexManifest = Join-Path $repoRoot "third_party\openai-codex\codex-rs\Cargo.toml"
    if (-not (Test-Path -LiteralPath $codexManifest -PathType Leaf)) {
        Fail "vendored Codex manifest not found: $codexManifest"
    }

    Write-Host "building vendored Codex TUI"
    Write-Host "  manifest: $codexManifest"
    Write-Host "  profile: release"
    Write-Host "  target_dir: $codexBuildTargetDir"
    if (-not [string]::IsNullOrWhiteSpace($TargetTriple)) {
        Write-Host "  target: $TargetTriple"
    }

    $buildArgs = @("build", "--manifest-path", $codexManifest, "-p", "codex-cli", "--bin", "codex", "--release")
    if (-not [string]::IsNullOrWhiteSpace($TargetTriple)) {
        $buildArgs += @("--target", $TargetTriple)
    }

    $previousCargoTargetDir = $env:CARGO_TARGET_DIR
    try {
        $env:CARGO_TARGET_DIR = $codexBuildTargetDir
        & $cargoExe @buildArgs
        if ($LASTEXITCODE -ne 0) {
            Fail "vendored Codex build failed"
        }
    } finally {
        if ([string]::IsNullOrWhiteSpace($previousCargoTargetDir)) {
            Remove-Item Env:\CARGO_TARGET_DIR -ErrorAction SilentlyContinue
        } else {
            $env:CARGO_TARGET_DIR = $previousCargoTargetDir
        }
    }

    $codexSource = if ([string]::IsNullOrWhiteSpace($TargetTriple)) {
        Join-Path $codexBuildTargetDir "release\codex.exe"
    } else {
        Join-Path $codexBuildTargetDir "$TargetTriple\release\codex.exe"
    }
    Copy-RequiredFile $codexSource $codexTarget
}

$bundleBin = Join-Path $rootDir "bin"
Copy-RequiredFile (Join-Path $binaryDir "rocm.exe") (Join-Path $bundleBin "rocm.exe")
Copy-RequiredFile (Join-Path $binaryDir "rocmd.exe") (Join-Path $bundleBin "rocmd.exe")
Copy-RequiredFile (Join-Path $binaryDir "rocm-engine-pytorch.exe") (Join-Path $bundleBin "rocm-engine-pytorch.exe")
Copy-RequiredFile (Join-Path $binaryDir "rocm-engine-llama-cpp.exe") (Join-Path $bundleBin "rocm-engine-llama-cpp.exe")
Copy-RequiredFile (Join-Path $binaryDir "rocm-engine-lemonade.exe") (Join-Path $bundleBin "rocm-engine-lemonade.exe")
Copy-RequiredFile (Join-Path $binaryDir "rocm-engine-atom.exe") (Join-Path $bundleBin "rocm-engine-atom.exe")
Copy-RequiredFile (Join-Path $binaryDir "rocm-engine-vllm.exe") (Join-Path $bundleBin "rocm-engine-vllm.exe")
Copy-RequiredFile (Join-Path $binaryDir "rocm-engine-sglang.exe") (Join-Path $bundleBin "rocm-engine-sglang.exe")
Copy-RequiredFile (Join-Path $binaryDir "rocm-codex.exe") (Join-Path $bundleBin "rocm-codex.exe")
Copy-RequiredFile (Join-Path $repoRoot "README.md") (Join-Path $rootDir "README.md")
Copy-RequiredFile (Join-Path $repoRoot "LICENSE") (Join-Path $rootDir "LICENSE")
Copy-RequiredFile (Join-Path $repoRoot "install.ps1") (Join-Path $rootDir "install.ps1")

Compress-Archive -LiteralPath $rootDir -DestinationPath $archivePath -Force
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash.ToLowerInvariant()
Set-Content -LiteralPath "$archivePath.sha256" -Value "$hash  $(Split-Path -Leaf $archivePath)" -Encoding ascii

$tempSigningDir = Join-Path ([System.IO.Path]::GetTempPath()) ("rocm-cli-sign-" + [System.Guid]::NewGuid().ToString("N"))
try {
    New-Item -ItemType Directory -Force -Path $tempSigningDir | Out-Null
    $privateKeyPath = Resolve-OptionalSigningPrivateKey $tempSigningDir
    if ((Test-Truthy $env:ROCM_CLI_REQUIRE_SIGNATURE) -and [string]::IsNullOrWhiteSpace($privateKeyPath)) {
        Fail "signature is required but ROCM_CLI_SIGNING_PRIVATE_KEY_PATH/PEM is not configured"
    }
    Write-ArchiveSignature $archivePath $privateKeyPath
} finally {
    Remove-Item -LiteralPath $tempSigningDir -Recurse -Force -ErrorAction SilentlyContinue
}

if ((Test-Truthy $env:ROCM_CLI_REQUIRE_SIGNATURE) -and -not (Test-Path -LiteralPath "$archivePath.sig" -PathType Leaf)) {
    Fail "signature is required but $archivePath.sig was not produced"
}

Write-Host "created:"
Write-Host "  $archivePath"
Write-Host "  $archivePath.sha256"
if (Test-Path -LiteralPath "$archivePath.sig" -PathType Leaf) {
    Write-Host "  $archivePath.sig"
}
