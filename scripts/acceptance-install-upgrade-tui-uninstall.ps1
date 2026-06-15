$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
$AcceptanceRoot = Join-Path $RepoRoot "target\acceptance-windows"
$DistDirRelative = "target\acceptance-windows\dist"
$DistDir = Join-Path $RepoRoot $DistDirRelative
$DistName = "rocm-cli-windows-amd64"
$PemDistDirRelative = "target\acceptance-windows\pem-dist"
$PemDistDir = Join-Path $RepoRoot $PemDistDirRelative
$InstallDir = Join-Path $AcceptanceRoot "install\bin"
$PemInstallDir = Join-Path $AcceptanceRoot "pem-install\bin"
$PathUpdateInstallDir = Join-Path $AcceptanceRoot ("path-update-install-" + [System.Guid]::NewGuid().ToString("N") + "\bin")
$ConfigDir = Join-Path $AcceptanceRoot "rocm-config"
$ConfigFile = Join-Path $ConfigDir "config.json"
$DataDir = Join-Path $AcceptanceRoot "rocm-data"
$CacheDir = Join-Path $AcceptanceRoot "rocm-cache"
$AppDataDir = Join-Path $AcceptanceRoot "appdata"
$LocalAppDataDir = Join-Path $AcceptanceRoot "localappdata"
$InstallLog1 = Join-Path $AcceptanceRoot "install-1.log"
$InstallLog2 = Join-Path $AcceptanceRoot "install-2.log"
$ChecksumLog = Join-Path $AcceptanceRoot "checksum-failure.log"
$SignatureLog = Join-Path $AcceptanceRoot "signature-failure.log"
$MissingSignatureLog = Join-Path $AcceptanceRoot "missing-signature-failure.log"
$NoPublicKeyLog = Join-Path $AcceptanceRoot "no-public-key-failure.log"
$PemInstallLog = Join-Path $AcceptanceRoot "pem-install.log"
$PathUpdateInstallLog = Join-Path $AcceptanceRoot "path-update-install.log"
$DoctorLog = Join-Path $AcceptanceRoot "doctor.log"
$UninstallLog = Join-Path $AcceptanceRoot "uninstall.log"

function Fail {
    param([string] $Message)
    Write-Error "acceptance failed: $Message"
    exit 1
}

function Assert-File {
    param([string] $Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        Fail "expected file: $Path"
    }
}

function Assert-Missing {
    param([string] $Path)
    if (Test-Path -LiteralPath $Path) {
        Fail "expected path to be removed: $Path"
    }
}

function Invoke-Checked {
    param(
        [string] $Label,
        [string] $FilePath,
        [string[]] $Arguments,
        [string] $LogPath = ""
    )

    Write-Host $Label
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $FilePath @Arguments 2>&1
        $exitCode = if ($LASTEXITCODE -is [int]) { $LASTEXITCODE } else { 0 }
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }
    if (-not [string]::IsNullOrWhiteSpace($LogPath)) {
        $output | Set-Content -LiteralPath $LogPath -Encoding utf8
    }
    $output | ForEach-Object { Write-Host $_ }
    if ($exitCode -ne 0) {
        Fail "$Label exited with status $exitCode"
    }
}

function Invoke-ExpectFailure {
    param(
        [string] $Label,
        [string] $FilePath,
        [string[]] $Arguments,
        [string] $LogPath = ""
    )

    Write-Host $Label
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $FilePath @Arguments 2>&1
        $exitCode = if ($LASTEXITCODE -is [int]) { $LASTEXITCODE } else { 0 }
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }
    if (-not [string]::IsNullOrWhiteSpace($LogPath)) {
        $output | Set-Content -LiteralPath $LogPath -Encoding utf8
    }
    $output | ForEach-Object { Write-Host $_ }
    if ($exitCode -eq 0) {
        Fail "$Label unexpectedly succeeded"
    }
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

function Test-PathListContains {
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

try {
    if (Test-Path -LiteralPath $AcceptanceRoot) {
        Remove-Item -LiteralPath $AcceptanceRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $AcceptanceRoot | Out-Null

    $psCommand = Get-Command pwsh -ErrorAction SilentlyContinue
    if (-not $psCommand) {
        $psCommand = Get-Command powershell -ErrorAction Stop
    }
    $psExe = $psCommand.Source

    $cargoExe = Resolve-CommandPath "cargo" (Join-Path $env:USERPROFILE ".cargo\bin\cargo.exe")
    $opensslExe = Resolve-CommandPath "openssl"

    $signingPrivateKey = Join-Path $AcceptanceRoot "signing-private.pem"
    $signingPublicKey = Join-Path $AcceptanceRoot "signing-public.pem"
    & $opensslExe genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out $signingPrivateKey | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Fail "failed to generate acceptance signing private key"
    }
    & $opensslExe rsa -in $signingPrivateKey -pubout -out $signingPublicKey | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Fail "failed to generate acceptance signing public key"
    }

    Write-Host "acceptance: build release binaries"
    Push-Location $RepoRoot
    try {
        & $cargoExe build --release -p rocm -p rocmd -p rocm-engine-pytorch -p rocm-engine-llama-cpp -p rocm-engine-lemonade -p rocm-engine-atom -p rocm-engine-vllm -p rocm-engine-sglang
        if ($LASTEXITCODE -ne 0) {
            Fail "cargo build failed"
        }
    } finally {
        Pop-Location
    }

    Write-Host "acceptance: package local release bundle"
    Push-Location $RepoRoot
    try {
        $env:ROCM_CLI_SIGNING_PRIVATE_KEY_PATH = $signingPrivateKey
        $env:ROCM_CLI_REQUIRE_SIGNATURE = "1"
        & $psExe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\package-windows-release.ps1") $DistName $DistDirRelative
        if ($LASTEXITCODE -ne 0) {
            Fail "Windows package failed"
        }
    } finally {
        Remove-Item Env:\ROCM_CLI_SIGNING_PRIVATE_KEY_PATH -ErrorAction SilentlyContinue
        Remove-Item Env:\ROCM_CLI_REQUIRE_SIGNATURE -ErrorAction SilentlyContinue
        Pop-Location
    }
    Assert-File (Join-Path $DistDir "$DistName.zip.sig")

    Write-Host "acceptance: package with generated private key PEM"
    Push-Location $RepoRoot
    try {
        $env:ROCM_CLI_SIGNING_PRIVATE_KEY_PEM = Get-Content -LiteralPath $signingPrivateKey -Raw
        $env:ROCM_CLI_REQUIRE_SIGNATURE = "1"
        & $psExe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\package-windows-release.ps1") $DistName $PemDistDirRelative
        if ($LASTEXITCODE -ne 0) {
            Fail "Windows PEM package failed"
        }
    } finally {
        Remove-Item Env:\ROCM_CLI_SIGNING_PRIVATE_KEY_PEM -ErrorAction SilentlyContinue
        Remove-Item Env:\ROCM_CLI_REQUIRE_SIGNATURE -ErrorAction SilentlyContinue
        Pop-Location
    }
    Assert-File (Join-Path $PemDistDir "$DistName.zip.sig")

    $downloadBase = ([System.Uri]::new($DistDir + [System.IO.Path]::DirectorySeparatorChar)).AbsoluteUri.TrimEnd("/")
    $pemDownloadBase = ([System.Uri]::new($PemDistDir + [System.IO.Path]::DirectorySeparatorChar)).AbsoluteUri.TrimEnd("/")
    $env:ROCM_CLI_CONFIG_DIR = $ConfigDir

    Write-Host "acceptance: default install updates user PATH"
    $originalUserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    try {
        Remove-Item Env:\ROCM_CLI_UPDATE_USER_PATH -ErrorAction SilentlyContinue
        Remove-Item Env:\ROCM_CLI_UPDATE_SHELL_PATH -ErrorAction SilentlyContinue
        $pathUpdateInstallArgs = @(
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            (Join-Path $RepoRoot "install.ps1"),
            "release",
            "-InstallDir",
            $PathUpdateInstallDir,
            "-DownloadBase",
            $downloadBase,
            "-SigningPublicKeyPath",
            $signingPublicKey,
            "-RequireSignature"
        )
        Invoke-Checked "acceptance: PATH-updating install" $psExe $pathUpdateInstallArgs $PathUpdateInstallLog
        $updatedUserPath = [Environment]::GetEnvironmentVariable("Path", "User")
        if (-not (Test-PathListContains $updatedUserPath $PathUpdateInstallDir)) {
            Fail "installer did not add install dir to the user PATH"
        }
        if (-not (Select-String -LiteralPath $PathUpdateInstallLog -Pattern "user PATH updated" -Quiet)) {
            Fail "installer did not report the user PATH update"
        }
        if (-not (Select-String -LiteralPath $PathUpdateInstallLog -Pattern "installer PATH updated" -Quiet)) {
            Fail "installer did not update the installer process PATH"
        }
    } finally {
        [Environment]::SetEnvironmentVariable("Path", $originalUserPath, "User")
        $env:ROCM_CLI_UPDATE_USER_PATH = "0"
    }

    $installArgs = @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        (Join-Path $RepoRoot "install.ps1"),
        "release",
        "-InstallDir",
        $InstallDir,
        "-DownloadBase",
        $downloadBase,
        "-SigningPublicKeyPath",
        $signingPublicKey,
        "-RequireSignature",
        "-NoPathUpdate"
    )

    Write-Host "acceptance: install with generated public key PEM"
    $pemInstallArgs = @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        (Join-Path $RepoRoot "install.ps1"),
        "release",
        "-InstallDir",
        $PemInstallDir,
        "-DownloadBase",
        $pemDownloadBase,
        "-SigningPublicKeyPem",
        (Get-Content -LiteralPath $signingPublicKey -Raw),
        "-RequireSignature",
        "-NoPathUpdate"
    )
    Invoke-Checked "acceptance: PEM public key install" $psExe $pemInstallArgs $PemInstallLog
    if (-not (Select-String -LiteralPath $PemInstallLog -Pattern "signature verified" -Quiet)) {
        Fail "installer did not report PEM signature verification"
    }
    Assert-File (Join-Path $PemInstallDir "rocm.exe")
    Assert-File (Join-Path $PemInstallDir ".rocm-cli-manifest")
    Assert-File $ConfigFile
    if (-not (Select-String -LiteralPath $ConfigFile -Pattern '"default_engine"\s*:\s*"pytorch"' -Quiet)) {
        Fail "installer did not seed minimal config with the expected default engine"
    }

    Write-Host "acceptance: reject required signature without public key"
    $noPublicKeyInstallDir = Join-Path $AcceptanceRoot "no-public-key-install\bin"
    $noPublicKeyInstallArgs = @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        (Join-Path $RepoRoot "install.ps1"),
        "release",
        "-InstallDir",
        $noPublicKeyInstallDir,
        "-DownloadBase",
        $downloadBase,
        "-RequireSignature",
        "-NoPathUpdate"
    )
    Invoke-ExpectFailure "acceptance: required signature no public key install" $psExe $noPublicKeyInstallArgs $NoPublicKeyLog
    if (-not (Select-String -LiteralPath $NoPublicKeyLog -Pattern "signature verification requires ROCM_CLI_SIGNING_PUBLIC_KEY_PATH" -Quiet)) {
        Fail "installer did not report missing public key for required signature"
    }
    Assert-Missing (Join-Path $noPublicKeyInstallDir "rocm.exe")
    Assert-Missing (Join-Path $noPublicKeyInstallDir ".rocm-cli-manifest")

    Write-Host "acceptance: reject mismatched checksum before activation"
    $badChecksumDistDir = Join-Path $AcceptanceRoot "bad-sha-dist"
    $checksumInstallDir = Join-Path $AcceptanceRoot "bad-checksum-install\bin"
    New-Item -ItemType Directory -Force -Path $badChecksumDistDir | Out-Null
    Copy-Item -LiteralPath (Join-Path $DistDir "$DistName.zip") -Destination (Join-Path $badChecksumDistDir "$DistName.zip") -Force
    Set-Content -LiteralPath (Join-Path $badChecksumDistDir "$DistName.zip.sha256") -Value ("0" * 64 + "  $DistName.zip") -Encoding ascii
    $badChecksumDownloadBase = ([System.Uri]::new($badChecksumDistDir + [System.IO.Path]::DirectorySeparatorChar)).AbsoluteUri.TrimEnd("/")
    $badChecksumInstallArgs = @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        (Join-Path $RepoRoot "install.ps1"),
        "release",
        "-InstallDir",
        $checksumInstallDir,
        "-DownloadBase",
        $badChecksumDownloadBase,
        "-SigningPublicKeyPath",
        $signingPublicKey,
        "-RequireSignature",
        "-NoPathUpdate"
    )
    Invoke-ExpectFailure "acceptance: checksum mismatch install" $psExe $badChecksumInstallArgs $ChecksumLog
    if (-not (Select-String -LiteralPath $ChecksumLog -Pattern "checksum verification failed" -Quiet)) {
        Fail "installer did not report checksum verification failure"
    }
    Assert-Missing (Join-Path $checksumInstallDir "rocm.exe")
    Assert-Missing (Join-Path $checksumInstallDir ".rocm-cli-manifest")

    Write-Host "acceptance: reject mismatched signature before activation"
    $badSignatureDistDir = Join-Path $AcceptanceRoot "bad-sig-dist"
    $signatureInstallDir = Join-Path $AcceptanceRoot "bad-signature-install\bin"
    New-Item -ItemType Directory -Force -Path $badSignatureDistDir | Out-Null
    Copy-Item -LiteralPath (Join-Path $DistDir "$DistName.zip") -Destination (Join-Path $badSignatureDistDir "$DistName.zip") -Force
    Copy-Item -LiteralPath (Join-Path $DistDir "$DistName.zip.sha256") -Destination (Join-Path $badSignatureDistDir "$DistName.zip.sha256") -Force
    Set-Content -LiteralPath (Join-Path $badSignatureDistDir "$DistName.zip.sig") -Value "not a real signature" -Encoding ascii
    $badSignatureDownloadBase = ([System.Uri]::new($badSignatureDistDir + [System.IO.Path]::DirectorySeparatorChar)).AbsoluteUri.TrimEnd("/")
    $badSignatureInstallArgs = @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        (Join-Path $RepoRoot "install.ps1"),
        "release",
        "-InstallDir",
        $signatureInstallDir,
        "-DownloadBase",
        $badSignatureDownloadBase,
        "-SigningPublicKeyPath",
        $signingPublicKey,
        "-RequireSignature",
        "-NoPathUpdate"
    )
    Invoke-ExpectFailure "acceptance: signature mismatch install" $psExe $badSignatureInstallArgs $SignatureLog
    if (-not (Select-String -LiteralPath $SignatureLog -Pattern "signature verification failed" -Quiet)) {
        Fail "installer did not report signature verification failure"
    }
    Assert-Missing (Join-Path $signatureInstallDir "rocm.exe")
    Assert-Missing (Join-Path $signatureInstallDir ".rocm-cli-manifest")

    Write-Host "acceptance: reject missing signature before activation"
    $missingSignatureDistDir = Join-Path $AcceptanceRoot "missing-sig-dist"
    $missingSignatureInstallDir = Join-Path $AcceptanceRoot "missing-signature-install\bin"
    New-Item -ItemType Directory -Force -Path $missingSignatureDistDir | Out-Null
    Copy-Item -LiteralPath (Join-Path $DistDir "$DistName.zip") -Destination (Join-Path $missingSignatureDistDir "$DistName.zip") -Force
    Copy-Item -LiteralPath (Join-Path $DistDir "$DistName.zip.sha256") -Destination (Join-Path $missingSignatureDistDir "$DistName.zip.sha256") -Force
    $missingSignatureDownloadBase = ([System.Uri]::new($missingSignatureDistDir + [System.IO.Path]::DirectorySeparatorChar)).AbsoluteUri.TrimEnd("/")
    $missingSignatureInstallArgs = @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        (Join-Path $RepoRoot "install.ps1"),
        "release",
        "-InstallDir",
        $missingSignatureInstallDir,
        "-DownloadBase",
        $missingSignatureDownloadBase,
        "-SigningPublicKeyPath",
        $signingPublicKey,
        "-RequireSignature",
        "-NoPathUpdate"
    )
    Invoke-ExpectFailure "acceptance: missing signature install" $psExe $missingSignatureInstallArgs $MissingSignatureLog
    if (-not (Select-String -LiteralPath $MissingSignatureLog -Pattern "required signature sidecar is missing or unavailable" -Quiet)) {
        Fail "installer did not report missing signature download failure"
    }
    Assert-Missing (Join-Path $missingSignatureInstallDir "rocm.exe")
    Assert-Missing (Join-Path $missingSignatureInstallDir ".rocm-cli-manifest")

    Set-Content -LiteralPath $ConfigFile -Value '{"default_engine":"llama.cpp"}' -Encoding utf8
    Invoke-Checked "acceptance: first install" $psExe $installArgs $InstallLog1
    if (-not (Select-String -LiteralPath $InstallLog1 -Pattern "signature verified" -Quiet)) {
        Fail "installer did not report signature verification"
    }
    if (-not (Select-String -LiteralPath $ConfigFile -Pattern '"default_engine"\s*:\s*"llama.cpp"' -Quiet)) {
        Fail "installer overwrote an existing config file"
    }
    Assert-File (Join-Path $InstallDir "rocm.exe")
    Assert-File (Join-Path $InstallDir "rocmd.exe")
    Assert-File (Join-Path $InstallDir "rocm-engine-pytorch.exe")
    Assert-File (Join-Path $InstallDir "rocm-engine-llama-cpp.exe")
    Assert-File (Join-Path $InstallDir "rocm-engine-lemonade.exe")
    Assert-File (Join-Path $InstallDir "rocm-engine-atom.exe")
    Assert-File (Join-Path $InstallDir "rocm-engine-vllm.exe")
    Assert-File (Join-Path $InstallDir "rocm-engine-sglang.exe")
    Assert-File (Join-Path $InstallDir ".rocm-cli-manifest")

    Write-Host "acceptance: simulate stale prior install entry and reinstall"
    $stalePath = Join-Path $InstallDir "rocm-engine-stale.exe"
    Set-Content -LiteralPath $stalePath -Value "stale" -Encoding ascii
    Add-Content -LiteralPath (Join-Path $InstallDir ".rocm-cli-manifest") -Value $stalePath
    Invoke-Checked "acceptance: reinstall" $psExe $installArgs $InstallLog2
    Assert-Missing $stalePath
    Assert-File (Join-Path $InstallDir ".rocm-cli-manifest")
    if (-not (Select-String -LiteralPath $InstallLog2 -Pattern "removing previous rocm-cli install" -Quiet)) {
        Fail "installer did not report removal of previous install"
    }

    $rocmExe = Join-Path $InstallDir "rocm.exe"
    $rocmdExe = Join-Path $InstallDir "rocmd.exe"
    $RealUserRocmDir = Join-Path $env:USERPROFILE ".rocm"
    $env:ROCM_CLI_CONFIG_DIR = $ConfigDir
    $env:ROCM_CLI_DATA_DIR = $DataDir
    $env:ROCM_CLI_CACHE_DIR = $CacheDir
    $env:APPDATA = $AppDataDir
    $env:LOCALAPPDATA = $LocalAppDataDir

    Invoke-Checked "acceptance: installed rocm version" $rocmExe @("version")
    Invoke-Checked "acceptance: installed rocm doctor" $rocmExe @("doctor") $DoctorLog
    if (-not (Select-String -LiteralPath $DoctorLog -SimpleMatch -Pattern $ConfigDir -Quiet)) {
        Fail "installed rocm doctor did not use the isolated config dir"
    }
    if (-not (Select-String -LiteralPath $DoctorLog -SimpleMatch -Pattern $DataDir -Quiet)) {
        Fail "installed rocm doctor did not use the isolated data dir"
    }
    if (-not (Select-String -LiteralPath $DoctorLog -SimpleMatch -Pattern $CacheDir -Quiet)) {
        Fail "installed rocm doctor did not use the isolated cache dir"
    }
    if (Select-String -LiteralPath $DoctorLog -SimpleMatch -Pattern $RealUserRocmDir -Quiet) {
        Fail "installed rocm doctor read the real user rocm state"
    }
    Invoke-Checked "acceptance: installed rocm engines list" $rocmExe @("engines", "list")
    Invoke-Checked "acceptance: installed rocmd status" $rocmdExe @("status")

    Invoke-Checked "acceptance: uninstall from the installed binary" $rocmExe @("uninstall", "--yes", "--keep-config", "--keep-data", "--keep-cache") $UninstallLog

    Assert-Missing (Join-Path $InstallDir "rocmd.exe")
    Assert-Missing (Join-Path $InstallDir "rocm-engine-pytorch.exe")
    Assert-Missing (Join-Path $InstallDir "rocm-engine-llama-cpp.exe")
    Assert-Missing (Join-Path $InstallDir "rocm-engine-lemonade.exe")
    Assert-Missing (Join-Path $InstallDir "rocm-engine-atom.exe")
    Assert-Missing (Join-Path $InstallDir "rocm-engine-vllm.exe")
    Assert-Missing (Join-Path $InstallDir "rocm-engine-sglang.exe")
    Assert-Missing (Join-Path $InstallDir ".rocm-cli-manifest")
    if (-not (Select-String -LiteralPath $UninstallLog -Pattern "skipping running executable on Windows" -Quiet)) {
        Fail "uninstall did not report the expected running executable skip"
    }

    Write-Host "acceptance: ok"
} finally {
    if (Test-Path -LiteralPath $AcceptanceRoot) {
        Remove-Item -LiteralPath $AcceptanceRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
