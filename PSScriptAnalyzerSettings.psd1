# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# PSScriptAnalyzer configuration for the repo's PowerShell scripts: every tracked
# .ps1 file, as listed by `git ls-files '*.ps1'` in xtask/src/powershell.rs
# (today install.ps1 and .github/scripts/Invoke-WslBash.ps1). Used by the prek
# hook and CI.
#
# Only .ps1 files are linted. The licence-header check (licenserc.toml) also
# covers .psm1 and .psd1, so a module or data file added later gets a header but
# no analyzer pass until the xtask and the hook's `files:` pattern are extended.
@{
    # Enforce the full default rule set at Warning and Error severity.
    Severity = @('Error', 'Warning')

    # install.ps1 is a user-facing console program where `Write-Host` is the
    # intended way to print progress to the terminal, so the "avoid Write-Host"
    # rule does not apply here.
    ExcludeRules = @('PSAvoidUsingWriteHost')
}
