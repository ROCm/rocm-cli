# PSScriptAnalyzer configuration for the repo's PowerShell scripts
# (install.ps1, scripts/*.ps1). Used by the prek hook and CI.
@{
    # Enforce the full default rule set at Warning and Error severity.
    Severity = @('Error', 'Warning')

    # install.ps1 and the acceptance/packaging scripts are user-facing console
    # programs where `Write-Host` is the intended way to print progress to the
    # terminal, so the "avoid Write-Host" rule does not apply here.
    ExcludeRules = @('PSAvoidUsingWriteHost')
}
