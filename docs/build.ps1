# Builds the simulation manual (simulation-manual.pdf) with XeLaTeX.
# Two passes so the table of contents and cross-references resolve.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

foreach ($pass in 1, 2) {
    Write-Host "XeLaTeX pass $pass..."
    xelatex -interaction=nonstopmode -halt-on-error simulation-manual.tex | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Get-Content simulation-manual.log -Tail 30
        throw "XeLaTeX failed on pass $pass (see simulation-manual.log)"
    }
}
Write-Host "Built simulation-manual.pdf"
