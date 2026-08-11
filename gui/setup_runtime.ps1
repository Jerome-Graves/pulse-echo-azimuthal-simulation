# Installs a self-contained Python runtime for the simulation GUI into
# .\runtime so the application runs with a double click, without any
# system-wide Python installation. Called by "Run Simulation GUI.bat" on
# first launch; safe to re-run (rebuilds the runtime from scratch).
#
# Package versions follow the repository's pinned requirements.txt (the
# bit-exact reproduction contract for the paper's analysis gates), with
# the GUI extras (streamlit, plotly) on top. CUDA/cupy for new solver
# runs is deliberately not installed here.

$ErrorActionPreference = "Stop"

$PyVersion = "3.12.8"
$PyZipUrl = "https://www.python.org/ftp/python/$PyVersion/python-$PyVersion-embed-amd64.zip"
$GetPipUrl = "https://bootstrap.pypa.io/get-pip.py"

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Split-Path -Parent $Here
$Runtime = Join-Path $Here "runtime"
$PyDir = Join-Path $Runtime "python"
$Python = Join-Path $PyDir "python.exe"

if (Test-Path $Runtime) { Remove-Item -Recurse -Force $Runtime }
New-Item -ItemType Directory -Force $PyDir | Out-Null

Write-Host "Downloading Python $PyVersion (embeddable)..."
$Zip = Join-Path $Runtime "python-embed.zip"
Invoke-WebRequest -Uri $PyZipUrl -OutFile $Zip
Expand-Archive -Path $Zip -DestinationPath $PyDir
Remove-Item $Zip

# The embeddable distribution ships with site-packages disabled; enable it
# so pip-installed packages are importable.
$Pth = Get-ChildItem $PyDir -Filter "python3*._pth" | Select-Object -First 1
(Get-Content $Pth.FullName) -replace "^#\s*import site", "import site" |
    Set-Content $Pth.FullName -Encoding ascii

Write-Host "Installing pip..."
$GetPip = Join-Path $Runtime "get-pip.py"
Invoke-WebRequest -Uri $GetPipUrl -OutFile $GetPip
& $Python $GetPip --no-warn-script-location
if ($LASTEXITCODE -ne 0) { throw "pip installation failed" }
Remove-Item $GetPip

Write-Host "Installing pinned analysis packages plus the GUI extras..."
& $Python -m pip install --no-warn-script-location `
    -r (Join-Path $Repo "requirements.txt") streamlit plotly
if ($LASTEXITCODE -ne 0) { throw "package installation failed" }

Write-Host "Runtime ready."
