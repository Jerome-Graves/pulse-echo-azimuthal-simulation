# Creates "Pulse-Echo Azimuthal Simulation.lnk" in the repository root:
# a double-click shortcut to the GUI launcher carrying the project icon.
# Called by "Run Simulation GUI.bat" whenever the shortcut is missing;
# shortcuts store absolute paths, so it is rebuilt per machine and
# excluded from version control.

$ErrorActionPreference = "Stop"

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $Here

$Shell = New-Object -ComObject WScript.Shell
$Lnk = $Shell.CreateShortcut((Join-Path $Root "Pulse-Echo Azimuthal Simulation.lnk"))
$Lnk.TargetPath = $env:ComSpec
$Lnk.Arguments = '/c ""' + (Join-Path $Root "Run Simulation GUI.bat") + '""'
$Lnk.WorkingDirectory = $Root
$Lnk.IconLocation = (Join-Path $Here "icon.ico") + ",0"
$Lnk.Description = "Launch the Pulse-Echo Azimuthal Simulation GUI"
$Lnk.Save()

Write-Host "Shortcut created."
