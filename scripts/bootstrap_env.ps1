param(
    [string]$Python = "python",
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

Write-Host "Creating virtual environment at $VenvPath"
& $Python -m venv $VenvPath

$venvPython = Join-Path $VenvPath "Scripts\python.exe"

Write-Host "Upgrading pip"
& $venvPython -m pip install --upgrade pip

Write-Host "Installing project dependencies"
& $venvPython -m pip install -r requirements.txt

Write-Host "Verifying environment"
& $venvPython scripts/check_environment.py

Write-Host ""
Write-Host "Environment ready."
Write-Host "Activate with: .\$VenvPath\Scripts\Activate.ps1"
