# One-command dev setup for Windows (PowerShell).
#   ./setup.ps1
#
# Creates a local .venv, installs embench editable (API backends + dev tools),
# and seeds a .env from .env.example if you don't have one yet. To also work on
# the local backend (PyTorch), run afterwards:  pip install -e ".[all]"
$ErrorActionPreference = "Stop"

$python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

Write-Host "Creating virtual environment in .venv ..."
& $python -m venv .venv

$venvPy = Join-Path ".venv" "Scripts\python.exe"
& $venvPy -m pip install --upgrade pip

Write-Host "Installing embench (core + API backends + dev tools) ..."
& $venvPy -m pip install -e ".[dev]"

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example - add your API keys."
}

Write-Host ""
Write-Host "Done. Next:"
Write-Host "  .venv\Scripts\Activate.ps1     # activate the environment"
Write-Host "  pytest                          # run the tests"
Write-Host "  embench run -m dummy:256 --retrieval sample_data\retrieval.json"
