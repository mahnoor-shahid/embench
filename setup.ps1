# One-command dev setup for Windows (PowerShell).
#   ./setup.ps1            # core + dev (pytest)
#   ./setup.ps1 all        # every model backend too
#   ./setup.ps1 openai,google
#
# Creates a local .venv, installs embench editable with the chosen extras,
# and seeds a .env from .env.example if you don't have one yet.
param(
    [string]$Extras = "dev"
)
$ErrorActionPreference = "Stop"

$python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

Write-Host "Creating virtual environment in .venv ..."
& $python -m venv .venv

$venvPy = Join-Path ".venv" "Scripts\python.exe"
& $venvPy -m pip install --upgrade pip

# always include dev so tests run; merge with any user-requested extras
$wanted = ($Extras -split ",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
if ($wanted -notcontains "dev") { $wanted += "dev" }
$spec = "-e", ".[$([string]::Join(",", $wanted))]"
Write-Host "Installing embench [$($wanted -join ', ')] ..."
& $venvPy -m pip install @spec

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example - add your API keys."
}

Write-Host ""
Write-Host "Done. Next:"
Write-Host "  .venv\Scripts\Activate.ps1     # activate the environment"
Write-Host "  pytest                          # run the tests"
Write-Host "  embench run -m dummy:256 --retrieval sample_data\retrieval.json"
