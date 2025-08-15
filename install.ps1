# Sets up a Python venv, installs requirements, checks/installs FFmpeg,
# and creates a .env from .env.example if missing.
$ErrorActionPreference = 'Stop'

# Move to repo root
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

Write-Host "==> Checking Python (3.10+)..."
$pyCmd = (Get-Command py -ErrorAction SilentlyContinue)
if ($pyCmd) { $PY = "py" }
elseif (Get-Command python -ErrorAction SilentlyContinue) { $PY = "python" }
else { throw "Python not found. Install Python 3.10+ and re-run." }

# Version check
$verOk = & $PY -c "import sys; print('OK' if sys.version_info >= (3,10) else 'NO')" 
if ($verOk -ne "OK") { throw "Python 3.10+ required." }

Write-Host "==> Creating virtual environment: .venv"
& $PY -m venv .venv

$PIP = ".\.venv\Scripts\pip.exe"
$PYV = ".\.venv\Scripts\python.exe"

Write-Host "==> Upgrading pip/setuptools/wheel"
& $PYV -m pip install --upgrade pip setuptools wheel

Write-Host "==> Installing Python requirements"
& $PYV -m pip install -r requirements.txt

Write-Host "==> Checking FFmpeg"
$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if ($ffmpeg) {
  $v = (& ffmpeg -version | Select-Object -First 1)
  Write-Host "FFmpeg found: $v"
} else {
  Write-Warning "FFmpeg not found."
  if ($env:NO_FFMPEG_INSTALL -eq "1") {
    Write-Host "Skipping FFmpeg install (NO_FFMPEG_INSTALL=1). Please install it manually."
  } else {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
      Write-Host "Installing FFmpeg via winget..."
      winget install -e --id Gyan.FFmpeg --source winget
    } elseif (Get-Command choco -ErrorAction SilentlyContinue) {
      Write-Host "Installing FFmpeg via Chocolatey..."
      choco install ffmpeg -y
    } else {
      Write-Warning "Couldn't auto-install FFmpeg. Install it manually: https://www.gyan.dev/ffmpeg/builds/"
    }
  }
}

# Ensure .env exists
if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
  Copy-Item ".env.example" ".env"
  Write-Host "==> Created .env from .env.example"
}

# Quick import test (optional)
Write-Host "==> Verifying imports"
& $PYV -c "import moviepy, gtts, PIL, requests; print('[i] Core imports OK.'); \
try: import faster_whisper; print('[i] faster-whisper import OK.'); \
except Exception as e: print('[!] faster-whisper import warning:', e)"

Write-Host ""
Write-Host "✅ Setup complete."
Write-Host ""
Write-Host "Activate your venv:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "Then run:"
Write-Host "  python faceless_maker.py make --topic 'AI facts' --subs srt"
