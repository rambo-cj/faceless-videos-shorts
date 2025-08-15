#!/usr/bin/env bash
# Sets up a Python venv, installs requirements, checks/installs FFmpeg,
# and creates a .env from .env.example if missing.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

have() { command -v "$1" >/dev/null 2>&1; }

echo "==> Checking Python (3.10+)..."
PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
  if have python3; then PYTHON="python3"
  elif have python; then PYTHON="python"
  else
    echo "Python not found. Install Python 3.10+ and re-run." >&2
    exit 1
  fi
fi

if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
  echo "Python 3.10+ required. Current: $("$PYTHON" -c "import sys; print(sys.version.split()[0])")" >&2
  exit 1
fi

echo "==> Creating virtual environment: .venv"
"$PYTHON" -m venv .venv

# Resolve venv python/pip paths for both POSIX and Windows layouts
PIP=".venv/bin/pip"
PYV=".venv/bin/python"
if [[ -f ".venv/Scripts/pip.exe" ]]; then
  PIP=".venv/Scripts/pip.exe"
  PYV=".venv/Scripts/python.exe"
fi

echo "==> Upgrading pip/setuptools/wheel"
"$PYV" -m pip install --upgrade pip setuptools wheel

echo "==> Installing Python requirements"
"$PYV" -m pip install -r requirements.txt

echo "==> Checking FFmpeg"
if have ffmpeg; then
  echo "FFmpeg found: $(ffmpeg -version | head -n1)"
else
  echo "FFmpeg not found."
  if [[ "${NO_FFMPEG_INSTALL:-0}" == "1" ]]; then
    echo "Skipping FFmpeg install (NO_FFMPEG_INSTALL=1). Please install it manually."
  else
    case "$(uname -s)" in
      Darwin)
        if have brew; then
          echo "Installing FFmpeg via Homebrew..."
          brew install ffmpeg
        else
          echo "Homebrew not found. Install Homebrew (https://brew.sh) then run: brew install ffmpeg"
        fi
        ;;
      Linux)
        if have apt-get; then
          echo "Installing FFmpeg via apt..."
          sudo apt-get update
          sudo apt-get install -y ffmpeg
        elif have dnf; then
          echo "Installing FFmpeg via dnf..."
          sudo dnf install -y ffmpeg
        elif have pacman; then
          echo "Installing FFmpeg via pacman..."
          sudo pacman -Sy --noconfirm ffmpeg
        else
          echo "Could not auto-install FFmpeg. Please use your distro's package manager."
        fi
        ;;
      *)
        echo "Unknown platform. Please install FFmpeg manually: https://ffmpeg.org/download.html"
        ;;
    esac
  fi
fi

# Ensure .env exists
if [[ ! -f ".env" && -f ".env.example" ]]; then
  cp .env.example .env
  echo "==> Created .env from .env.example"
fi

# Quick import test (optional)
echo "==> Verifying imports"
"$PYV" - <<'PY'
mods = ["moviepy", "gtts", "PIL", "requests"]
failed = []
for m in mods:
    try:
        __import__(m)
    except Exception as e:
        failed.append((m, str(e)))
try:
    __import__("faster_whisper")
    print("[i] faster-whisper import OK.")
except Exception as e:
    print("[!] faster-whisper import warning:", e)
if failed:
    raise SystemExit("Import errors: " + ", ".join([f"{m}: {e}" for m,e in failed]))
print("[i] All core imports OK.")
PY

echo
echo "✅ Setup complete."
echo
if [[ -f ".venv/bin/activate" ]]; then
  echo "Activate your venv:"
  echo "  source .venv/bin/activate"
else
  echo "Activate your venv (Windows layout detected):"
  echo "  source .venv/Scripts/activate"
fi
echo "Then run:"
echo "  python faceless_maker.py make --topic \"AI facts\" --subs srt"
