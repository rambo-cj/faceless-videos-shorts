# Faceless Video Studio (MVP)

Generate faceless vertical videos/shorts from a topic or a text script:
- Script → TTS → B‑roll (Pexels/local) → Music → 9:16 render.

## Features
- 9:16 vertical video (1080x1920)
- TTS via gTTS (pluggable)
- Pexels stock B‑roll or local fallback
- Title overlay and music ducking

## Setup
1) Install FFmpeg and Python 3.10+
2) `pip install -r requirements.txt`
3) Copy `.env.example` → `.env`, set any keys (optional but recommended)

## Usage
- Generate from topic:  
  `python faceless_maker.py make --topic "5 tips to sleep better" --length 60`
- Use your own script:  
  `python faceless_maker.py make --script-file script.txt`
- Force local B‑roll only:  
  `python faceless_maker.py make --topic "Space facts" --no-pexels`

Outputs to `out/` with timestamped name.

## Environment (.env)
