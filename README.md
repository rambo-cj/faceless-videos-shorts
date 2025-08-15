# Faceless Video Studio (MVP)

Generate faceless vertical videos/shorts from a topic or a text script:
- Script → TTS → B‑roll (Pexels/local) → Music → 9:16 render → Subtitles (SRT or burned in)

## Features
- 9:16 vertical video (1080x1920) with light motion and readability overlay
- TTS via gTTS (pluggable)
- Pexels stock B‑roll (optional) or local fallback
- Title overlay and background music with ducking
- Subtitles: sidecar SRT and optional hard burn‑in (ffmpeg)

## Setup
1) Install FFmpeg (make sure your build includes `libass` for subtitle burn‑in).
   - macOS: `brew install ffmpeg`
   - Windows: `choco install ffmpeg`
   - Linux (Debian/Ubuntu): `sudo apt-get install ffmpeg`
2) Python 3.10+ recommended.
3) Install deps: `pip install -r requirements.txt`
4) Copy `.env.example` → `.env` and set any keys (optional but recommended).

## Environment (.env)

## Notes
- Add an MP3 to `assets/music/` for background music (optional).
- Place any fonts in `assets/fonts/` and point `--font` to a .ttf if you like.
- This is an MVP. Swap in your preferred TTS (e.g., ElevenLabs, Amazon Polly), add Whisper for perfect subtitles, and wire up uploaders (YouTube/TikTok) as you grow it.

## License
MIT
