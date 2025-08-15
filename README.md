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

 ## Features
 - 9:16 vertical video (1080x1920)
 - TTS via gTTS (pluggable)
 - Pexels stock B‑roll or local fallback
 - Title overlay and music ducking
+ - Subtitles: sidecar SRT, optional hard burn‑in (ffmpeg)

 ## Usage
 - Generate from topic:  
   `python faceless_maker.py make --topic "5 tips to sleep better" --length 60`
 - Use your own script:  
   `python faceless_maker.py make --script-file script.txt`
 - Force local B‑roll only:  
   `python faceless_maker.py make --topic "Space facts" --no-pexels`
+ - With subtitles:
+   - Sidecar SRT (default): `python faceless_maker.py make --topic "Space facts" --subs srt`
+   - Burn subtitles into video: `python faceless_maker.py make --topic "Space facts" --subs burn`
+   - Keep both: `python faceless_maker.py make --topic "Space facts" --subs both`
+
+### Subtitles notes
+- Uses faster-whisper; the model downloads on first run (set `WHISPER_MODEL` in `.env`).
+- Burning requires ffmpeg built with libass (most standard builds include it).
+- Language hint uses `DEFAULT_LANG` (e.g., `en`, `es`). We normalize `en-US` → `en`.

## Notes
- Add an MP3 to `assets/music/` for background music (optional).
- Place any fonts in `assets/fonts/` and point `--font` to a .ttf if you like.
- This is an MVP. Swap in your preferred TTS (e.g., ElevenLabs, Amazon Polly), add Whisper for perfect subtitles, and wire up uploaders (YouTube/TikTok) as you grow it.

## License
MIT
