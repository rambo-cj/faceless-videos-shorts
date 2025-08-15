import os
import re
import math
import json
import time
import argparse
import random
import textwrap
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
import requests

# MoviePy
from moviepy.editor import (
    VideoFileClip, AudioFileClip, ImageClip,
    concatenate_videoclips, CompositeVideoClip,
    CompositeAudioClip, ColorClip, vfx, afx
)

# TTS
from gtts import gTTS

# Imaging for text overlays
from PIL import Image, ImageDraw, ImageFont

# -----------------------------------------------------------------------------
# Paths and constants
# -----------------------------------------------------------------------------
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)
(ASSETS / "broll").mkdir(parents=True, exist_ok=True)
(ASSETS / "music").mkdir(parents=True, exist_ok=True)
(ASSETS / "fonts").mkdir(parents=True, exist_ok=True)
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)
CACHE = ROOT / ".cache"
CACHE.mkdir(exist_ok=True)

TARGET_W, TARGET_H = 1080, 1920
FPS = 30

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def ts():
    return time.strftime("%Y%m%d_%H%M%S")

def sanitize_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", s).strip("_")[:80]

def info(msg): print(f"[i] {msg}")
def warn(msg): print(f"[!] {msg}")
def err(msg): print(f"[x] {msg}")

def seconds_from_length_hint(length_hint: Optional[int]) -> int:
    if length_hint and 10 <= length_hint <= 180:
        return length_hint
    return 60

# -----------------------------------------------------------------------------
# Script generation (simple template; plug your LLM here)
# -----------------------------------------------------------------------------
def generate_script(topic: str, length_sec: int) -> dict:
    # Simple heuristic: N bullets based on length
    n = max(3, min(8, length_sec // 10))
    bullets = [
        f"{i+1}. {line}"
        for i, line in enumerate(make_bullets(topic, n))
    ]
    title = title_from_topic(topic)
    return {
        "title": title,
        "body": "\n".join(bullets)
    }

def title_from_topic(topic: str) -> str:
    topic = topic.strip().rstrip(".")
    if not topic:
        return "Quick Tips"
    # Make it punchy
    starters = ["5 Tips for", "Quick Guide to", "Smart Ways to", "What No One Tells You About"]
    start = random.choice(starters)
    return f"{start} {topic}"

def make_bullets(topic: str, n: int) -> List[str]:
    seeds = [
        f"Keep it simple: one action at a time related to {topic.lower()}",
        f"Use a timer: 20–25 minutes of focused work helps with {topic.lower()}",
        f"Remove friction: prepare essentials ahead to boost {topic.lower()}",
        f"Track one metric that matters for {topic.lower()}",
        f"Small daily reps beat big irregular pushes in {topic.lower()}",
        f"Batch similar tasks to stay in flow for {topic.lower()}",
        f"Set a tiny 'minimum' so you never skip {topic.lower()}",
        f"Reflect weekly: what's one tweak to improve {topic.lower()}?"
    ]
    random.shuffle(seeds)
    return seeds[:n]

# -----------------------------------------------------------------------------
# TTS (gTTS)
# -----------------------------------------------------------------------------
def synth_tts_gtts(text: str, lang: str, out_path: Path) -> Path:
    info("Synthesizing voiceover with gTTS...")
    tts = gTTS(text=text, lang=lang)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tts.save(str(out_path))
    return out_path

# -----------------------------------------------------------------------------
# Pexels B-roll (optional)
# -----------------------------------------------------------------------------
def pexels_search_videos(query: str, per_page=10, api_key: Optional[str]=None) -> List[dict]:
    if not api_key:
        return []
    url = "https://api.pexels.com/videos/search"
    params = {"query": query, "per_page": per_page}
    headers = {"Authorization": api_key}
    r = requests.get(url, params=params, headers=headers, timeout=20)
    if r.status_code != 200:
        warn(f"Pexels API error {r.status_code}: {r.text[:200]}")
        return []
    return r.json().get("videos", [])

def best_video_file(video: dict) -> Optional[dict]:
    # Choose a file near or above 1080p height, fallback to highest
    files = sorted(video.get("video_files", []), key=lambda f: f.get("height", 0), reverse=True)
    if not files:
        return None
    for f in files:
        if f.get("height", 0) >= 1080:
            return f
    return files[0]

def download_url(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1<<16):
                if chunk:
                    f.write(chunk)

def fetch_broll_from_pexels(keywords: List[str], api_key: Optional[str], want=6) -> List[Path]:
    if not api_key:
        return []
    saved = []
    random.shuffle(keywords)
    for kw in keywords:
        vids = pexels_search_videos(kw, per_page=8, api_key=api_key)
        for v in vids:
            vf = best_video_file(v)
            if not vf:
                continue
            link = vf.get("link")
            vid_id = v.get("id")
            ext = ".mp4"
            dst = CACHE / "broll" / f"pexels_{vid_id}{ext}"
            if not dst.exists():
                try:
                    info(f"Downloading b-roll: {kw} (id {vid_id})")
                    download_url(link, dst)
                except Exception as e:
                    warn(f"Failed {kw}/{vid_id}: {e}")
                    continue
            saved.append(dst)
            if len(saved) >= want:
                return saved
    return saved

def local_broll_paths() -> List[Path]:
    return sorted((ASSETS / "broll").glob("*.mp4"))

# -----------------------------------------------------------------------------
# Visuals and composition
# -----------------------------------------------------------------------------
def ensure_vertical(clip: VideoFileClip) -> VideoFileClip:
    # Resize and center-crop to 1080x1920
    # Step 1: resize height to TARGET_H
    clip = clip.resize(height=TARGET_H)
    # Step 2: crop width to TARGET_W (centered) if wider, else pad
    if clip.w >= TARGET_W:
        x_center = clip.w / 2
        clip = clip.crop(x_center=x_center, y_center=clip.h/2, width=TARGET_W, height=TARGET_H)
    else:
        # pad left/right
        pad_w = (TARGET_W - clip.w) // 2
        bg = ColorClip(size=(TARGET_W, TARGET_H), color=(0, 0, 0)).set_duration(clip.duration)
        clip = CompositeVideoClip([bg, clip.set_position(("center", "center"))]).set_duration(clip.duration)
    return clip

def assemble_broll(video_paths: List[Path], target_duration: float) -> VideoFileClip:
    if not video_paths:
        raise RuntimeError("No b-roll videos available. Add files to assets/broll or set PEXELS_API_KEY.")
    # Load clips
    raw = [VideoFileClip(str(p)).without_audio() for p in video_paths]
    # Build a sequence of small segments until we cover target duration
    segments = []
    remaining = target_duration
    i = 0
    while remaining > 0 and i < 400:  # safety
        src = raw[i % len(raw)]
        seg_len = max(2.0, min(4.5, remaining))
        if src.duration <= seg_len + 0.2:
            sub = src
        else:
            start = random.uniform(0, max(0, src.duration - seg_len))
            sub = src.subclip(start, start + seg_len)
        sub = ensure_vertical(sub)
        # mild motion (zoom/pan)
        if random.random() < 0.7:
            z = random.uniform(1.02, 1.08)
            sub = sub.fx(vfx.resize, newsize=(int(sub.w*z), int(sub.h*z))).fx(vfx.crop, width=TARGET_W, height=TARGET_H, x_center=sub.w*z/2, y_center=sub.h*z/2)
        segments.append(sub)
        remaining -= sub.duration
        i += 1
    seq = concatenate_videoclips(segments, method="compose")
    # Darken slightly for readability
    overlay = ColorClip((TARGET_W, TARGET_H), color=(0, 0, 0)).set_opacity(0.12).set_duration(seq.duration)
    seq = CompositeVideoClip([seq, overlay]).set_duration(seq.duration)
    return seq

def text_image_clip(text: str, width: int, font_path: Optional[Path], font_size: int, color=(255,255,255), stroke=3, stroke_color=(0,0,0), align="center", padding=20, bg=None, duration=2.5):
    lines = []
    for para in text.split("\n"):
        lines.extend(textwrap.wrap(para, width=32))
    font = ImageFont.truetype(str(font_path), font_size) if font_path and font_path.exists() else ImageFont.load_default()
    # measure
    tmp = Image.new("RGBA", (width, 10), (0,0,0,0))
    draw = ImageDraw.Draw(tmp)
    w = 0
    h = 0
    line_heights = []
    for line in lines:
        bbox = draw.textbbox((0,0), line, font=font, stroke_width=stroke)
        w = max(w, bbox[2]-bbox[0])
        lh = bbox[3]-bbox[1]
        h += lh + 8
        line_heights.append(lh)
    w = min(width, w + padding*2)
    h = h + padding*2
    if bg is None:
        img = Image.new("RGBA", (w, h), (0,0,0,0))
    else:
        img = Image.new("RGBA", (w, h), bg)
    draw = ImageDraw.Draw(img)
    y = padding
    for idx, line in enumerate(lines):
        bbox = draw.textbbox((0,0), line, font=font, stroke_width=stroke)
        tw = bbox[2]-bbox[0]
        x = padding + (w - 2*padding - tw)//2 if align == "center" else padding
        draw.text((x, y), line, font=font, fill=color, stroke_width=stroke, stroke_fill=stroke_color)
        y += line_heights[idx] + 8
    frame = Image.new("RGB", (TARGET_W, TARGET_H), (0,0,0))
    # place near top
    fx = (TARGET_W - w)//2
    fy = int(TARGET_H*0.08)
    frame.paste(img, (fx, fy), img)
    return ImageClip(frame).set_duration(duration)

# -----------------------------------------------------------------------------
# Audio helpers
# -----------------------------------------------------------------------------
def pick_music() -> Optional[Path]:
    choices = sorted((ASSETS / "music").glob("*.mp3"))
    return random.choice(choices) if choices else None

def build_audio(voice_mp3: Path, music_path: Optional[Path]) -> AudioFileClip:
    voice = AudioFileClip(str(voice_mp3))
    tracks = [voice.volumex(1.0)]
    if music_path and Path(music_path).exists():
        music = AudioFileClip(str(music_path)).fx(afx.audio_loop, duration=voice.duration).volumex(0.10)
        tracks.append(music)
    return CompositeAudioClip(tracks).set_duration(voice.duration)

# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------
def make_video(topic: Optional[str],
               script_text: Optional[str],
               length_hint: Optional[int],
               use_pexels: bool,
               lang: str,
               font: Optional[str]) -> Path:
    # 1) Script
    if script_text:
        lines = [l.strip() for l in script_text.strip().splitlines() if l.strip()]
        title = lines[0][:80] if lines else "Quick Tips"
        body = "\n".join(lines[1:] if len(lines)>1 else lines)
    elif topic:
        scr = generate_script(topic, seconds_from_length_hint(length_hint))
        title, body = scr["title"], scr["body"]
    else:
        raise ValueError("Provide either --topic or --script-file")

    full_narration = f"{title}. " + re.sub(r"\s+", " ", body)
    info(f"Title: {title}")

    # 2) Voiceover
    voice_mp3 = CACHE / "voice" / f"{sanitize_filename(title)}.mp3"
    synth_tts_gtts(full_narration, lang, voice_mp3)
    voice = AudioFileClip(str(voice_mp3))
    target_duration = voice.duration

    # 3) B‑roll
    keywords = extract_keywords(title + " " + body)
    api_key = os.getenv("PEXELS_API_KEY")
    video_paths = []
    if use_pexels and api_key:
        video_paths = fetch_broll_from_pexels(keywords, api_key, want=8)
    if not video_paths:
        local = local_broll_paths()
        if local:
            info(f"Using {len(local)} local b‑roll file(s)")
            video_paths = local
        else:
            raise RuntimeError("No b‑roll available. Either add files to assets/broll or set PEXELS_API_KEY and omit --no-pexels.")

    # 4) Visual assembly
    seq = assemble_broll(video_paths, target_duration)
    # 5) Title overlay (first 2–3 seconds)
    font_path = Path(font) if font else pick_font_fallback()
    title_clip = text_image_clip(title, width=int(TARGET_W*0.9), font_path=font_path, font_size=64, duration=min(3.0, seq.duration))
    video = CompositeVideoClip([seq, title_clip.set_start(0)]).set_duration(seq.duration)

    # 6) Audio mix
    music = pick_music()
    final_audio = build_audio(voice_mp3, music)
    video = video.set_audio(final_audio)

    # 7) Export
    out_name = f"{ts()}_{sanitize_filename(title)}_short.mp4"
    out_path = OUT / out_name
    info(f"Rendering: {out_path}")
    video.write_videofile(
        str(out_path),
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        bitrate="5M",
        threads=4
    )
    info("Done.")
    return out_path

def extract_keywords(text: str) -> List[str]:
    words = [w.lower() for w in re.findall(r"[a-zA-Z]{4,}", text)]
    common = set("this that with from your have will into they them then when what were been being over into more only very just make like much many time tips ways best good".split())
    freq = {}
    for w in words:
        if w in common:
            continue
        freq[w] = freq.get(w, 0) + 1
    keys = [w for w,_ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)]
    if not keys:
        keys = ["abstract", "background", "city", "nature", "technology"]
    # diversify
    base = keys[:6]
    filler = ["abstract", "bokeh", "city night", "clouds", "nature", "office", "technology", "patterns", "gradient"]
    base.extend(random.sample(filler, k=min(6, len(filler))))
    return base[:10]

def pick_font_fallback() -> Optional[Path]:
    # Try a commonly available font (adjust path per OS if desired)
    candidates = list((ASSETS / "fonts").glob("*.ttf"))
    return candidates[0] if candidates else None

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Faceless Shorts Generator (MVP)")
    sub = parser.add_subparsers(dest="cmd")

    mk = sub.add_parser("make", help="Create a faceless short")
    mk.add_argument("--topic", type=str, help="Topic to generate script")
    mk.add_argument("--script-file", type=str, help="Path to a text file to read script from")
    mk.add_argument("--length", type=int, default=60, help="Target length in seconds (hint)")
    mk.add_argument("--no-pexels", action="store_true", help="Disable Pexels and use local assets only")
    mk.add_argument("--lang", type=str, default=os.getenv("DEFAULT_LANG", "en"), help="gTTS language code, e.g., en")
    mk.add_argument("--font", type=str, default=None, help="Path to a .ttf font for title")

    args = parser.parse_args()
    if args.cmd == "make":
        topic = args.topic
        script_text = None
        if args.script_file:
            p = Path(args.script_file)
            if not p.exists():
                err(f"Script file not found: {p}")
                raise SystemExit(2)
            script_text = p.read_text(encoding="utf-8")
        try:
            make_video(topic, script_text, args.length, use_pexels=not args.no_pexels, lang=args.lang, font=args.font)
        except Exception as e:
            err(str(e))
            raise SystemExit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
