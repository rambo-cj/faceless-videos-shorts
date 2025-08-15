import os
import re
import math
import time
import argparse
import random
import textwrap
import subprocess
from pathlib import Path
from typing import List, Optional, Union, Callable

from dotenv import load_dotenv
import requests
import numpy as np

# MoviePy
from moviepy.editor import (
    VideoFileClip, AudioFileClip, ImageClip, VideoClip,
    concatenate_videoclips, CompositeVideoClip,
    CompositeAudioClip, ColorClip, vfx, afx
)

# TTS
from gtts import gTTS

# Imaging for text overlays
from PIL import Image, ImageDraw, ImageFont

# Audio gen
from pydub import AudioSegment
from pydub.generators import Sine, WhiteNoise
from pydub.effects import low_pass_filter

# Whisper (optional; used for subtitles)
try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except Exception:
    HAS_WHISPER = False

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
# Auto assets: font + music + procedural b-roll
# -----------------------------------------------------------------------------
def ensure_auto_font() -> Optional[Path]:
    fonts = sorted((ASSETS / "fonts").glob("*.ttf"))
    if fonts:
        return fonts[0]
    # Try to download a good open font
    choices = [
        ("Inter-SemiBold.ttf", "https://github.com/google/fonts/raw/main/ofl/inter/Inter-SemiBold.ttf"),
        ("Montserrat-SemiBold.ttf", "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat-SemiBold.ttf"),
        ("Roboto-Bold.ttf", "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Bold.ttf"),
    ]
    for name, url in choices:
        try:
            dst = ASSETS / "fonts" / f"Auto_{name}"
            info(f"Fetching font: {name}")
            download_url(url, dst)
            return dst
        except Exception as e:
            warn(f"Font download failed for {name}: {e}")
    warn("No TTF font available; will use PIL default.")
    return None

def pick_font_fallback() -> Optional[Path]:
    fonts = sorted((ASSETS / "fonts").glob("*.ttf"))
    if fonts:
        return fonts[0]
    return ensure_auto_font()

def hz(note: int) -> float:
    # MIDI note -> Hz
    return 440.0 * (2 ** ((note - 69) / 12.0))

def build_chord(root_midi: int, quality: str = "maj", duration_ms: int = 4000, gain_each_db: float = -18.0) -> AudioSegment:
    # Simple triad: root, third, fifth
    third = 4 if quality == "maj" else 3
    fifth = 7
    freqs = [hz(root_midi), hz(root_midi + third), hz(root_midi + fifth)]
    seg = AudioSegment.silent(duration=duration_ms)
    for f in freqs:
        seg = seg.overlay(Sine(f).to_audio_segment(duration=duration_ms).apply_gain(gain_each_db))
    # gentle fade at segment edges
    seg = seg.fade_in(200).fade_out(200)
    return seg

def ensure_auto_music(min_seconds: int = 18) -> Optional[Path]:
    # If music exists, keep it
    existing = sorted((ASSETS / "music").glob("*.mp3"))
    if existing:
        return existing[0]
    try:
        info("Auto-generating background music...")
        tempo = random.randint(72, 92)  # BPM
        beat_ms = int(60000 / tempo)
        bar_ms = beat_ms * 4
        # pick a key
        roots = {"C":60, "D":62, "E":64, "F":65, "G":67, "A":69, "B":71}
        key = random.choice(list(roots.keys()))
        major = random.random() < 0.65
        root_midi = roots[key] + (random.choice([0,12]))  # octave 4 or 5
        # simple progression
        degrees = [0, 5, 9 if major else 8, 7]  # I - V - vi/VI - IV
        prog = [root_midi + d for d in degrees]
        segs = []
        for p in prog:
            segs.append(build_chord(p, "maj" if major else "min", duration_ms=bar_ms*2, gain_each_db=-20))
        music = sum(segs)
        # add soft bass drone
        bass = Sine(hz(root_midi-24)).to_audio_segment(duration=len(music)).apply_gain(-24).fade_in(500).fade_out(500)
        music = music.overlay(bass)
        # subtle noise texture
        noise = WhiteNoise().to_audio_segment(duration=len(music)).apply_gain(-36)
        noise = low_pass_filter(noise, 2000)
        music = music.overlay(noise)
        # normalize-ish for a bed
        target_dbfs = -20.0
        change = target_dbfs - music.dBFS
        music = music.apply_gain(change)
        out = ASSETS / "music" / "autogen_lofi.mp3"
        music.export(out, format="mp3")
        info(f"Generated music: {out}")
        return out
    except Exception as e:
        warn(f"Music autogen failed: {e}")
        return None

def seeded_random(text: str) -> random.Random:
    return random.Random(abs(hash(text)) % (2**32))

def palette_from_text(text: str) -> List[tuple]:
    rnd = seeded_random(text)
    # Generate 3-4 pleasing colors
    base_h = rnd.random()
    def hsl_to_rgb(h, s, l):
        import colorsys
        r, g, b = colorsys.hls_to_rgb(h % 1.0, l, s)
        return int(r*255), int(g*255), int(b*255)
    cols = [
        hsl_to_rgb(base_h, 0.5 + 0.2*rnd.random(), 0.35 + 0.2*rnd.random()),
        hsl_to_rgb(base_h + 0.12, 0.5 + 0.2*rnd.random(), 0.45 + 0.1*rnd.random()),
        hsl_to_rgb(base_h + 0.24, 0.5 + 0.2*rnd.random(), 0.55 + 0.1*rnd.random()),
        hsl_to_rgb(base_h + 0.36, 0.4 + 0.2*rnd.random(), 0.30 + 0.2*rnd.random())
    ]
    return cols

def gradient_image(w: int, h: int, c1: tuple, c2: tuple, diagonal: bool = True) -> np.ndarray:
    # Simple linear gradient between two colors
    if diagonal:
        x = np.linspace(0, 1, w)[None, :]
        y = np.linspace(0, 1, h)[:, None]
        t = (x + y) / 2.0
    else:
        t = np.tile(np.linspace(0, 1, w)[None, :], (h, 1))
    c1 = np.array(c1).reshape(1,1,3).astype(np.float32)
    c2 = np.array(c2).reshape(1,1,3).astype(np.float32)
    t3 = t[..., None]
    img = (1 - t3) * c1 + t3 * c2
    return np.clip(img, 0, 255).astype(np.uint8)

def shapes_overlay(img: Image.Image, palette: List[tuple], rnd: random.Random, n: int = 6) -> Image.Image:
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    for _ in range(n):
        c = random.choice(palette)
        alpha = rnd.randint(40, 110)
        x1 = rnd.randint(0, w-1)
        y1 = rnd.randint(0, h-1)
        x2 = rnd.randint(x1, min(w, x1 + rnd.randint(w//8, w//3)))
        y2 = rnd.randint(y1, min(h, y1 + rnd.randint(h//10, h//3)))
        if rnd.random() < 0.5:
            draw.ellipse([x1, y1, x2, y2], fill=(c[0], c[1], c[2], alpha))
        else:
            draw.rectangle([x1, y1, x2, y2], fill=(c[0], c[1], c[2], alpha))
    return img

def make_proc_image(topic_text: str, w: int = TARGET_W, h: int = TARGET_H) -> np.ndarray:
    rnd = seeded_random(topic_text + str(random.random()))
    pal = palette_from_text(topic_text)
    base = gradient_image(w, h, pal[0], pal[1], diagonal=True)
    img = Image.fromarray(base)
    img = shapes_overlay(img, pal, rnd, n=rnd.randint(4, 8))
    # add subtle grain
    arr = np.array(img).astype(np.float32)
    noise = (np.random.randn(h, w, 1) * 2.0).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return arr

def ken_burns_from_image(img_np: np.ndarray, duration: float) -> VideoClip:
    # Create a gentle zoom + slight pan from a static image
    base = ImageClip(img_np).set_duration(duration)
    # Up-scale a bit and crop pan
    zoom = 1.06
    base = base.resize(lambda t: 1.0 + (zoom - 1.0) * (t / max(0.001, duration)))
    # Add a very subtle left-right pan by cropping a slightly bigger frame
    W, H = base.w, base.h
    if W < TARGET_W or H < TARGET_H:
        base = base.resize(height=TARGET_H)
    W, H = base.w, base.h
    pan_px = int(max(12, W * 0.01))
    x_start = (W // 2) - pan_px
    x_end = (W // 2) + pan_px
    def x_center(t):
        return x_start + (x_end - x_start) * (t / max(0.001, duration))
    kb = vfx.crop(base, width=TARGET_W, height=TARGET_H, x_center=x_center, y_center=H/2)
    return kb

def generate_procedural_broll_clips(topic_text: str, want: int = 8) -> List[VideoClip]:
    info(f"Generating procedural b-roll (no assets/API)...")
    clips = []
    for i in range(want):
        dur = random.uniform(3.0, 5.0)
        img = make_proc_image(topic_text + f"_{i}", TARGET_W, TARGET_H)
        clip = ken_burns_from_image(img, duration=dur)
        # subtle saturation tweak: overlay with transparent color or leave as is
        clips.append(clip)
    return clips

# -----------------------------------------------------------------------------
# Visuals and composition
# -----------------------------------------------------------------------------
def ensure_vertical(clip: VideoClip) -> VideoClip:
    # Resize and center-crop to 1080x1920
    clip = clip.resize(height=TARGET_H)
    if clip.w >= TARGET_W:
        x_center = clip.w / 2
        clip = clip.crop(x_center=x_center, y_center=clip.h/2, width=TARGET_W, height=TARGET_H)
    else:
        bg = ColorClip(size=(TARGET_W, TARGET_H), color=(0, 0, 0)).set_duration(clip.duration)
        clip = CompositeVideoClip([bg, clip.set_position(("center", "center"))]).set_duration(clip.duration)
    return clip

def assemble_broll(sources: List[Union[Path, str, VideoClip]], target_duration: float) -> VideoClip:
    if not sources:
        raise RuntimeError("No b-roll sources provided.")
    # Normalize sources into clips
    raw: List[VideoClip] = []
    for s in sources:
        if isinstance(s, (str, Path)):
            raw.append(VideoFileClip(str(s)).without_audio())
        else:
            raw.append(s)
    segments = []
    remaining = target_duration
    i = 0
    while remaining > 0 and i < 400:
        src = raw[i % len(raw)]
        seg_len = max(2.0, min(4.5, remaining))
        if getattr(src, "duration", 0) <= seg_len + 0.2:
            sub = src
        else:
            start = random.uniform(0, max(0, src.duration - seg_len))
            sub = src.subclip(start, start + seg_len)
        sub = ensure_vertical(sub)
        # mild motion (zoom/pan) for videos only (ImageClips already have)
        if not isinstance(sub, ImageClip) and random.random() < 0.7:
            z = random.uniform(1.02, 1.08)
            sub = sub.fx(vfx.resize, newsize=(int(sub.w*z), int(sub.h*z))).fx(
                vfx.crop, width=TARGET_W, height=TARGET_H, x_center=sub.w*z/2, y_center=sub.h*z/2
            )
        segments.append(sub)
        remaining -= sub.duration
        i += 1
    seq = concatenate_videoclips(segments, method="compose")
    overlay = ColorClip((TARGET_W, TARGET_H), color=(0, 0, 0)).set_opacity(0.12).set_duration(seq.duration)
    seq = CompositeVideoClip([seq, overlay]).set_duration(seq.duration)
    return seq

def text_image_clip(text: str, width: int, font_path: Optional[Path], font_size: int, color=(255,255,255), stroke=3, stroke_color=(0,0,0), align="center", padding=20, bg=None, duration=2.5):
    lines = []
    for para in text.split("\n"):
        lines.extend(textwrap.wrap(para, width=32))
    font = ImageFont.truetype(str(font_path), font_size) if font_path and font_path.exists() else ImageFont.load_default()
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
    img = Image.new("RGBA", (w, h), (0,0,0,0) if bg is None else bg)
    draw = ImageDraw.Draw(img)
    y = padding
    for idx, line in enumerate(lines):
        bbox = draw.textbbox((0,0), line, font=font, stroke_width=stroke)
        tw = bbox[2]-bbox[0]
        x = padding + (w - 2*padding - tw)//2 if align == "center" else padding
        draw.text((x, y), line, font=font, fill=color, stroke_width=stroke, stroke_fill=stroke_color)
        y += line_heights[idx] + 8
    frame = Image.new("RGB", (TARGET_W, TARGET_H), (0,0,0))
    fx = (TARGET_W - w)//2
    fy = int(TARGET_H*0.08)
    frame.paste(img, (fx, fy), img)
    return ImageClip(np.array(frame)).set_duration(duration)

# -----------------------------------------------------------------------------
# Audio helpers
# -----------------------------------------------------------------------------
def pick_music() -> Optional[Path]:
    # Ensure at least one music bed exists
    ensure_auto_music(min_seconds=20)
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
# Subtitles (Whisper + SRT + optional burn-in via ffmpeg)
# -----------------------------------------------------------------------------
def normalize_lang(lang: Optional[str]) -> Optional[str]:
    if not lang:
        return None
    lang = lang.lower().strip()
    for sep in ("-", "_"):
        if sep in lang:
            return lang.split(sep)[0]
    return lang[:2] if len(lang) > 2 else lang

def transcribe_with_whisper(audio_path: Path, lang: Optional[str]) -> Optional[List[dict]]:
    if not HAS_WHISPER:
        warn("faster-whisper not installed; using rough subtitles fallback.")
        return None
    model_name = os.getenv("WHISPER_MODEL", "small")
    device = os.getenv("WHISPER_DEVICE", "cpu")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", None)
    if compute_type is None:
        compute_type = "int8" if device == "cpu" else "float16"
    info(f"Transcribing with whisper model '{model_name}' on {device} ({compute_type})...")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, _ = model.transcribe(str(audio_path), language=normalize_lang(lang), vad_filter=True, beam_size=1)
    out = []
    for seg in segments:
        out.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()})
    if not out:
        return None
    return out

def rough_subs_from_text(text: str, total_duration: float) -> List[dict]:
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    if not sents:
        sents = [text.strip()]
    counts = [max(1, len(s)) for s in sents]
    total_chars = sum(counts)
    segments = []
    t = 0.0
    for i, s in enumerate(sents):
        frac = counts[i] / total_chars
        dur = max(1.2, total_duration * frac)
        start = t
        end = total_duration if i == len(sents) - 1 else min(total_duration, t + dur)
        segments.append({"start": float(start), "end": float(end), "text": s})
        t = end
        if t >= total_duration:
            break
    merged = []
    for seg in segments:
        if merged and (seg["end"] - seg["start"]) < 0.7:
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] += " " + seg["text"]
        else:
            merged.append(seg)
    return merged

def _srt_ts(t: float) -> str:
    if t < 0: t = 0.0
    ms = int(round((t - math.floor(t)) * 1000))
    s = int(t) % 60
    m = (int(t) // 60) % 60
    h = int(t) // 3600
    if ms == 1000:
        ms = 0
        s += 1
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def write_srt(segments: List[dict], out_path: Path, total_duration: Optional[float] = None):
    lines = []
    for i, seg in enumerate(segments, 1):
        start = max(0.0, float(seg["start"]))
        end = float(seg["end"])
        if total_duration is not None:
            end = min(end, total_duration)
        if end <= start:
            end = start + 0.5
        text = seg["text"].strip()
        lines.append(str(i))
        lines.append(f"{_srt_ts(start)} --> {_srt_ts(end)}")
        lines.append(text)
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    info(f"Wrote subtitles: {out_path}")

def _escape_for_ffmpeg_subtitles(p: Path) -> str:
    s = str(Path(p).resolve())
    s = s.replace("\\", "\\\\").replace(":", r"\:")
    return s

def burn_subtitles_ffmpeg(in_video: Path, srt_path: Path, out_video: Path, font_name: Optional[str] = None, font_size: int = 42, margin_v: int = 80):
    style_parts = []
    if font_name:
        style_parts.append(f"FontName={font_name}")
    style_parts.extend([
        f"FontSize={font_size}",
        "BorderStyle=3",
        "Outline=2",
        "Shadow=0",
        f"MarginV={margin_v}"
    ])
    style = ",".join(style_parts)
    sub_arg = f"subtitles={_escape_for_ffmpeg_subtitles(srt_path)}:force_style={style}"
    cmd = [
        "ffmpeg", "-y", "-i", str(in_video),
        "-vf", sub_arg,
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(out_video)
    ]
    info(f"Burning subtitles with ffmpeg → {out_video}")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        warn("ffmpeg burn-in failed; keeping sidecar SRT only.")
        warn(e.stderr.decode(errors="ignore")[:800])

def create_subtitles(voice_mp3: Path, srt_out: Path, lang: str, fallback_text: str, total_duration: float):
    segments = transcribe_with_whisper(voice_mp3, lang)
    if not segments:
        segments = rough_subs_from_text(fallback_text, total_duration)
    cleaned = []
    last_end = 0.0
    for seg in segments:
        start = max(last_end, float(seg["start"]))
        end = max(start + 0.3, float(seg["end"]))
        cleaned.append({"start": start, "end": end, "text": seg["text"].strip()})
        last_end = end
    write_srt(cleaned, srt_out, total_duration=total_duration)
    return cleaned

# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------
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
    base = keys[:6]
    filler = ["abstract", "bokeh", "city night", "clouds", "nature", "office", "technology", "patterns", "gradient"]
    base.extend(random.sample(filler, k=min(6, len(filler))))
    return base[:10]

def make_video(topic: Optional[str],
               script_text: Optional[str],
               length_hint: Optional[int],
               use_pexels: bool,
               lang: str,
               font: Optional[str],
               subs_mode: str = "srt") -> Path:
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

    # 3) B‑roll sources (Pexels → local → procedural)
    keywords = extract_keywords(title + " " + body)
    api_key = os.getenv("PEXELS_API_KEY")
    sources: List[Union[Path, VideoClip]] = []
    if use_pexels and api_key:
        paths = fetch_broll_from_pexels(keywords, api_key, want=8)
        sources.extend(paths)
    if not sources:
        local = local_broll_paths()
        if local:
            info(f"Using {len(local)} local b‑roll file(s)")
            sources.extend(local)
    if not sources:
        # Procedural generation
        sources.extend(generate_procedural_broll_clips(" ".join(keywords), want=8))

    # 4) Visual assembly
    seq = assemble_broll(sources, target_duration)
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

    # 8) Subtitles (SRT + optional burn)
    if subs_mode in ("srt", "burn", "both"):
        srt_path = out_path.with_suffix(".srt")
        try:
            create_subtitles(voice_mp3, srt_path, lang, full_narration, target_duration)
            if subs_mode in ("burn", "both"):
                burned = out_path.with_name(out_path.stem + "_subbed" + out_path.suffix)
                font_name = None
                if font:
                    font_name = Path(font).stem
                burn_subtitles_ffmpeg(out_path, srt_path, burned, font_name=font_name, font_size=42, margin_v=80)
                info("Done.")
                return burned if subs_mode == "burn" else out_path
        except Exception as e:
            warn(f"Subtitles generation failed: {e}")
    info("Done.")
    return out_path

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
    mk.add_argument("--no-pexels", action="store_true", help="Disable Pexels and use local/procedural assets")
    mk.add_argument("--lang", type=str, default=os.getenv("DEFAULT_LANG", "en"), help="gTTS language code, e.g., en")
    mk.add_argument("--font", type=str, default=None, help="Path to a .ttf font for title")
    mk.add_argument("--subs", type=str, choices=["none","srt","burn","both"], default="srt", help="Generate subtitles: srt (default), burn (hard), both, or none")

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
            make_video(topic, script_text, args.length, use_pexels=not args.no_pexels, lang=args.lang, font=args.font, subs_mode=args.subs)
        except Exception as e:
            err(str(e))
            raise SystemExit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
