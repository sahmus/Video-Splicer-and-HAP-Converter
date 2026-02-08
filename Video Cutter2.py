"""
FFmpeg Video Segmenter UI
Features:
- Cut video into 5 / 10 / 20 second segments (sequential)
- Optional: Randomize clip length by ± jitter seconds
- Optional: Mute audio
- Output modes:
    * Stream copy (fast, may be off-keyframe)
    * Re-encode H.264 (.mp4)
    * Re-encode HAP (.mov) optimized for TouchDesigner playback
- BPM segmentation:
    * Enter BPM
    * Choose beats-per-clip (e.g., 4 beats, 8 beats)
    * Clips are aligned to beat grid
    * Optional length jitter still applies (clamped)
- Random pool generator:
    * Generate N random clips of chosen base durations (or BPM-beat durations)
    * Optional beat-grid alignment (if BPM provided)
- Optional parallel processing (multiple ffmpeg workers)
- Input support:
    * Accepts common containers/codecs supported by your FFmpeg build (mkv, mp4, mov, avi, webm, mpeg, ts, etc.)
    * Fast validation via ffprobe (has video stream)
- NEW: Range selection
    * Choose Start and End time so you segment only a portion of the video
    * Time format: seconds (e.g. 90.5) OR HH:MM:SS(.ms) OR MM:SS(.ms)

Run:
    python video_segmenter_ui.py
"""

import os
import math
import shutil
import random
import threading
import subprocess
from dataclasses import dataclass
from typing import Optional, List, Tuple
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from concurrent.futures import ThreadPoolExecutor, as_completed


# ----------------------------
# Utilities / FFmpeg helpers
# ----------------------------

def which(exe: str) -> Optional[str]:
    return shutil.which(exe)


def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err


def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ffprobe_duration_seconds(input_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nokey=1:noprint_wrappers=1",
        input_path
    ]
    rc, out, err = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"ffprobe failed:\n{err.strip()}")
    try:
        return float(out.strip())
    except ValueError:
        raise RuntimeError(f"Could not parse duration from ffprobe output: {out!r}")


def ffprobe_has_video_stream(input_path: str) -> bool:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=nokey=1:noprint_wrappers=1",
        input_path
    ]
    rc, out, err = run_cmd(cmd)
    if rc != 0:
        return False
    return bool(out.strip())


def ffmpeg_has_hap() -> bool:
    rc, out, err = run_cmd(["ffmpeg", "-hide_banner", "-encoders"])
    if rc != 0:
        return False
    text = (out + "\n" + err).lower()
    return "\n" in text and " hap" in text


def format_time_for_name(seconds: int) -> str:
    return f"{seconds:04d}"


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def parse_timecode(s: str) -> float:
    """
    Accept:
      - "" => raises ValueError (caller can treat empty specially)
      - "90" or "90.5"
      - "MM:SS" or "MM:SS.mmm"
      - "HH:MM:SS" or "HH:MM:SS.mmm"
    """
    raw = s.strip()
    if raw == "":
        raise ValueError("empty")

    # If it's a plain number, interpret as seconds
    try:
        return float(raw)
    except ValueError:
        pass

    parts = raw.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"Invalid time format: {s!r}")

    try:
        parts_f = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"Invalid time format: {s!r}")

    if len(parts_f) == 2:
        mm, ss = parts_f
        if mm < 0 or ss < 0:
            raise ValueError("Negative time not allowed")
        return mm * 60.0 + ss

    hh, mm, ss = parts_f
    if hh < 0 or mm < 0 or ss < 0:
        raise ValueError("Negative time not allowed")
    return hh * 3600.0 + mm * 60.0 + ss


def build_ffmpeg_cmd(
    *,
    input_path: str,
    output_path: str,
    start_s: float,
    dur_s: float,
    mode: str,
    mute_audio: bool,
    # H.264 options
    h264_codec: str = "libx264",
    h264_crf: int = 18,
    h264_preset: str = "veryfast",
) -> List[str]:
    """
    mode:
      - "copy"  => stream copy (fast, may be off keyframes)
      - "h264"  => re-encode H.264 MP4 (accurate)
      - "hap"   => re-encode HAP MOV (TouchDesigner-friendly)
    """
    base = ["ffmpeg", "-y", "-ss", str(start_s), "-i", input_path, "-t", str(dur_s)]

    if mode == "copy":
        if mute_audio:
            return base + ["-map", "0:v:0", "-c:v", "copy", "-an", output_path]
        return base + ["-c", "copy", output_path]

    if mode == "h264":
        cmd = base + [
            "-c:v", h264_codec,
            "-preset", h264_preset,
            "-crf", str(h264_crf),
            "-movflags", "+faststart",
        ]
        if mute_audio:
            cmd += ["-an"]
        else:
            cmd += ["-c:a", "aac"]
        cmd += [output_path]
        return cmd

    if mode == "hap":
        # HAP (DXT texture compression) requires dimensions divisible by 4.
        # Auto-pad to the next multiple of 4 so any input resolution can convert.
        hap_resolution_filter = "pad=ceil(iw/4)*4:ceil(ih/4)*4"
        cmd = base + [
            "-c:v", "hap",
            "-format", "hap",
            "-pix_fmt", "rgba",
            "-vf", hap_resolution_filter,
        ]
        if mute_audio:
            cmd += ["-an"]
        else:
            cmd += ["-c:a", "aac"]
        cmd += [output_path]
        return cmd

    raise ValueError(f"Unknown mode: {mode}")


def cut_segment(
    *,
    input_path: str,
    output_path: str,
    start_s: float,
    dur_s: float,
    mode: str,
    mute_audio: bool
) -> Tuple[bool, str]:
    safe_makedirs(os.path.dirname(output_path))
    cmd = build_ffmpeg_cmd(
        input_path=input_path,
        output_path=output_path,
        start_s=start_s,
        dur_s=dur_s,
        mode=mode,
        mute_audio=mute_audio
    )
    rc, out, err = run_cmd(cmd)
    if rc != 0:
        return False, (err.strip() or "ffmpeg failed with unknown error")
    return True, "ok"


# ----------------------------
# Job modeling
# ----------------------------

@dataclass(frozen=True)
class Job:
    start_s: float
    dur_s: float
    out_path: str
    label: str


# ----------------------------
# UI App
# ----------------------------

class VideoSegmenterUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FFmpeg Segmenter — Fixed / BPM / Random Pool + Mute + HAP")
        self.geometry("940x700")

        # Core paths
        self.input_path = tk.StringVar(value="")
        self.output_dir = tk.StringVar(value="")

        # Output mode
        self.mode = tk.StringVar(value="hap")      # copy / h264 / hap
        self.mute_audio = tk.BooleanVar(value=True)

        # NEW: Range selection (timecodes)
        self.range_enabled = tk.BooleanVar(value=False)
        self.range_start = tk.StringVar(value="0")
        self.range_end = tk.StringVar(value="")  # empty => to end

        # Randomize duration
        self.randomize = tk.BooleanVar(value=True)
        self.jitter = tk.IntVar(value=5)           # seconds (+/-)

        # Parallelism
        self.workers = tk.IntVar(value=1)          # 1 = sequential

        # Segmentation mode: fixed / bpm / random_pool
        self.seg_mode = tk.StringVar(value="fixed")

        # Fixed lengths
        self.use_5 = tk.BooleanVar(value=True)
        self.use_10 = tk.BooleanVar(value=True)
        self.use_20 = tk.BooleanVar(value=True)

        # BPM controls
        self.bpm = tk.DoubleVar(value=120.0)
        self.beats_1 = tk.BooleanVar(value=False)
        self.beats_2 = tk.BooleanVar(value=False)
        self.beats_4 = tk.BooleanVar(value=True)
        self.beats_8 = tk.BooleanVar(value=True)
        self.beats_16 = tk.BooleanVar(value=False)
        self.bpm_align_starts = tk.BooleanVar(value=True)

        # Random pool controls
        self.pool_count = tk.IntVar(value=200)
        self.pool_align_to_beatgrid = tk.BooleanVar(value=False)

        # Runtime state
        self.status = tk.StringVar(value="Ready.")
        self._worker_thread = None
        self._stop_flag = threading.Event()
        self._hap_available: Optional[bool] = None

        self._build_ui()
        self._check_dependencies()

    # ---------------- UI ----------------

    def _build_ui(self):
        pad = {"padx": 12, "pady": 6}

        header = ttk.Label(self, text="FFmpeg Video Segmenter", font=("Segoe UI", 16, "bold"))
        header.pack(anchor="w", **pad)

        sub = ttk.Label(
            self,
            text="Fixed segmentation, BPM-aligned segmentation, or random clip pool. Optional mute, H.264, or HAP for TouchDesigner."
        )
        sub.pack(anchor="w", padx=12, pady=(0, 10))

        # IO
        frame_io = ttk.Frame(self)
        frame_io.pack(fill="x", **pad)

        ttk.Label(frame_io, text="Input video:").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame_io, textvariable=self.input_path).grid(row=0, column=1, sticky="we", padx=8)
        ttk.Button(frame_io, text="Browse…", command=self._browse_input).grid(row=0, column=2)

        ttk.Label(frame_io, text="Output folder:").grid(row=1, column=0, sticky="w")
        ttk.Entry(frame_io, textvariable=self.output_dir).grid(row=1, column=1, sticky="we", padx=8)
        ttk.Button(frame_io, text="Browse…", command=self._browse_output).grid(row=1, column=2)

        frame_io.columnconfigure(1, weight=1)

        # Options container
        frame_opts = ttk.LabelFrame(self, text="Options")
        frame_opts.pack(fill="x", **pad)

        # Output mode row
        enc = ttk.Frame(frame_opts)
        enc.pack(fill="x", padx=10, pady=(8, 6))
        ttk.Label(enc, text="Output mode:").grid(row=0, column=0, sticky="w")

        ttk.Radiobutton(enc, text="Fast cut (stream copy)", variable=self.mode, value="copy").grid(row=0, column=1, sticky="w", padx=8)
        ttk.Radiobutton(enc, text="Re-encode H.264 (.mp4)", variable=self.mode, value="h264").grid(row=0, column=2, sticky="w", padx=8)
        ttk.Radiobutton(enc, text="Re-encode HAP (.mov) for TouchDesigner", variable=self.mode, value="hap").grid(row=0, column=3, sticky="w", padx=8)

        # Range row (NEW)
        rr = ttk.Frame(frame_opts)
        rr.pack(fill="x", padx=10, pady=(0, 6))
        ttk.Checkbutton(rr, text="Only segment a portion of the video", variable=self.range_enabled).grid(row=0, column=0, sticky="w")
        ttk.Label(rr, text="Start:").grid(row=0, column=1, sticky="e", padx=(14, 4))
        ttk.Entry(rr, textvariable=self.range_start, width=12).grid(row=0, column=2, sticky="w")
        ttk.Label(rr, text="End:").grid(row=0, column=3, sticky="e", padx=(14, 4))
        ttk.Entry(rr, textvariable=self.range_end, width=12).grid(row=0, column=4, sticky="w")
        ttk.Label(rr, text='Formats: seconds or "MM:SS" or "HH:MM:SS" (End blank = to end)').grid(row=1, column=0, columnspan=5, sticky="w", pady=(2, 0))

        # Mute + randomize + workers row
        row2 = ttk.Frame(frame_opts)
        row2.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Checkbutton(row2, text="Mute audio", variable=self.mute_audio).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(row2, text="Randomize clip length", variable=self.randomize).grid(row=0, column=1, sticky="w", padx=12)
        ttk.Label(row2, text="Jitter ± seconds:").grid(row=0, column=2, sticky="e")
        ttk.Spinbox(row2, from_=0, to=20, textvariable=self.jitter, width=5).grid(row=0, column=3, sticky="w", padx=(6, 14))

        ttk.Label(row2, text="Workers:").grid(row=0, column=4, sticky="e")
        ttk.Spinbox(row2, from_=1, to=8, textvariable=self.workers, width=5).grid(row=0, column=5, sticky="w", padx=(6, 0))

        # Segmentation mode selector
        seg = ttk.LabelFrame(self, text="Segmentation Mode")
        seg.pack(fill="x", **pad)

        seg_top = ttk.Frame(seg)
        seg_top.pack(fill="x", padx=10, pady=8)

        ttk.Radiobutton(seg_top, text="Fixed (5/10/20 sec, sequential)", variable=self.seg_mode, value="fixed", command=self._refresh_mode_panels).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(seg_top, text="BPM-aligned (beats-per-clip, sequential)", variable=self.seg_mode, value="bpm", command=self._refresh_mode_panels).grid(row=0, column=1, sticky="w", padx=14)
        ttk.Radiobutton(seg_top, text="Random Pool (generate N clips)", variable=self.seg_mode, value="pool", command=self._refresh_mode_panels).grid(row=0, column=2, sticky="w", padx=14)

        # Panels for each mode (we show/hide)
        self.panel_fixed = ttk.Frame(seg)
        self.panel_bpm = ttk.Frame(seg)
        self.panel_pool = ttk.Frame(seg)

        # Fixed panel content
        fixed_row = ttk.Frame(self.panel_fixed)
        fixed_row.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Label(fixed_row, text="Segment lengths:").grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(fixed_row, text="5s", variable=self.use_5).grid(row=0, column=1, sticky="w", padx=8)
        ttk.Checkbutton(fixed_row, text="10s", variable=self.use_10).grid(row=0, column=2, sticky="w", padx=8)
        ttk.Checkbutton(fixed_row, text="20s", variable=self.use_20).grid(row=0, column=3, sticky="w", padx=8)

        # BPM panel content
        bpm_row1 = ttk.Frame(self.panel_bpm)
        bpm_row1.pack(fill="x", padx=10, pady=(0, 6))
        ttk.Label(bpm_row1, text="BPM:").grid(row=0, column=0, sticky="w")
        ttk.Entry(bpm_row1, textvariable=self.bpm, width=10).grid(row=0, column=1, sticky="w", padx=8)
        ttk.Checkbutton(bpm_row1, text="Align starts to beat grid", variable=self.bpm_align_starts).grid(row=0, column=2, sticky="w", padx=14)

        bpm_row2 = ttk.Frame(self.panel_bpm)
        bpm_row2.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Label(bpm_row2, text="Beats per clip:").grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(bpm_row2, text="1", variable=self.beats_1).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Checkbutton(bpm_row2, text="2", variable=self.beats_2).grid(row=0, column=2, sticky="w", padx=6)
        ttk.Checkbutton(bpm_row2, text="4", variable=self.beats_4).grid(row=0, column=3, sticky="w", padx=6)
        ttk.Checkbutton(bpm_row2, text="8", variable=self.beats_8).grid(row=0, column=4, sticky="w", padx=6)
        ttk.Checkbutton(bpm_row2, text="16", variable=self.beats_16).grid(row=0, column=5, sticky="w", padx=6)

        # Pool panel content
        pool_row1 = ttk.Frame(self.panel_pool)
        pool_row1.pack(fill="x", padx=10, pady=(0, 6))
        ttk.Label(pool_row1, text="How many clips to generate (N):").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(pool_row1, from_=1, to=10000, textvariable=self.pool_count, width=8).grid(row=0, column=1, sticky="w", padx=8)
        ttk.Checkbutton(pool_row1, text="Align random starts to beat grid (requires BPM)", variable=self.pool_align_to_beatgrid).grid(row=0, column=2, sticky="w", padx=14)

        pool_row2 = ttk.Frame(self.panel_pool)
        pool_row2.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Label(pool_row2, text="Pool uses:").grid(row=0, column=0, sticky="w")
        ttk.Label(pool_row2, text="• Fixed mode: picks from 5/10/20 lengths you checked").grid(row=0, column=1, sticky="w", padx=10)
        ttk.Label(pool_row2, text="• BPM mode: picks from beats-per-clip you checked").grid(row=1, column=1, sticky="w", padx=10)

        self._refresh_mode_panels()

        # Actions
        frame_actions = ttk.Frame(self)
        frame_actions.pack(fill="x", **pad)

        self.btn_start = ttk.Button(frame_actions, text="Start", command=self._start)
        self.btn_start.pack(side="left")

        self.btn_stop = ttk.Button(frame_actions, text="Stop", command=self._stop, state="disabled")
        self.btn_stop.pack(side="left", padx=10)

        ttk.Label(frame_actions, textvariable=self.status).pack(side="right")

        # Log
        frame_log = ttk.LabelFrame(self, text="Log")
        frame_log.pack(fill="both", expand=True, **pad)

        self.log = tk.Text(frame_log, height=18, wrap="word")
        self.log.pack(fill="both", expand=True, padx=10, pady=10)

        self._log("Notes:")
        self._log("• Stream copy is fastest but may cut slightly off-keyframe (timing can drift).")
        self._log("• HAP .mov is large but plays very smoothly in TouchDesigner.")
        self._log("• Input can be MKV/MP4/MOV/AVI/WEBM/etc if your FFmpeg build supports the codecs.")
        self._log("• Range mode lets you segment only part of the video.\n")

    def _refresh_mode_panels(self):
        for p in (self.panel_fixed, self.panel_bpm, self.panel_pool):
            p.pack_forget()

        mode = self.seg_mode.get()
        if mode == "fixed":
            self.panel_fixed.pack(fill="x", padx=6, pady=(0, 6))
        elif mode == "bpm":
            self.panel_bpm.pack(fill="x", padx=6, pady=(0, 6))
        else:
            self.panel_pool.pack(fill="x", padx=6, pady=(0, 6))

    # ---------------- Dependency checks ----------------

    def _check_dependencies(self):
        missing = []
        if which("ffmpeg") is None:
            missing.append("ffmpeg")
        if which("ffprobe") is None:
            missing.append("ffprobe")

        if missing:
            messagebox.showerror(
                "Missing dependencies",
                "Missing: " + ", ".join(missing) + "\n\nInstall FFmpeg (includes ffprobe) and ensure it's on PATH, then restart."
            )
            self.status.set("Missing ffmpeg/ffprobe.")
            self.btn_start.config(state="disabled")
            return

        self._hap_available = ffmpeg_has_hap()

    # ---------------- Browsing ----------------

    def _browse_input(self):
        path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[
                ("Video files",
                 "*.mp4 *.mov *.mkv *.avi *.m4v *.webm *.mpg *.mpeg *.m2ts *.ts *.mts *.wmv *.flv *.3gp *.ogv"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.input_path.set(path)
            if not self.output_dir.get():
                base = os.path.splitext(os.path.basename(path))[0]
                out = os.path.join(os.path.dirname(path), f"{base}_segments")
                self.output_dir.set(out)

    def _browse_output(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_dir.set(path)

    # ---------------- Logging / state ----------------

    def _log(self, text: str):
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def _ui_log(self, text: str):
        self.after(0, lambda: self._log(text))

    def _ui_status(self, text: str):
        self.after(0, lambda: self.status.set(text))

    def _set_running(self, running: bool):
        self.btn_start.config(state="disabled" if running else "normal")
        self.btn_stop.config(state="normal" if running else "disabled")

    def _ui_done(self, text: str):
        def finish():
            self.status.set(text)
            self._set_running(False)
        self.after(0, finish)

    def _ui_error(self, msg: str):
        def show():
            self._set_running(False)
            self.status.set("Error.")
            messagebox.showerror("Error", msg)
        self.after(0, show)

    # ---------------- Range handling (NEW) ----------------

    def _get_effective_range(self, full_duration: float) -> Tuple[float, float]:
        """
        Returns (range_start, range_end) in seconds, validated and clamped to [0, full_duration].
        If range is disabled => (0, full_duration).
        """
        if not self.range_enabled.get():
            return 0.0, full_duration

        # Start
        start_raw = self.range_start.get().strip()
        if start_raw == "":
            start = 0.0
        else:
            start = parse_timecode(start_raw)

        # End (blank => full duration)
        end_raw = self.range_end.get().strip()
        if end_raw == "":
            end = full_duration
        else:
            end = parse_timecode(end_raw)

        if start < 0 or end < 0:
            raise ValueError("Range start/end cannot be negative.")

        start = clamp(start, 0.0, full_duration)
        end = clamp(end, 0.0, full_duration)

        if end <= start:
            raise ValueError("Range end must be greater than range start.")

        return start, end

    # ---------------- Job generation ----------------

    def _selected_fixed_lengths(self) -> List[int]:
        lengths = []
        if self.use_5.get():
            lengths.append(5)
        if self.use_10.get():
            lengths.append(10)
        if self.use_20.get():
            lengths.append(20)
        return lengths

    def _selected_beats(self) -> List[int]:
        beats = []
        if self.beats_1.get():
            beats.append(1)
        if self.beats_2.get():
            beats.append(2)
        if self.beats_4.get():
            beats.append(4)
        if self.beats_8.get():
            beats.append(8)
        if self.beats_16.get():
            beats.append(16)
        return beats

    def _apply_jitter(self, base_dur: float) -> float:
        if not self.randomize.get():
            return base_dur
        j = float(self.jitter.get())
        if j <= 0:
            return base_dur
        return base_dur + random.uniform(-j, j)

    def _beat_grid_step(self, bpm: float) -> float:
        return 60.0 / bpm

    def _align_to_grid(self, t: float, grid: float) -> float:
        if grid <= 0:
            return t
        return round(t / grid) * grid

    def _make_jobs_fixed_sequential(
        self,
        full_duration: float,
        range_start: float,
        range_end: float,
        inp: str,
        outdir: str,
        mode: str
    ) -> List[Job]:
        lengths = self._selected_fixed_lengths()
        if not lengths:
            raise ValueError("Select at least one segment length (5/10/20).")

        ext = ".mov" if mode == "hap" else ".mp4"
        base = os.path.splitext(os.path.basename(inp))[0]

        window = range_end - range_start
        jobs: List[Job] = []

        for L in lengths:
            approx_count = math.ceil(window / L)
            for i in range(approx_count):
                start = range_start + i * L
                if start >= range_end:
                    continue

                seg_dur = self._apply_jitter(float(L))
                seg_dur = clamp(seg_dur, 0.5, range_end - start)

                sub = os.path.join(outdir, f"fixed_{L}s_{mode}")
                # name includes absolute start (seconds)
                name = f"{base}_{format_time_for_name(int(start))}_{L}s{ext}"
                out_path = os.path.join(sub, name)
                label = f"fixed {L}s @ {start:.2f}s (dur={seg_dur:.2f}s)"
                jobs.append(Job(start_s=start, dur_s=seg_dur, out_path=out_path, label=label))

        return jobs

    def _make_jobs_bpm_sequential(
        self,
        full_duration: float,
        range_start: float,
        range_end: float,
        inp: str,
        outdir: str,
        mode: str
    ) -> List[Job]:
        bpm = float(self.bpm.get())
        if bpm <= 0:
            raise ValueError("BPM must be > 0.")

        beats_list = self._selected_beats()
        if not beats_list:
            raise ValueError("Select at least one beats-per-clip option (1/2/4/8/16).")

        beat = self._beat_grid_step(bpm)
        ext = ".mov" if mode == "hap" else ".mp4"
        base = os.path.splitext(os.path.basename(inp))[0]

        jobs: List[Job] = []

        # We build on a relative timeline, then add range_start offset
        window = range_end - range_start

        for beats in beats_list:
            base_dur = beats * beat
            step = base_dur

            t = 0.0
            while t < window:
                abs_start = range_start + t

                if self.bpm_align_starts.get():
                    abs_start = self._align_to_grid(abs_start, beat)

                if abs_start >= range_end:
                    break

                seg_dur = self._apply_jitter(base_dur)
                seg_dur = clamp(seg_dur, 0.5, range_end - abs_start)

                sub = os.path.join(outdir, f"bpm_{int(round(bpm))}bpm_{beats}beats_{mode}")
                name = f"{base}_{format_time_for_name(int(abs_start))}_{beats}beats{ext}"
                out_path = os.path.join(sub, name)
                label = f"bpm {bpm:.2f} | {beats} beats @ {abs_start:.2f}s (dur={seg_dur:.2f}s)"
                jobs.append(Job(start_s=abs_start, dur_s=seg_dur, out_path=out_path, label=label))

                t += step

        return jobs

    def _make_jobs_random_pool(
        self,
        full_duration: float,
        range_start: float,
        range_end: float,
        inp: str,
        outdir: str,
        mode: str
    ) -> List[Job]:
        n = int(self.pool_count.get())
        if n <= 0:
            raise ValueError("Pool count must be >= 1.")

        ext = ".mov" if mode == "hap" else ".mp4"
        base = os.path.splitext(os.path.basename(inp))[0]

        fixed_lengths = self._selected_fixed_lengths()
        beats_list = self._selected_beats()
        bpm_val = float(self.bpm.get())

        durations: List[float] = []
        beat_grid = None

        if beats_list and bpm_val > 0:
            beat = self._beat_grid_step(bpm_val)
            beat_grid = beat
            for b in beats_list:
                durations.append(b * beat)

        for L in fixed_lengths:
            durations.append(float(L))

        if not durations:
            raise ValueError("For Random Pool: select some fixed lengths (5/10/20) and/or beats-per-clip + BPM.")

        align = self.pool_align_to_beatgrid.get()
        if align and (beat_grid is None):
            raise ValueError("Random Pool beat-grid alignment requires BPM + at least one beats-per-clip selection.")

        window = range_end - range_start
        jobs: List[Job] = []
        sub = os.path.join(outdir, f"pool_{n}_{mode}")

        for i in range(n):
            base_dur = random.choice(durations)
            seg_dur = self._apply_jitter(base_dur)
            seg_dur = max(0.5, seg_dur)

            # Choose random start within [range_start, range_end - seg_dur]
            max_start = max(0.0, window - seg_dur)
            rel_start = random.uniform(0.0, max_start)
            abs_start = range_start + rel_start

            if align and beat_grid is not None:
                abs_start = self._align_to_grid(abs_start, beat_grid)
                abs_start = clamp(abs_start, range_start, max(range_start, range_end - seg_dur))

            seg_dur = clamp(seg_dur, 0.5, range_end - abs_start)

            name = f"{base}_pool_{i:05d}_{int(round(seg_dur))}s{ext}"
            out_path = os.path.join(sub, name)
            label = f"pool #{i} @ {abs_start:.2f}s (dur={seg_dur:.2f}s)"
            jobs.append(Job(start_s=abs_start, dur_s=seg_dur, out_path=out_path, label=label))

        return jobs

    def _build_jobs(
        self,
        full_duration: float,
        range_start: float,
        range_end: float,
        inp: str,
        outdir: str,
        mode: str
    ) -> List[Job]:
        seg_mode = self.seg_mode.get()
        if seg_mode == "fixed":
            return self._make_jobs_fixed_sequential(full_duration, range_start, range_end, inp, outdir, mode)
        if seg_mode == "bpm":
            return self._make_jobs_bpm_sequential(full_duration, range_start, range_end, inp, outdir, mode)
        return self._make_jobs_random_pool(full_duration, range_start, range_end, inp, outdir, mode)

    # ---------------- Run / Stop ----------------

    def _stop(self):
        if self._worker_thread and self._worker_thread.is_alive():
            self._stop_flag.set()
            self._log("Stop requested… (running jobs will finish; queued jobs will not start)")

    def _start(self):
        inp = self.input_path.get().strip()
        outdir = self.output_dir.get().strip()
        mode = self.mode.get()
        mute = self.mute_audio.get()

        if not inp or not os.path.isfile(inp):
            messagebox.showwarning("Input required", "Please select a valid input video file.")
            return
        if not outdir:
            messagebox.showwarning("Output required", "Please select an output folder.")
            return

        # Fast sanity check for “wide codec” support
        if not ffprobe_has_video_stream(inp):
            messagebox.showerror(
                "Unsupported input",
                "This file does not appear to contain a readable video stream (or ffprobe cannot parse it)."
            )
            return

        if mode == "hap" and self._hap_available is False:
            if not messagebox.askyesno(
                "HAP encoder not detected",
                "Your ffmpeg build does not appear to include the HAP encoder.\n\n"
                "Continue anyway? (It may fail.)"
            ):
                return

        self._stop_flag.clear()
        self._set_running(True)
        self.status.set("Running…")

        self._worker_thread = threading.Thread(
            target=self._worker_run,
            args=(inp, outdir, mode, mute, int(self.workers.get())),
            daemon=True
        )
        self._worker_thread.start()

    def _worker_run(self, inp: str, outdir: str, mode: str, mute: bool, workers: int):
        try:
            full_duration = ffprobe_duration_seconds(inp)
            range_start, range_end = self._get_effective_range(full_duration)
            jobs = self._build_jobs(full_duration, range_start, range_end, inp, outdir, mode)
        except Exception as e:
            self._ui_error(str(e))
            return

        safe_makedirs(outdir)

        # Summary
        self._ui_log(f"Input: {inp}")
        self._ui_log(f"Duration: {full_duration:.2f}s")
        self._ui_log(f"Range: {range_start:.2f}s → {range_end:.2f}s (window={range_end - range_start:.2f}s)")
        self._ui_log(f"Output: {outdir}")
        self._ui_log(f"Segmentation: {self.seg_mode.get()}")
        self._ui_log(f"Mode: {mode} ({'.mov' if mode == 'hap' else '.mp4'})")
        self._ui_log(f"Mute: {'Yes' if mute else 'No'}")
        self._ui_log(f"Randomize length: {'Yes' if self.randomize.get() else 'No'} (±{self.jitter.get()}s)")
        self._ui_log(f"Workers: {workers}")
        self._ui_log(f"Jobs: {len(jobs)}\n")

        total = len(jobs)
        if total == 0:
            self._ui_done("No jobs.")
            return

        done = 0
        failed = 0

        def do_one(job: Job) -> Tuple[Job, bool, str]:
            ok, msg = cut_segment(
                input_path=inp,
                output_path=job.out_path,
                start_s=job.start_s,
                dur_s=job.dur_s,
                mode=mode,
                mute_audio=mute
            )
            return job, ok, msg

        if workers <= 1:
            for idx, job in enumerate(jobs, start=1):
                if self._stop_flag.is_set():
                    self._ui_log("\nStopped by user.")
                    self._ui_done(f"Stopped. Completed {done}/{total}, Failed {failed}.")
                    return

                self._ui_status(f"Cutting {idx}/{total}…")
                self._ui_log(f"[{idx}/{total}] {job.label}")
                ok, msg = cut_segment(
                    input_path=inp,
                    output_path=job.out_path,
                    start_s=job.start_s,
                    dur_s=job.dur_s,
                    mode=mode,
                    mute_audio=mute
                )
                if ok:
                    done += 1
                    self._ui_log("  ✅ done")
                else:
                    failed += 1
                    self._ui_log(f"  ❌ ERROR: {msg}")
        else:
            max_workers = int(clamp(workers, 1, 8))
            self._ui_log(f"Parallel enabled: max_workers={max_workers}\n")

            futures = []
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                submit_index = 0
                in_flight_cap = max_workers * 2

                while submit_index < total or futures:
                    while (not self._stop_flag.is_set()) and submit_index < total and len(futures) < in_flight_cap:
                        job = jobs[submit_index]
                        submit_index += 1
                        self._ui_log(f"[submit {submit_index}/{total}] {job.label}")
                        futures.append(ex.submit(do_one, job))

                    if not futures:
                        break

                    for f in as_completed(list(futures), timeout=None):
                        futures.remove(f)
                        job, ok, msg = f.result()
                        done_so_far = done + failed + 1
                        self._ui_status(f"Completed {done_so_far}/{total}…")
                        if ok:
                            done += 1
                            self._ui_log(f"  ✅ done → {os.path.basename(job.out_path)}")
                        else:
                            failed += 1
                            self._ui_log(f"  ❌ ERROR → {os.path.basename(job.out_path)} | {msg}")
                        break

                    if self._stop_flag.is_set() and submit_index < total:
                        self._ui_log("\nStop requested: no more jobs will be submitted. Waiting for in-flight jobs to finish…")

        self._ui_log(f"\nFinished. Completed {done}/{total}, Failed {failed}.")
        self._ui_done("Done." if failed == 0 else f"Done (with {failed} failures).")


if __name__ == "__main__":
    app = VideoSegmenterUI()
    app.mainloop()
