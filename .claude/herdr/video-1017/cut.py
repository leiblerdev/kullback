"""Cut raw.webm into the walkthrough MP4 and GIF using marks.json.

Usage: python3 cut.py [out-stem]. Finds the sync flash (last magenta frame), shifts every mark
onto the video clock, trims each shot 0.15 s inward and joins the shots with short fades.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FF = "/opt/homebrew/bin/ffmpeg"
stem = sys.argv[1] if len(sys.argv) > 1 else "leibler-environments-walkthrough"
marks = json.loads((HERE / "marks.json").read_text())
sync = next(m for m in marks if m["shot"] == "sync")["start"]

# Mean colour of a tiny centre crop for every frame of the first 6 seconds.
probe = subprocess.run(
    [FF, "-v", "error", "-t", "6", "-i", str(HERE / "raw.webm"), "-vf", "crop=8:8:956:536,scale=1:1",
     "-f", "rawvideo", "-pix_fmt", "rgb24", "-vsync", "passthrough", "-"],
    capture_output=True, check=True).stdout
pts = subprocess.run(
    ["ffprobe", "-v", "error", "-read_intervals", "%+6", "-select_streams", "v",
     "-show_entries", "frame=pts_time", "-of", "csv=p=0", str(HERE / "raw.webm")],
    capture_output=True, text=True, check=True).stdout.split()
flip, seen = None, False
for i in range(min(len(pts), len(probe) // 3)):
    r, g, b = probe[3 * i: 3 * i + 3]
    if r > 200 and g < 60 and b > 200:
        seen = True
    elif seen:
        flip = float(pts[i])
        break
assert flip is not None, "sync flash not found"
offset = flip - sync
print(f"sync flip at video {flip:.3f}s, script {sync:.3f}s, offset {offset:+.3f}s")

shots = [m for m in marks if m["shot"] != "sync"]
parts, labels = [], []
for i, m in enumerate(shots):
    a, b = m["start"] + offset + 0.15, m["end"] + offset - 0.15
    parts.append(f"[0:v]trim={a:.3f}:{b:.3f},setpts=PTS-STARTPTS,fps=30,format=yuv420p[s{i}]")
    m["video_start"], m["length"] = round(a, 3), round(b - a, 3)
fade = 0.3
chain, prev, acc = [], "s0", shots[0]["length"]
for i in range(1, len(shots)):
    out = f"x{i}"
    chain.append(f"[{prev}][s{i}]xfade=transition=fade:duration={fade}:offset={acc - fade:.3f}[{out}]")
    acc += shots[i]["length"] - fade
    prev = out
graph = ";".join(parts + chain)
mp4 = HERE / f"{stem}.mp4"
subprocess.run([FF, "-v", "error", "-y", "-i", str(HERE / "raw.webm"), "-filter_complex", graph,
                "-map", f"[{prev}]", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
                "-pix_fmt", "yuv420p", "-r", "30", "-movflags", "+faststart", str(mp4)], check=True)
gif = HERE / f"{stem}.gif"
subprocess.run([FF, "-v", "error", "-y", "-i", str(mp4), "-filter_complex",
                "fps=10,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];"
                "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle", str(gif)], check=True)
# Where each shot sits in the output, for frame checks.
t = 0.0
for i, m in enumerate(shots):
    m["out_mid"] = round(t + m["length"] / 2, 2)
    t += m["length"] - (fade if i < len(shots) - 1 else 0)
(HERE / f"{stem}.shots.json").write_text(json.dumps(shots, indent=1))
print(f"total {acc:.2f}s")
for m in shots:
    print(m["shot"], m["out_mid"], m["length"])
