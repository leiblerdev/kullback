"""Record the Hub walkthrough of the published Environments.

Usage: python3 record.py airline [retail]
Writes raw.webm and marks.json (clean intervals in seconds) next to this script; cut.sh
turns them into the MP4 and the GIF. Captions are copied from the dataset card by hand and
checked against the frames before release.
"""
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
HUB = "https://huggingface.co/datasets/leibler"
TASK = ("airline", "harbor/airline-task_a8888bc3514f")

# One caption per shot; numbers as they read on each card (checked 2026-10-17).
CARD = {
    "airline": {
        "top": "leibler/airline: an executable Environment for tool-using agents, rebuilt from traces",
        "fidelity": "Replay fidelity 100.0%, 130 of 130 Tasks, and 42 of them trusted",
        "tiers": "Harbor export, 123 Tasks: a blind model rater labels each Verifier right, partial or wrong",
    },
    "retail": {
        "top": "leibler/retail: the second Environment, built and published the same way",
        "fidelity": "Replay fidelity 100.0%, 223 of 223 Tasks, and 76 of them trusted",
        "tiers": "Harbor export, 186 Tasks: 72 trusted, 48 of them labelled right",
    },
}
FILES = {
    "tree": "One trusted Task in Harbor format, with an oracle in solution/",
    "instruction": "instruction.md: the user's ask, one line per fact, then the policy",
    "toml": "task.toml: spec tier, label, judge items, oracle, and the judge settings",
    "test": "tests/test.sh replays the agent's calls and scores the Run by code",
    "end": "The commands come straight from the dataset cards",
}
# The Harbor commands exactly as each card prints them.
HARBOR = {
    "airline": [
        "uv run harbor run -p harbor/airline-task_58d9d004df44 -a oracle",
        "uv run harbor run -p harbor/airline-task_58d9d004df44 -a <agent> -m <provider/model>",
    ],
    "retail": [
        "uv run harbor run -p harbor/retail-task_19fa3e612e6a -a oracle",
        "uv run harbor run -p harbor/retail-task_19fa3e612e6a -a <agent> -m <provider/model>",
    ],
}

CAPTION_JS = """(text) => {
  let box = document.getElementById('wt-caption');
  if (!box) {
    box = document.createElement('div');
    box.id = 'wt-caption';
    Object.assign(box.style, {
      position: 'fixed', left: '6%', width: '88%', bottom: '28px', boxSizing: 'border-box',
      padding: '24px', background: '#111827', color: '#ffffff',
      font: '500 30px/1.3 "Helvetica Neue", Arial, sans-serif',
      textAlign: 'center', borderRadius: '14px', zIndex: 2147483647,
      transition: 'opacity 250ms ease', opacity: '0'});
    document.documentElement.appendChild(box);  // outside the zoomed body
    // A band in the page colour behind the caption, so no half-hidden line shows around the box.
    const band = document.createElement('div');
    Object.assign(band.style, {position: 'fixed', left: '0', right: '0', bottom: '0', height: '135px',
      background: getComputedStyle(document.body).backgroundColor || '#ffffff', zIndex: 2147483646});
    if (band.style.background.includes('rgba(0, 0, 0, 0)')) band.style.background = '#ffffff';
    document.documentElement.appendChild(band);
  }
  box.textContent = text;
  requestAnimationFrame(() => { box.style.opacity = '1'; });
}"""

SCROLL_JS = """([to, ms]) => new Promise(done => {
  const from = window.scrollY, t0 = performance.now();
  const ease = x => x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2;
  const step = now => {
    const x = Math.min(1, (now - t0) / ms);
    window.scrollTo(0, from + (to - from) * ease(x));
    x < 1 ? requestAnimationFrame(step) : done();
  };
  requestAnimationFrame(step);
})"""

END_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
body { margin: 0; height: 100vh; display: flex; align-items: center; justify-content: center;
  background: #f7f7f5; color: #111827; font-family: "Helvetica Neue", Arial, sans-serif; }
.card { width: 88%; max-width: 1600px; padding: 56px 64px; box-sizing: border-box; margin-bottom: 110px; }
h1 { font-size: 52px; margin: 0 0 36px; font-weight: 600; }
.url { font-size: 36px; margin: 0 0 12px; }
h2 { font-size: 30px; font-weight: 500; color: #4b5563; margin: 48px 0 16px; }
pre { font: 24px/1.6 Menlo, monospace; background: #111827; color: #f9fafb; padding: 28px 32px;
  border-radius: 12px; margin: 0; white-space: pre-wrap; }
</style></head><body><div class="card"><h1>Kullback Environments by Leibler</h1>%URLS%
<h2>Run a Task with Harbor</h2><pre>%CMDS%</pre></div></body></html>"""


# Page zoom for legibility (correction r1): body text about 1.65x, the Hub sidebar hidden so the
# card fills the width. The caption hangs off <html>, so it keeps its size.
ZOOM = 1.65
ZOOM_CSS = f"""body {{ zoom: {ZOOM}; }}
section.md\\:w-64 {{ display: none !important; }}
section.md\\:col-span-8 {{ grid-column: 1 / -1 !important; }}
main .container {{ padding-left: 4% !important; padding-right: 4% !important; }}
body {{ padding-bottom: 100vh; }}
pre, pre *, code, td.blob-line, td.blob-line * {{ white-space: pre-wrap !important; overflow-wrap: anywhere; }}"""
FREE = 945  # pixels above the caption box

# Page-coordinate span [top, bottom] from a start element to an end element.
SPAN_JS = """([a, b]) => {
  const find = spec => {
    if (spec.css) return document.querySelector(spec.css);
    const hits = [...document.querySelectorAll('body *')].filter(e =>
      e.textContent.trim().startsWith(spec.text));
    // The deepest matches: no other match inside them.
    const all = hits.filter(e => !hits.some(o => o !== e && e.contains(o)));
    return spec.last ? all[all.length - 1] : all[0];
  };
  const ea = find(a), eb = b ? find(b) : ea;
  if (!ea || !eb) return null;
  const ra = ea.getBoundingClientRect(), rb = eb.getBoundingClientRect();
  return [ra.top + scrollY, rb.bottom + scrollY];
}"""


def main(corpora):
    marks = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1920, "height": 1080},
                                  record_video_dir=str(HERE / "video-raw"),
                                  record_video_size={"width": 1920, "height": 1080})
        page = ctx.new_page()
        t0 = time.monotonic()

        def now():
            return round(time.monotonic() - t0, 3)

        # Sync flash: magenta until `sync`, so cut.py can find the offset of the video clock.
        page.set_content('<body style="margin:0;background:#ff00ff;height:100vh"></body>')
        page.wait_for_timeout(3000)
        page.evaluate("document.body.style.background = '#ffffff'")
        page.wait_for_timeout(50)
        marks.append({"shot": "sync", "start": now(), "end": now()})
        page.wait_for_timeout(500)

        def scroll(to, ms=1200):
            page.evaluate(SCROLL_JS, [max(0, to), ms])

        def span(a, b=None):
            got = page.evaluate(SPAN_JS, [a, b])
            assert got, f"not found on {page.url}: {a} {b}"
            return got

        def centre(a, b=None, ms=1400):
            """Scroll so the span sits centred above the caption; a taller span starts at the top."""
            top, bottom = span(a, b)
            h = bottom - top
            scroll(top - (FREE - h) / 2 if h <= FREE - 40 else top - 20, ms)
            return top, bottom

        def goto(url):
            # A dataset viewer that is still converting polls forever, so networkidle never comes.
            page.goto(url, wait_until="load")
            try:
                page.wait_for_load_state("networkidle", timeout=8000)
            except Exception:
                pass
            page.add_style_tag(content=ZOOM_CSS)
            page.wait_for_timeout(1500)

        def begin(caption):
            page.evaluate(CAPTION_JS, caption)
            return now()

        def end(name, start):
            marks.append({"shot": name, "start": start, "end": now()})

        for corpus in corpora:
            c = CARD[corpus]
            goto(f"{HUB}/{corpus}")
            start = begin(c["top"])
            page.wait_for_timeout(3000)
            centre({"css": "section.md\\:col-span-8 .prose h1"}, {"text": "Release: replays"})
            page.wait_for_timeout(4000)
            end(f"1-{corpus}", start)
            start = now()
            top, bottom = centre({"text": "Fidelity over Tasks"}, {"text": "Content hash"})
            page.evaluate(CAPTION_JS, c["fidelity"])
            page.wait_for_timeout(3500)
            if bottom - top > FREE - 40:  # too tall for one frame: glide on to its last row
                scroll(bottom - FREE + 20, 2200)
                page.wait_for_timeout(2000)
            end(f"2a-{corpus}", start)
            start = now()
            centre({"text": "Task set:"}, {"text": "unchecked_ask", "last": True})
            page.evaluate(CAPTION_JS, c["tiers"])
            page.wait_for_timeout(4500)
            end(f"2b-{corpus}", start)

        corpus, task = TASK
        goto(f"{HUB}/{corpus}/tree/main/{task}")
        centre({"css": "a[href$='/tree/main/" + task + "/environment']"}, {"css": "a[href$='/task.toml']"}, 1)
        start = begin(FILES["tree"])
        page.wait_for_timeout(4000)
        end("3-tree", start)
        page.click("a[href$='/instruction.md']")
        page.wait_for_load_state("load")
        page.add_style_tag(content=ZOOM_CSS)
        page.wait_for_timeout(1500)
        centre({"text": "A user writes in"}, {"text": "wants at least", "last": True}, 1)
        start = begin(FILES["instruction"])
        page.wait_for_timeout(4500)
        end("3-instruction", start)
        goto(f"{HUB}/{corpus}/blob/main/{task}/task.toml")
        centre({"text": "[metadata]"}, {"text": "has_oracle"}, 1)
        start = begin(FILES["toml"])
        page.wait_for_timeout(4000)
        centre({"text": "[verifier]"}, {"text": "ANTHROPIC_API_KEY", "last": True})
        page.wait_for_timeout(3000)
        end("4-toml", start)
        goto(f"{HUB}/{corpus}/blob/main/{task}/tests/test.sh")
        centre({"text": "#!/bin/bash"}, {"text": "fi", "last": True}, 1)
        start = begin(FILES["test"])
        page.wait_for_timeout(4000)
        end("5-test", start)

        urls = "".join(f'<p class="url">huggingface.co/datasets/leibler/{c}</p>' for c in corpora)
        cmds = "\n".join(c.replace("<", "&lt;").replace(">", "&gt;") for k in corpora for c in HARBOR[k])
        page.set_content(END_HTML.replace("%URLS%", urls).replace("%CMDS%", cmds))
        page.wait_for_timeout(300)
        start = begin(FILES["end"])
        page.wait_for_timeout(6500)
        end("6-end", start)

        video = page.video
        ctx.close()
        Path(video.path()).replace(HERE / "raw.webm")
        browser.close()
    (HERE / "marks.json").write_text(json.dumps(marks, indent=1))
    print(json.dumps(marks, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or ["airline"])
