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
    document.body.appendChild(box);
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

        def shot(name, caption, hold):
            page.evaluate(CAPTION_JS, caption)
            start = now()
            page.wait_for_timeout(hold)
            marks.append({"shot": name, "start": start, "end": now()})

        def scroll(to, ms=1200):
            page.evaluate(SCROLL_JS, [to, ms])

        def top_of(selector_js):
            return page.evaluate(f"() => Math.round(({selector_js}).getBoundingClientRect().top + scrollY)")

        def goto(url):
            # A dataset viewer that is still converting polls forever, so networkidle never comes.
            page.goto(url, wait_until="load")
            try:
                page.wait_for_load_state("networkidle", timeout=8000)
            except Exception:
                pass
            page.wait_for_timeout(1500)

        for corpus in corpora:
            c = CARD[corpus]
            goto(f"{HUB}/{corpus}")
            page.evaluate(CAPTION_JS, c["top"])
            start = now()
            page.wait_for_timeout(3000)
            card_h1 = top_of("[...document.querySelectorAll('h1')].find(h => h.innerText.trim() === '%s')" % corpus)
            scroll(card_h1 - 90, 1400)
            page.wait_for_timeout(4000)
            marks.append({"shot": f"1-{corpus}", "start": start, "end": now()})
            fid = top_of("[...document.querySelectorAll('table')].find(t => t.innerText.includes('Fidelity over Tasks'))")
            start = now()
            scroll(fid - 40, 1400)
            page.evaluate(CAPTION_JS, c["fidelity"])
            page.wait_for_timeout(4500)
            marks.append({"shot": f"2a-{corpus}", "start": start, "end": now()})
            harbor = top_of("[...document.querySelectorAll('h2')].find(h => h.innerText.includes('Run it with Harbor'))")
            start = now()
            scroll(harbor - 50, 1400)
            page.evaluate(CAPTION_JS, c["tiers"])
            page.wait_for_timeout(4500)
            marks.append({"shot": f"2b-{corpus}", "start": start, "end": now()})

        corpus, task = TASK
        goto(f"{HUB}/{corpus}/tree/main/{task}")
        shot("3-tree", FILES["tree"], 4000)
        page.click("a[href$='/instruction.md']")
        page.wait_for_load_state("load")
        page.wait_for_timeout(1500)
        shot("3-instruction", FILES["instruction"], 4500)
        goto(f"{HUB}/{corpus}/blob/main/{task}/task.toml")
        page.evaluate(CAPTION_JS, FILES["toml"])
        start = now()
        page.wait_for_timeout(1200)
        scroll(300, 1400)
        page.wait_for_timeout(4500)
        marks.append({"shot": "4-toml", "start": start, "end": now()})
        goto(f"{HUB}/{corpus}/blob/main/{task}/tests/test.sh")
        shot("5-test", FILES["test"], 4000)

        urls = "".join(f'<p class="url">huggingface.co/datasets/leibler/{c}</p>' for c in corpora)
        cmds = "\n".join(c.replace("<", "&lt;").replace(">", "&gt;") for k in corpora for c in HARBOR[k])
        page.set_content(END_HTML.replace("%URLS%", urls).replace("%CMDS%", cmds))
        page.wait_for_timeout(300)
        shot("6-end", FILES["end"], 6500)

        video = page.video
        ctx.close()
        Path(video.path()).replace(HERE / "raw.webm")
        browser.close()
    (HERE / "marks.json").write_text(json.dumps(marks, indent=1))
    print(json.dumps(marks, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or ["airline"])
