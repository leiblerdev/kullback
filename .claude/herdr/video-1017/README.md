# Walkthrough video of the published Environments

Screen recording of the public Hugging Face pages of leibler/airline (and leibler/retail for the
final cut). No login, no keys. Work data lives in .work-video/ at the worktree root (ignored).

    python3 record.py airline [retail]   # raw.webm and marks.json (one interval per shot)
    python3 cut.py                       # MP4 1920x1080 30 fps H.264 and a 960 wide GIF
    ./frames.sh leibler-environments-walkthrough   # one frame per shot into frames/

record.py needs Playwright for Python and its cached Chromium. It starts with a magenta sync
flash; cut.py finds it to put the script's marks on the video clock, trims each shot and joins
the shots with 0.3 s fades. Captions are typed in record.py from the cards; check every number
against the frames after each recording, and add the retail captions to CARD from its card.
