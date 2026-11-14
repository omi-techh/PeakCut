"""Live recording: a bad channel answer, a crash while starting, and a stop that comes during the lookup.
Runs against the code directly with its own empty data folder, and never starts a real ffmpeg or contacts a site."""
import os, sys, tempfile, threading, time
os.environ["PEAKCUT_DATA"] = tempfile.mkdtemp(prefix="pc-live-"); os.environ["DATABASE_URL"] = ""
sys.path.insert(0, ".")
import yt_dlp
from studio import auth, creators, ctx, db, jobs, live
db.init()
threading.Thread(target=jobs._live_worker, daemon=True).start()   # the live cleanup queue only runs when this is up
fails = []
def check(n, ok, x=""):
    print(("PASS " if ok else "FAIL ") + n + (f"  {x}" if x and not ok else "")); ok or fails.append(n)

u, _ = auth._create("live@example.com", "Live", password_hash="x")
TWITCH = "https://www.twitch.tv/somebody"

# Test 1: yt-dlp answers live but gives no stream address
yt_dlp.YoutubeDL.extract_info = lambda self, url, download=False: {"is_live": True, "requested_formats": [{}]}
try:
    live.resolve(TWITCH)
    ok = False
except live.NotLive:
    ok = True
except Exception as e:  # noqa: BLE001
    ok = False
    print("  got", type(e).__name__, e)
check("a live answer with no stream address raises NotLive, not KeyError", ok)

def wait_for(cond, secs=4.0):
    end = time.time() + secs
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return cond()

with ctx.as_user(u.id):
    cr = creators.create({"name": "Streamer", "channel": TWITCH})

    # Test 2: something unexpected breaks while the recording starts
    def boom(url):
        raise RuntimeError("boom")
    live.resolve = boom
    j = jobs.create({"type": "live", "url": TWITCH, "name": "x live"}, {}, enqueue=False, creator=cr["id"], kind="live")
    j.update(status="live", stage="Connecting to the stream"); jobs.save(j)
    s = live.Session(j, cr)
    s.start()
    done = wait_for(lambda: s.ended and s.finished)
    check("an unexpected error while starting ends the session", done, f"ended={s.ended} finished={s.finished}")
    check("the failed recording is marked as an error", s.job["status"] == "error", str(s.job["status"]))
    check("the failed session is no longer active", s not in live.active())

    # Test 3: stop() comes while the stream address is still being looked up
    entered, stopped = threading.Event(), threading.Event()
    popen_calls = []
    real_popen = live.subprocess.Popen

    def slow_resolve(url):
        entered.set()
        stopped.wait(4)
        return {"urls": ["http://127.0.0.1:9/x.m3u8"], "title": "t", "uploader": "u"}

    def fake_popen(*a, **k):
        popen_calls.append(a)
        raise FileNotFoundError("ffmpeg missing")

    live.resolve = slow_resolve
    live.subprocess.Popen = fake_popen
    try:
        j2 = jobs.create({"type": "live", "url": TWITCH, "name": "y live"}, {}, enqueue=False, creator=cr["id"], kind="live")
        j2.update(status="live", stage="Connecting to the stream"); jobs.save(j2)
        s2 = live.Session(j2, cr)
        s2.start()
        entered.wait(4)
        s2.stop()
        stopped.set()
        done2 = wait_for(lambda: s2.ended and s2.finished)
    finally:
        live.subprocess.Popen = real_popen
    check("ffmpeg is never started when stop came during the lookup", not popen_calls, f"Popen calls={len(popen_calls)}")
    check("a session stopped during the lookup ends", done2, f"ended={s2.ended} finished={s2.finished}")

print("ALL PASSED" if not fails else f"{len(fails)} failed")
import shutil; shutil.rmtree(os.environ["PEAKCUT_DATA"], ignore_errors=True)
sys.exit(1 if fails else 0)
