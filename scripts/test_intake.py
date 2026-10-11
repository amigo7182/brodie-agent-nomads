"""The intake (--intake): Perception on a recorded squats clip, driven the way the voice drives it (no room, no model).

 A. set_plan during the intake: the session waits in 'intake' (no cue), then starts the CHOSEN plan with "Let's do squats",
    the session events carry items = the new plan's length, the rest uses the chosen rest_s, and squats are counted.
 B. a conversation hold keeps the intake open past the timeout, and a chosen plan waits until the hold clears.
 C. nothing chosen: the configured plan starts after intake_timeout_s.
 D. set_plan is refused with a bad exercise name, and once the session is past the intake.
 E. starter/programs.py: every goal x level builds a plan the session accepts, reps within 3..30.

Run:  .venv/bin/python scripts/test_intake.py [--video data/tmp/mac_handoff/segments/run3_squats.mp4]
"""
import argparse, importlib.util, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from facetime.perception import Perception, Config
from facetime.pipeline import FPS
from rtcoach.video import read_frames

PASS = []


def check(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg, flush=True); PASS.append(bool(cond)); return cond


def run(fr, cfg, on_frame=None):
    """Feed the frames; on_frame(i, per) may act between frames. Returns (session events, cue events, per)."""
    per = Perception(cfg, t0=0.0); evs = []
    per.on_event(evs.append)
    for i in range(len(fr)):
        if on_frame: on_frame(i, per)
        per.process_frame(fr[i], t_recv=i / FPS, t_ms=i * 1000.0 / FPS)
    per.close()
    return [e for e in evs if e['type'] == 'session'], [e for e in evs if e['type'] == 'cue'], per


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--video', default=os.path.join(ROOT, 'data', 'tmp', 'mac_handoff', 'segments', 'run3_squats.mp4'))
    a = ap.parse_args()
    fr = read_frames(a.video); print(f'{a.video}: {len(fr)} frames ({len(fr) / FPS:.0f} s)')
    base = dict(plan='lunges:5,pushups:5', intake=True, intake_timeout_s=60.0)

    print('[A] set_plan during the intake starts the chosen plan')
    res = {}
    def choose(i, per):
        if i == 20: res['why'] = per.session.set_plan('squats:4,jumping jacks:6', 9.0)
    sess, cues, per = run(fr, Config(**base), choose)
    states = [e['state'] for e in sess]
    check(res.get('why') is None, f'set_plan accepted ({res.get("why")})')
    check(states[:2] == ['init', 'intake'] or states[0] == 'intake', f'opens in intake: {states[:3]}')
    t_intake = next(e['t'] for e in sess if e['state'] == 'intake'); t_wait = next(e['t'] for e in sess if e['state'] == 'wait')
    check(not [c for c in cues if t_intake <= c['t'] < t_wait and c['intent'] != 'visibility'], 'nothing said during the intake but visibility lines')
    w = next(e for e in sess if e['state'] == 'wait')
    check(w['exercise'] == 'squats' and w['target'] == 4 and w['items'] == 2, f'first set is the chosen one: {w["exercise"]} x{w["target"]}, items {w["items"]}')
    check(any(c['text'] == "Let's do squats" for c in cues), '"Let\'s do squats" announced')
    check(per.session.rest_s == 9.0, f'rest_s from the choice: {per.session.rest_s}')
    reps = max((e['count'] for e in sess if e['exercise'] == 'squats' or e['state'] == 'rest'), default=0)
    check(any(e['state'] == 'rest' and e['count'] == 4 for e in sess), f'the 4 squats were counted and the set closed (max count {reps})')

    print('[B] a conversation hold keeps the intake open and delays the start')
    def hold(i, per):
        per.hold = i < 120                                               # 12 s of "talking"
        if i == 20: per.session.set_plan('squats:4', 9.0)
    sess, cues, per = run(fr, Config(**dict(base, intake_timeout_s=5.0)), hold)
    w = next(e for e in sess if e['state'] == 'wait')
    check(w['exercise'] == 'squats' and w['t'] >= 12.0, f'the chosen plan started after the hold cleared (t={w["t"]:.1f} s), not the timeout fallback')

    print('[C] nothing chosen: the configured plan after the timeout')
    sess, cues, per = run(fr, Config(**dict(base, intake_timeout_s=5.0)))
    t_intake = next(e['t'] for e in sess if e['state'] == 'intake'); w = next(e for e in sess if e['state'] == 'wait')
    check(w['exercise'] == 'lunges' and w['items'] == 2 and 4.9 <= w['t'] - t_intake <= 6.0, f'configured plan started {w["t"] - t_intake:.1f} s into the intake: {w["exercise"]}')

    print('[D] set_plan refusals')
    res = {}
    def bad(i, per):
        if i == 20: res['bad'] = per.session.set_plan('burpees:5')
        if i == len(fr) - 1: res['late'] = per.session.set_plan('squats:3')
    sess, cues, per = run(fr, Config(**dict(base, intake_timeout_s=5.0)), bad)
    check(res.get('bad') is not None, f'unknown exercise refused: {str(res.get("bad"))[:70]!r}')
    check(res.get('late') is not None and 'too late' in res['late'], f'after the intake refused: {res.get("late")!r}')
    off = Perception(Config(plan='squats:5'), t0=0.0)
    check(off.session.set_plan('squats:3') is not None and off.session.state == 'init', 'without --intake set_plan is refused'); off.close()

    print('[E] starter/programs.py')
    spec = importlib.util.spec_from_file_location('programs', os.path.join(ROOT, 'starter', 'programs.py')); progs = importlib.util.module_from_spec(spec); spec.loader.exec_module(progs)
    per = Perception(Config(**base), t0=0.0); bad = []
    for g in progs.GOALS:
        for lv in progs.LEVELS:
            plan, rest_s, said = progs.build(g, lv); per.session.pending_plan = None
            why = per.session.set_plan(plan, rest_s); ns = [int(x.rpartition(':')[2]) for x in plan.split(',')]
            if why is not None or not all(3 <= n <= 30 for n in ns): bad.append((g, lv, plan, why))
    per.close()
    check(not bad, f'all {len(progs.GOALS) * len(progs.LEVELS)} goal x level plans valid' + (f': {bad}' if bad else ''))
    check(progs.build('yoga', 'extreme')[0] == progs.build(*progs.DEFAULT)[0], 'an unknown goal/level falls back to the default')

    print('[F] out of view mid-set: the set closes after lost_close_s with the count reached (no endless "Step back")')
    import numpy as np
    blank = np.zeros_like(fr[0]); k0 = {}
    def vanish(i, per):                                                   # the caller walks out of view once the set has two reps
        if 'i' not in k0 and per.session.state == 'active' and per.session.last_count >= 2: k0['i'] = i
    fr2 = list(fr[:200]); per = Perception(Config(plan='squats:20', lost_close_s=8.0), t0=0.0); evs = []; per.on_event(evs.append)
    for i in range(len(fr2) + 160):
        if i < len(fr2): vanish(i, per)
        img = blank if ('i' in k0 and i >= k0['i']) or i >= len(fr2) else fr2[i]
        per.process_frame(img, t_recv=i / FPS, t_ms=i * 1000.0 / FPS)
    per.close()
    sess = [e for e in evs if e['type'] == 'session']; cues = [e for e in evs if e['type'] == 'cue']
    rest = next((e for e in sess if e['state'] in ('rest', 'done')), None)        # a one-set plan: the close goes straight to "All done"
    check(rest is not None and 2 <= rest['count'] < 20 and any('nice work' in c['text'] for c in cues if c['intent'] == 'milestone'),
          f'set closed out of view at {rest and rest["count"]} reps with a milestone line (n_lost_closed {per.session.n_lost_closed})')
    per = Perception(Config(plan='squats:20', lost_close_s=0.0), t0=0.0); evs = []; per.on_event(evs.append)
    for i in range(160): per.process_frame(fr[i] if i < 60 else blank, t_recv=i / FPS, t_ms=i * 1000.0 / FPS)
    per.close()
    check(not any(e['type'] == 'session' and e['state'] == 'rest' for e in evs), 'lost_close_s=0 keeps the old behaviour (waits)')

    print('[G] request_next: "I\'m ready" during a rest starts the next exercise at once, and waits while the voice talks')
    res = {}
    def ready(i, per):
        s = per.session
        if s.state == 'rest' and 'rest_i' not in res: res['rest_i'] = i
        if 'rest_i' in res and i == res['rest_i'] + 5:
            res['why'] = s.request_next(); per.hold = True                 # the voice is still answering
        if 'rest_i' in res and i == res['rest_i'] + 25: per.hold = False
    sess, cues, per = run(fr, Config(plan='squats:3,lunges:4', rest_s=60.0), ready)
    nxt = next((e for e in sess if e['state'] == 'wait' and e['exercise'] == 'lunges'), None)
    rest_t = next(e['t'] for e in sess if e['state'] == 'rest')
    check(res.get('why') is None and nxt is not None and 2.0 <= nxt['t'] - rest_t < 10.0,
          f'lunges started {nxt and nxt["t"] - rest_t:.1f} s into a 60 s rest, after the hold cleared')
    fresh = Perception(Config(plan='squats:3'), t0=0.0); why = fresh.session.request_next(); fresh.close()   # still in init: nothing to skip
    check(why is not None and 'init' in why, f'request_next outside a rest or wait is refused ({why})')

    print('[H] auto-start: the caller just starts the next exercise during a rest; the set starts on that rep and keeps it')
    sess, cues, per = run(fr, Config(plan='squats:2,squats:5', rest_s=60.0))         # one clip of continuous squats: the second set starts by itself
    rests = [e for e in sess if e['state'] == 'rest']; w2 = next((e for e in sess if e['state'] == 'wait' and e.get('item') == 1), None)
    check(len(rests) >= 1 and w2 is not None and w2['t'] - rests[0]['t'] < 20.0 and per.session.n_autostart == 1,
          f'second set started {w2 and w2["t"] - rests[0]["t"]:.1f} s into a 60 s rest (auto-starts {per.session.n_autostart})')
    rest_events = [e for e in sess if e['state'] == 'rest' and e.get('item') == 0]
    check(len(rest_events) == 1 and all(e['exercise'] is None for e in rest_events), f'one rest event for the first set, no exercise named in it ({len(rest_events)})')
    second = [e for e in sess if e.get('item') == 1 and e['state'] in ('rest', 'done')]
    check(any(c['text'] == 'Moving on to squats' for c in cues) and (not second or second[0]['count'] >= 1), f'"Moving on to squats" said; set two counted {second and second[0]["count"]}')
    sess, cues, per = run(fr, Config(plan='squats:2,squats:5', rest_s=60.0, rest_autostart=False))
    check(per.session.n_autostart == 0 and not any(e.get('item') == 1 for e in sess), '--no-rest-autostart keeps the full rest')

    print('[J] the last set ends the workout at once: "All done" with no rest, and no "Step back" after it')
    import numpy as np
    blank = np.zeros_like(fr[0]); per = Perception(Config(plan='squats:3'), t0=0.0); evs = []; per.on_event(evs.append)
    frames = list(fr[:250]) + [blank] * 80                                 # the caller walks off after the set
    for i, img in enumerate(frames): per.process_frame(img, t_recv=i / FPS, t_ms=i * 1000.0 / FPS)
    per.close()
    sess = [e for e in evs if e['type'] == 'session']; cues = [e for e in evs if e['type'] == 'cue']
    states = [e['state'] for e in sess]; done = next((e for e in sess if e['state'] == 'done'), None)
    t_done = done['t'] if done else 1e9
    check(done is not None and 'rest' not in states and done['count'] == 3 and any(c['text'] == 'All done, great session' for c in cues),
          f'states {states}; done with count {done and done["count"]}')
    check(not any(c['intent'] == 'visibility' and c['t'] > t_done for c in cues), 'no visibility line after "All done"')

    print('[I] the results card as a video frame')
    from facetime.recap import CallRecap, card_image, card_frame
    r = CallRecap(); [r.on_event(e) for e in (dict(type='session', state='wait', exercise='squats', target=7), dict(type='session', state='done', count=7))]
    f = card_frame(card_image(r, dict(current=2, best=2)))
    check(f.shape == (1280, 720, 3) and f.dtype.name == 'uint8' and f[:50].max() == 10 and f[600].max() > 100, f'720x1280 frame, black bars, card in the middle ({f.shape})')

    n = len(PASS); ok = sum(PASS); print(f'\n{ok}/{n} checks passed' + ('' if ok == n else '  <-- FAILURES'), flush=True)
    return 0 if ok == n else 1


if __name__ == '__main__':
    rc = main(); sys.stdout.flush(); os._exit(rc)
