"""The boxing lesson (starter/boxing.py and its wiring), offline: the tracker's form report, the session's lesson state on a real
clip, the lesson timeline with a fake coach, the worker's start_boxing_lesson tool end to end, and the lesson card.

Run:  .venv/bin/python scripts/test_boxing.py
"""
import asyncio, importlib.util, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from facetime.perception import Perception, Config
from facetime.pipeline import FPS
from rtcoach.video import read_frames

PASS = []
def check(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg, flush=True); PASS.append(bool(cond)); return cond

spec = importlib.util.spec_from_file_location('boxing', os.path.join(ROOT, 'starter', 'boxing.py')); boxing = importlib.util.module_from_spec(spec); spec.loader.exec_module(boxing)


def body(wrist_y, ankle_dx=0.12):
    """33 landmarks, image-normalised: shoulders at y 0.30 (width 0.12), wrists at wrist_y, ankles ankle_dx apart."""
    lm = np.zeros((33, 5), np.float32); lm[:, 3] = 1.0
    lm[0, :2] = (0.5, 0.18); lm[11, :2] = (0.56, 0.30); lm[12, :2] = (0.44, 0.30)
    lm[15, :2] = (0.55, wrist_y[0]); lm[16, :2] = (0.45, wrist_y[1])
    lm[27, :2] = (0.5 + ankle_dx / 2, 0.9); lm[28, :2] = (0.5 - ankle_dx / 2, 0.9)
    return lm


async def amain():
    print('[A] form report from landmarks: guard height and stance width')
    t_up, d_up = boxing.form_report(body((0.22, 0.22))); t_low, d_low = boxing.form_report(body((0.22, 0.55))); t_both, d_both = boxing.form_report(body((0.6, 0.6)))
    check(d_up['hands'] == 'up' and d_low['hands'] == 'one low' and d_both['hands'] == 'both low', f"hands: {d_up['hands']}, {d_low['hands']}, {d_both['hands']}")
    _, narrow = boxing.form_report(body((0.22, 0.22), ankle_dx=0.04)); _, ok = boxing.form_report(body((0.22, 0.22), ankle_dx=0.15))
    check(narrow['stance_ratio'] < 0.7 and 0.7 <= ok['stance_ratio'] <= 2.2 and 'narrow' in boxing.form_report(body((0.22, 0.22), 0.04))[0], f"stance ratio narrow {narrow['stance_ratio']}, fine {ok['stance_ratio']}")
    check(boxing.form_report(None)[0] is None and 'left' not in t_low.lower() and 'right' not in t_low.lower(), f'no body -> None; never left/right: {t_low!r}')

    print('[B] the timeline with a fake coach: 3 rounds, an explanation, a mid-round cue with the tracker note, rests')
    asks = []
    class C:
        round_s = 1.2; rest_s = 0.2
        def ask(self, p): asks.append(p)
        def say(self, t): pass
        def status(self, t): self.last = t
        def form(self): return 'The tracker sees: one hand has dropped below the shoulders.'
        def log(self, t): pass
        async def wait_quiet(self, max_s): await asyncio.sleep(0.01)
    for style in boxing.STYLES:
        asks.clear(); taught = await boxing.run_lesson(style, C())
        drills = [d[0] for d in boxing.STYLES[style]['drills']]
        rounds = [a for a in asks if a.startswith('Round ')]; mids = [a for a in asks if a.startswith('Mid-round')]; rests = [a for a in asks if a.startswith('Time!')]
        check(taught == drills and len(rounds) == 3 and len(mids) == 3 and len(rests) == 2 and 'dropped' in mids[0], f'{style}: {len(rounds)} rounds, {len(mids)} cues, {len(rests)} rests')

    print('[B2] instant tracker cues: "Hands up" when the guard drops, praise once for a held guard, never while Brodie talks')
    said = []
    class F(C):
        round_s = 12.0; rest_s = 0.0
        def __init__(self): self.t0 = asyncio.get_running_loop().time(); self.talking = False
        def form_data(self):
            el = asyncio.get_running_loop().time() - self.t0
            return dict(hands='one low', stance_ratio=1.2) if el < 7 else dict(hands='up', stance_ratio=1.2)
        def say(self, t):
            if self.talking: return None
            said.append(t); return {'status': 'ok'}
    boxing_drills = boxing.STYLES['soviet']['drills']; boxing.STYLES['soviet']['drills'] = boxing_drills[:1]
    try: await boxing.run_lesson('soviet', F())
    finally: boxing.STYLES['soviet']['drills'] = boxing_drills
    check(said[:1] == ['Hands up'] and 'Widen your stance' not in said, f'instant cues in a 12 s round with the guard low for 7 s: {said}')
    said.clear(); f = F(); f.talking = True; boxing.STYLES['soviet']['drills'] = boxing_drills[:1]
    try: await boxing.run_lesson('soviet', f)
    finally: boxing.STYLES['soviet']['drills'] = boxing_drills
    check(said == [], 'no instant cue while Brodie is talking')

    print('[B3] "I\'m ready" in a lesson: the round and the rest end early')
    class K(C):
        round_s = 30.0; rest_s = 30.0
        def __init__(self): self.flag = False; self.logs = []
        def skip(self):
            if self.flag: self.flag = False; return True
            return False
        def log(self, t): self.logs.append(t)
    k = K(); t0 = asyncio.get_running_loop().time()
    async def press():
        while True: await asyncio.sleep(0.4); k.flag = True                # the caller keeps saying "next"
    presser = asyncio.create_task(press())
    taught = await boxing.run_lesson('mexican', k); presser.cancel()
    took = asyncio.get_running_loop().time() - t0
    check(len(taught) == 3 and took < 10 and any('ended early' in l for l in k.logs), f'3 rounds and 2 rests of 30 s finished in {took:.1f} s by skipping')
    _, d_line = boxing.form_report(body((0.29, 0.30)))                    # shoulders at y 0.30, width 0.12: wrists on the shoulder line
    t_mid, d_mid = boxing.form_report(body((0.31, 0.31)))                 # just under the line: what a straight punch looks like
    check(d_line['hands'] == 'up' and d_mid['hands'] == 'mid' and 'punching' in t_mid, f"guard on the shoulder line: {d_line['hands']}; just under it: {d_mid['hands']}, not called low")

    print('[C] the session on a real clip: intake -> lesson (no counting) -> done')
    fr = read_frames(os.path.join(ROOT, 'data', 'tmp', 'mac_handoff', 'segments', 'run3_squats.mp4'))
    per = Perception(Config(plan='squats:5', intake=True), t0=0.0); evs = []; per.on_event(evs.append)
    for i in range(len(fr)):
        if i == 20: check(per.session.start_lesson() is None, 'start_lesson accepted in the intake')
        if i == 200: per.session.finish_lesson()
        per.process_frame(fr[i], t_recv=i / FPS, t_ms=i * 1000.0 / FPS)
    per.close()
    states = [e['state'] for e in evs if e['type'] == 'session']
    check(states == ['intake', 'lesson', 'done'] and not any(e['type'] == 'rep' for e in evs), f'states {states}, no reps counted during the lesson')
    check(per.session.start_lesson() is not None, 'start_lesson refused after the intake')

    print('[D] the worker: start_boxing_lesson moves the session, runs the timeline, makes the lesson card, ends in done')
    from facetime import worker
    from facetime.recap import CallRecap
    boxing.ROUND_S = 0.6; boxing.REST_S = 0.1
    class S:
        state = 'intake'; finished = False
        def start_lesson(self): return None
        def finish_lesson(self): S.finished = True
    class Per: session = S()
    class Voice:
        asks = []
        def ask_voice(self, p): self.asks.append(p); return 'spoken'
        def say_now(self, t): return None
        class summary: done_note = None
    class Streaks:
        def record(self, p): return dict(current=2, best=2, new_day=True)
    class J(worker.CoachJob):
        def __init__(self):
            self.per = Per(); self.pipe = None; self.voice = Voice(); self.lesson = None; self.lesson_task = None; self.recap = CallRecap()
            self.streaks = Streaks(); self.profile = 'default'; self.day = None; self.player = None
    j = J(); tool = j.boxing_tool(boxing)
    reply = await tool('cuban')
    check('Cuban school' in reply and j.lesson['style'] == 'cuban' and j.recap.plan_label == 'Cuban school boxing', f'tool reply: {reply[:60]!r}...')
    await asyncio.wait_for(j.lesson_task, 10)
    check(S.finished and j.recap.lesson['drills'] == [d[0] for d in boxing.STYLES['cuban']['drills']] and 'day 2' in (j.voice.summary.done_note or ''),
          f'lesson finished, drills recorded, close note: {(j.voice.summary.done_note or "")[:70]!r}...')
    check((await tool('soviet')).startswith('A lesson is already'), 'a second lesson is refused')
    from facetime.recap import card_image
    img = card_image(j.recap, j.day); check(img.size == (1080, 1350), 'lesson card renders')
    played = []
    class P:
        def say(self, text, intent, meta=None): played.append(text)
    j.player = P(); j.session_state = 'lesson'; j._voice_audible = lambda: False; j.deferred_cues = []; j.defer_task = None
    j._say_or_defer('Step back so I can see your whole body', 'visibility', {}); j._say_or_defer('Rest for a moment', 'positive', {})
    check(played == ['Rest for a moment'], f'visibility lines are skipped during a lesson, others play ({played})')

    n = len(PASS); ok = sum(PASS); print(f'\n{ok}/{n} checks passed' + ('' if ok == n else '  <-- FAILURES'), flush=True)
    return 0 if ok == n else 1


if __name__ == '__main__':
    rc = asyncio.run(amain()); sys.stdout.flush(); os._exit(rc)
