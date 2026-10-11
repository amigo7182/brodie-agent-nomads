"""facetime/recap.py and the handler's bank lines, offline: the streak rules, the recap from events, the results card, the profile
key, and starter/handlers.py on a recorded call (Halfway / Two more / Last one / tips), plus the worker's session tools.

Run:  .venv/bin/python scripts/test_recap.py
"""
import asyncio, datetime, importlib.util, json, os, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from PIL import Image
from facetime.recap import Streaks, CallRecap, render_card, profile_key

PASS = []
def check(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg, flush=True); PASS.append(bool(cond)); return cond


def main():
    d0 = datetime.date(2026, 10, 9); tmp = tempfile.mkdtemp()
    print('[A] streaks: +1 on consecutive days, unchanged twice a day, back to 1 after a gap, per profile, best kept')
    st = Streaks(os.path.join(tmp, 'streaks.json'))
    r1 = st.record('p', d0); r2 = st.record('p', d0); r3 = st.record('p', d0 + datetime.timedelta(1)); r4 = st.record('p', d0 + datetime.timedelta(2))
    check((r1['current'], r1['new_day'], r2['current'], r2['new_day'], r3['current'], r4['current']) == (1, True, 1, False, 2, 3), f'1, 1 (same day), 2, 3: {r1["current"]}, {r2["current"]}, {r3["current"]}, {r4["current"]}')
    r5 = st.record('p', d0 + datetime.timedelta(5)); check(r5['current'] == 1 and r5['best'] == 3, f'after a gap: day 1, best 3 ({r5})')
    check(st.peek('p', d0 + datetime.timedelta(7))['current'] == 0 and st.peek('other', d0)['current'] == 0, 'a lapsed streak peeks as 0; another profile is separate')

    print('[B] recap from events, and the card')
    r = CallRecap()
    for e in [dict(type='session', state='intake'), dict(type='session', state='wait', exercise='squats', target=7), dict(type='session', state='active', exercise='squats', target=7),
              dict(type='cue', intent='correction', text='Hips back'), dict(type='cue', intent='correction', text='Hips back'), dict(type='cue', intent='correction', text='Sink lower'),
              dict(type='session', state='rest', count=7), dict(type='session', state='wait', exercise='pushups', target=6), dict(type='session', state='done', count=4)]: r.on_event(e)
    check([(s['exercise'], s['reps'], s['target']) for s in r.sets] == [('squats', 7, 7), ('pushups', 4, 6)] and r.total_reps == 11 and r.focus == 'Hips back',
          f'sets {r.sets}, total {r.total_reps}, focus {r.focus!r}')
    path = render_card(r, dict(current=3, best=5), os.path.join(tmp, 'card.png'))
    im = Image.open(path); check(im.size == (1080, 1350), f'card written {im.size} at {path}')

    print('[C] profile key from Chert metadata')
    check(profile_key({}) == 'default' and profile_key({'chert_call': {'from_handle_hash': 'abc'}}) == 'handle:abc'
          and profile_key({'chert_call': json.dumps({'from_handle_hash': 'x1'})}) == 'handle:x1' and profile_key({'chert_call': {'from_handle_hash': None}}) == 'default', 'hash when present, else default')

    print('[D] starter/handlers.py on a recorded call: bank lines on the right reps, a tip every other rest, all lines in the bank')
    spec = importlib.util.spec_from_file_location('h', os.path.join(ROOT, 'starter', 'handlers.py')); h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
    said, asked = [], []
    evs = [dict(type='session', state='wait', target=10)] + [dict(type='rep', n=n) for n in range(1, 11)] + [dict(type='session', state='rest', item=0, items=3)] \
        + [dict(type='session', state='wait', target=7)] + [dict(type='rep', n=n) for n in range(1, 8)] + [dict(type='session', state='rest', item=1, items=3)] \
        + [dict(type='session', state='wait', target=5)] + [dict(type='rep', n=n) for n in range(1, 6)] + [dict(type='session', state='rest', item=2, items=3), dict(type='session', state='done')]
    for e in evs: h.on_event(e, lambda text, intent='other': said.append(text), lambda p: asked.append(p) or 'spoken')
    check(said == ['Halfway there', 'Two more', 'Last one', 'Last one', 'Last one'], f'bank lines {said}')
    check(len(asked) == 3 and 'tip' not in asked[0] and 'tip' in asked[1] and 'stretch' in asked[2], f'asks: rest (no tip), rest (tip), done (stretch); the last rest is silent ({len(asked)} asks)')
    bank = json.load(open(os.path.join(ROOT, 'data', 'cues_audio_gemini', 'manifest.json')))
    check(all(t in bank for t in set(said) | set(h.HALFWAY)), 'every handler line is in the audio bank')

    print('[E] the worker\'s session tools: check_view answers from the tracker, start_next_set reaches the session')
    from facetime import worker
    class S:
        def __init__(self): self.calls = 0
        def request_next(self): self.calls += 1; return None
    class Per: session = S(); _vis_emitted = 'partial'; _paused = True
    class J:
        per = Per(); first_frame_t = 1.0
    nxt, view, *_ = worker.CoachJob.session_tools(J())
    v = asyncio.run(view()); n = asyncio.run(nxt())
    check('part of the body' in v and 'paused' in v and J.per.session.calls == 1 and 'next exercise' in n, f'check_view: {v[:60]!r}...; start_next_set: {n[:40]!r}')

    n = len(PASS); ok = sum(PASS); print(f'\n{ok}/{n} checks passed' + ('' if ok == n else '  <-- FAILURES'), flush=True)
    return 0 if ok == n else 1


if __name__ == '__main__':
    rc = main(); sys.stdout.flush(); os._exit(rc)
