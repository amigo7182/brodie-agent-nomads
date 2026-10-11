"""Your hook into the coach's event stream (docs/STARTER_EVENTS.md): the worker calls on_event for every event, in order.

on_event(event, say_now, ask_voice)
  event      one event dict (type rep | exercise_seen | fault | cue | visibility | gesture | probe | session | status | voice)
  say_now    say_now(text, intent='other') -> record: a pre-rendered bank string through the fast path, about 0.2 s to heard.
             Only strings in the bank play (facetime/cues_tts.py vocabulary(): counts, "Nice work", "Sink lower", ...; a missing
             string is a record with status "missing"). Priority 'other' never speaks over a count; 'announce' replaces a pending cue.
  ask_voice  ask_voice(prompt) -> 'spoken' | 'queued' | None: a line from the Gemini voice (about 1 s to first audio). Spoken at
             once while the floor is open (session state wait, rest or done, or a few seconds after a gesture or an armed probe),
             else queued for the next opening; a queued prompt is dropped when a set starts before the opening or when it is
             older than 15 s at the opening (a stale answer is worse than silence). None when the worker runs without
             --voice gemini (this file still loads: say_now works, ask_voice does nothing).
The fast path owns the set: the first count of the next set cuts the voice (the reply is interrupted, its buffered audio
dropped), so whatever you ask at a rest has about 5 s (the rest) plus the wait before the first rep to be heard. Ask for one
short sentence, not a speech. Runs on the worker's event loop: return quickly, never block. Exceptions are logged and ignored.
"""


STATE = {'target': 0, 'sets': 0}
# Bank lines (src/rtcoach/cues.json "brodie extras - ...", rendered into data/cues_audio_gemini): say_now plays them right after the count.
HALFWAY = ('Halfway there', 'Light weight, keep pushing')


def on_event(event, say_now, ask_voice):
    """Default: ONE short sentence from the voice at each rest (the set that just ended, what comes next) and a warm close at
    the end; nothing anywhere else (the fast path counts, corrects and announces). The set summary (reps, corrections spoken,
    the next item) is already in the model's context when this runs. After the last set the session goes straight to `done`
    (no rest); the `items` check below also keeps a final rest silent if one ever comes.
    With --intake the call opens in `intake`: the greeting asks the goal and level, and the voice picks the plan (starter/programs.py)."""
    t = event['type']
    if t == 'session' and event.get('state') == 'wait':
        STATE['target'] = event.get('target') or 0
    elif t == 'rep' and STATE['target']:
        n, target = event.get('n', 0), STATE['target']
        if n == target - 1 and target >= 4: say_now('Last one')
        elif n == target - 2 and target >= 10: say_now('Two more')
        elif n == target // 2 and target >= 8: say_now(HALFWAY[STATE['sets'] % 2])
    if t == 'session' and event.get('state') == 'intake':
        ask_voice('The caller just arrived: greet them warmly, say this call is coached by software that reads their body pose and keeps '
                  'a log of rep counts, not the video or audio, then ask their goal (strength, cardio or full body) and how hard (easy, medium or hard), '
                  'or whether they would rather have a boxing lesson in the Soviet, Mexican or Cuban style.')
    elif t == 'session' and event.get('state') == 'rest':
        n = event.get('items')                                           # the plan's length (it may come from the intake)
        if n is not None and event.get('item') is not None and event['item'] >= n - 1: return
        STATE['sets'] += 1
        tip = ' Then add one short, general recovery or hydration tip.' if STATE['sets'] % 2 == 0 else ''   # every other rest: a tip
        ask_voice('The set just ended: in ONE short sentence, say how it went and name what comes next, and tell them to just start '
                  'when they are ready (or say "ready").' + tip)
    elif t == 'session' and event.get('state') == 'done':
        ask_voice('The whole session is over: close it warmly in two short sentences, suggest one easy 30 second stretch for the muscles they worked, '
                  'and mention their streak if the notes give one.')
    # Ideas: say_now('Nice work') on a rep event with confidence agree 1.0 at the halfway mark (a bank string, 0.2 s, never over a
    # count); ask_voice(...) on a gesture event (name tpose: the caller is asking for the next thing; the floor is open 4 s).
