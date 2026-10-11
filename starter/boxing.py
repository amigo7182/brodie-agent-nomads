"""Boxing lessons: Brodie teaches a style (Soviet, Mexican, Cuban) as timed shadowboxing rounds on the call. Edit freely.

How it runs (worker --intake): the intake offers a boxing lesson next to the workouts; the voice calls start_boxing_lesson(style);
the worker then runs run_lesson() below. Each round: Brodie explains the drill, the caller shadowboxes for ROUND_S seconds, halfway
through Brodie gives ONE cue from what the pose tracker measures (form_report: guard height, stance width), then a rest.
No punch detector exists: feedback is about the guard, the stance and the style's principles, never about punch counts.
Shadowboxing only: no sparring advice, no equipment needed.
"""
import asyncio, math

ROUND_S = 45          # one drill per round
REST_S = 15
# Instant cues from the tracker during a round (pre-recorded, about 0.2 s): played when a problem holds for a moment and Brodie is quiet.
FAST_CUES = dict(hands='Hands up', stance='Widen your stance', good='Good guard, keep it up')
FAST_GAP_S = 6.0      # at least this long between two instant cues

STYLES = {
    'soviet': dict(
        title='Soviet school',
        idea='hit and do not get hit: control distance with the jab and footwork, score with straight punches, step out before the answer comes',
        stance='fairly upright and long, slightly side-on, weight balanced on both feet so you can move in and out at once',
        guard='rear hand by the chin, lead hand out in front and busy with the jab, chin down, elbows in',
        drills=[
            ('step-in jab, step back out', 'step in with the lead foot as the jab lands, then push straight back out to range', 'feet back to range after every jab, rear hand stays at the chin'),
            ('jab-cross, then step out', 'jab, rear straight with the hip turning, then step back or to the side', 'both punches straight down the middle, hands back to the face'),
            ('side step and jab', 'small step to the side with the lead foot leading, then a jab from the new angle', 'stance stays the same width while moving; never cross the feet'),
        ]),
    'mexican': dict(
        title='Mexican style',
        idea='pressure: walk forward behind a tight high guard, work the body, throw combinations and keep the other person on the back foot',
        stance='a bit lower and more square on, knees bent, ready to dig in and throw with both hands',
        guard='high and tight: gloves by the temples, elbows down covering the ribs, chin tucked',
        drills=[
            ('walk forward behind the high guard, jab', 'small steps forward, guard high, double up the jab as you close the distance', 'elbows stay glued to the ribs, chin tucked'),
            ('jab, cross, lead hook to the body', 'bend the knees to drop level for the body hook, do not just lean over', 'come back up with the guard high straight after the body shot'),
            ('roll under, hook to the body, hook to the head', 'roll under an imaginary punch by bending the knees, come up with a body hook then a head hook', 'eyes stay forward during the roll, hands stay up'),
        ]),
    'cuban': dict(
        title='Cuban school',
        idea='rhythm and deception: bounce on the balls of the feet, feint constantly, fast hands, then pivot to a new angle',
        stance='light and springy on the balls of the feet, relaxed shoulders, slightly side-on',
        guard='relaxed and ready: rear hand by the chin, lead hand a little lower and moving to feint',
        drills=[
            ('rhythm bounce and feint the jab', 'stay light and bouncy, show the jab with a shoulder twitch without throwing it', 'shoulders relaxed, rear hand never leaves the chin'),
            ('feint, then double jab and cross', 'feint first, then fire a fast double jab and a rear straight', 'speed over power, snap the hands back'),
            ('jab and pivot off the line', 'jab, then pivot on the lead foot and swing the back foot round to a new angle', 'finish the pivot in your stance with the guard up'),
        ]),
}
STYLE_NAMES = tuple(STYLES)

BOXING_PROMPT = (
    "\n\nYou can also teach a boxing lesson, shadowboxing only (no sparring, no equipment). Offer it in the intake next to the "
    "workouts: a Soviet, Mexican or Cuban style lesson. When the caller picks one, call the start_boxing_lesson tool with the style; "
    "the lesson then runs in timed rounds and you get a coach note before each part. What the styles are about: "
    + ' '.join(f"{s['title']}: {s['idea']}; stance {s['stance']}; guard {s['guard']}." for s in STYLES.values())
    + " Say lead hand and rear hand, never left or right, so it works for orthodox and southpaw. Remind them once to keep the elbows "
      "soft (never snap the arm straight) and to clear some space. You cannot see the video: the pose tracker reports the guard and the "
      "stance, so only correct those from its notes, and call check_form when the caller asks how their form looks."
)


def form_report(lm):
    """MediaPipe landmarks (33 x [x, y, z, visibility, ...], image-normalised, y down) -> (short text, dict) about the guard and
    the stance; text None when the body is not clearly in view."""
    if lm is None: return None, {}
    def p(i): return lm[i][0], lm[i][1], (lm[i][3] if len(lm[i]) > 3 else 1.0)
    nose, ls, rs, lw, rw, la, ra = p(0), p(11), p(12), p(15), p(16), p(27), p(28)
    if min(v for _, _, v in (ls, rs, lw, rw)) < 0.5: return None, {}
    sh_y = (ls[1] + rs[1]) / 2; sh_w = math.dist(ls[:2], rs[:2]) or 1e-6
    low = [w for w in (lw, rw) if w[1] > sh_y + 0.15 * sh_w]          # a wrist clearly below the shoulders
    high = all(w[1] < sh_y + 0.02 * sh_w for w in (lw, rw))           # both wrists at or above the shoulder line: a guard (a straight punch is at shoulder height too)
    d = dict(hands='up' if high else ('one low' if len(low) == 1 else ('both low' if low else 'mid')))
    parts = [{'up': 'both hands are up by the face', 'one low': 'one hand has dropped below the shoulders',
              'both low': 'both hands are down below the shoulders',
              'mid': 'the hands are around shoulder height (normal while punching; just remind them to bring the hands back to the face after each punch)'}[d['hands']]]
    if min(la[2], ra[2]) >= 0.5:
        ratio = abs(la[0] - ra[0]) / sh_w; d['stance_ratio'] = round(ratio, 2)
        parts.append('the feet are close together (narrow stance)' if ratio < 0.7 else
                     ('the feet are very wide apart' if ratio > 2.2 else 'the stance width looks fine'))
    return 'The tracker sees: ' + '; '.join(parts) + '.', d


async def run_lesson(style, coach):
    """The lesson's timeline. coach: ask(prompt) (the voice), say(text) (a bank line), wait_quiet(max_s), status(text) (the screen),
    form() -> text or None, log(text), round_s, rest_s. Returns the drills taught."""
    s = STYLES[style]; drills = s['drills']; taught = []
    await coach.wait_quiet(25)
    for i, (name, how, watch) in enumerate(drills, 1):
        coach.status(f'round {i} of {len(drills)}')
        coach.ask(f"Round {i} of {len(drills)}: {name}. Explain it in two short sentences ({how}), then say 'Go!'.")
        await coach.wait_quiet(20)
        loop = asyncio.get_running_loop(); t0 = loop.time(); half = False
        low_since = narrow_since = up_since = None; last_fast = -1e9; praised = False
        while (left := coach.round_s - (loop.time() - t0)) > 0:
            now = loop.time()
            if getattr(coach, 'skip', lambda: False)():                   # the caller asked to move on: end this round now
                coach.log(f'round {i} ended early (caller asked)'); break
            coach.status(f'round {i} · {int(left) // 60}:{int(left) % 60:02d}')
            d = coach.form_data() if hasattr(coach, 'form_data') else {}          # instant cues straight from the pose tracker
            hands = d.get('hands'); ratio = d.get('stance_ratio')
            low_since = (low_since or now) if hands in ('one low', 'both low') else None
            narrow_since = (narrow_since or now) if (ratio is not None and ratio < 0.7) else None
            up_since = (up_since or now) if hands == 'up' else None
            if now - last_fast >= FAST_GAP_S and now - t0 > 3:
                cue = (FAST_CUES['hands'] if low_since and now - low_since >= 1.5 else
                       FAST_CUES['stance'] if narrow_since and now - narrow_since >= 3 else
                       FAST_CUES['good'] if (not praised and up_since and now - up_since >= 8) else None)
                if cue and coach.say(cue) is not None:
                    last_fast = now; praised = praised or cue == FAST_CUES['good']
            if not half and left <= coach.round_s / 2:
                half = True; f = coach.form()
                coach.ask(f"Mid-round of {name}. {f or 'The tracker cannot see the whole body right now.'} Give ONE short cue: "
                          f"correct the guard or stance if the tracker shows a problem, otherwise say: {watch}.")
            await asyncio.sleep(0.5)
        taught.append(name); coach.log(f'round {i} done: {name}')
        if i < len(drills):
            coach.status(f'rest · next: {drills[i][0]}')
            coach.ask(f"Time! Round {i} done. One short line of praise, then tell them the next drill is {drills[i][0]}; rest {coach.rest_s} seconds, "
                      "or they can say they are ready.")
            t_rest = loop.time()
            while loop.time() - t_rest < coach.rest_s:                      # the rest, cut short by "I'm ready"
                if getattr(coach, 'skip', lambda: False)(): coach.log('rest ended early (caller is ready)'); break
                await asyncio.sleep(0.25)
    coach.status('lesson done')
    return taught
