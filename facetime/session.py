"""Session script on top of FramePipeline: what the coach says beyond the per-rep Coach policy.

Modes (one Session per call, frame-synchronous at `fps`, timers in frame time so replay and live agree):
- plan:   --plan "squats:10,pushups:8,jumping jacks:20,lunges:10". Per exercise: "Let's do X" (then "Moving on to X"),
          wait until the rep counter sees movement (or 3 s), run the Coach for X counting every rep, at the target
          say "That's <n>, nice work", rest `rest_s` ("Rest for a moment"), next; end with "All done, great session".
          During the wait only the rep counter runs (FramePipeline warm-up), so the Coach and trigger record nothing.
          Plan and exercise names are validated in __init__ (clear SystemExit listing the accepted names).
          A set that does not reach its target in `set_timeout_s` is closed with the count reached.
- single: one exercise for the whole call (announces "Let's do X" on the first frame), no target.
- auto:   no plan, no exercise: the classifier's exercise head is EMA-smoothed and summed over exercise families
          (FAMILY_DEMO by default: the four demo exercises and their look-alikes; --auto-classes all = every LR2CLF
          class); a family above 0.5 for 3 s is declared ("Looks like X"), the Coach and counter switch to it; every
          10 s a challenger that has beaten the current family by 0.3 for 3 s takes over. The head is only ~0.75
          accurate on real sessions, so plan mode is the demo default.
Visibility (all modes, facetime/visibility.py): no pose for 3 s or head/ankles out of frame for 3 s pauses the Coach
(counter keeps its state) and the session clock, says "Step back so I can see your whole body" at most once per 20 s and
"Great, I can see you" when the whole body is back; plan mode adds "I will wait here, step into view when you are ready"
once after 30 s without any pose and keeps waiting.
Rep-aware scheduling (all modes): while FramePipeline reports rep_imminent (a RepCounter2 tracker recovering from its
extreme, or >= 0.8 x the median rep period since the last count) a correction or praise is held up to hold_s (0.7 s) so the
count is spoken first; the player also lets a count jump ahead of a pending correction.
Tempo (all modes): median period of the last 4 reps (RepCounter2 turning points when available) vs TEMPO_TARGET_S: "Slow down" below 0.6x,
"Pick up the pace" above 1.8x, at most once per 15 s, sent as a correction through the same CuePlayer.
Gesture command (opt-in, gesture_next=True, --gesture-next): FramePipeline sets r['gesture_next'] on the frame a T-pose has been
held 1.5 s (facetime/probe.py TPose). Plan mode: in 'active' the set is closed the way a reached target is, with the milestone
line "Okay, moving on" (then the rest and the next exercise as usual); in 'rest' the next exercise starts at once; in 'wait' the
Coach is armed. Single mode: "Okay, moving on", the Coach is detached and the session is done (the agent still leaves the room
on --once / --max-s as today). Auto mode ignores it (logged once). A held cue is dropped on every one of these boundaries.
Intake (opt-in, intake=True, the worker's --intake, plan mode only): once someone is in frame the session enters 'intake' and
says nothing; the voice asks the caller's goal and level and calls set_plan() (from any thread). The new plan starts on the
first frame with no conversation hold ("Let's do X"); with no set_plan within intake_timeout_s (hold time not counted) the
configured plan runs.
Lesson (start_lesson() / finish_lesson(), any thread; the voice's start_boxing_lesson tool, starter/boxing.py): from the intake the
session enters 'lesson' (no counting, no clock: the worker runs the lesson's timeline), then 'done' when the lesson ends. Both
transitions happen on the next frame even when the caller is out of view.
Out of view (lost_close_s, 15 s by default; 0 = off): a set whose caller has been out of view that long is closed with the count
reached, so the session moves to the rest (where the voice may speak) instead of waiting forever.
Auto-start (rest_autostart, on by default): rest_min_s (3 s) into a rest the next exercise's rep counter runs in warm-up; the first
rep it counts starts the set ("Moving on to X", that rep kept), so a caller who just starts moving is counted without saying "ready".
Next (request_next(), any thread; the voice's start_next_set tool): in 'rest' the next exercise starts, in 'wait' the Coach is armed,
on the first frame with no conversation hold (the voice finishes its sentence first).
Cues go out through say(text, intent, meta); intents: announce, milestone, correction, count, positive, start.
"""
import logging
import numpy as np
from facetime.cues_tts import number_word
from facetime.visibility import WAIT_HERE
from facetime.pipeline import spoken_name, map_exercise, accepted_names, LR2CLF, PLAIN

log = logging.getLogger('session')
MOVE_ON = 'Okay, moving on'                                     # the T-pose command's line (in the cue vocabulary, facetime/cues_tts.py)
TEMPO_TARGET_S = {'squats': 2.5, 'pushups': 2.0, 'jumping jacks': 0.9, 'lunges': 2.0,
                  'alternating forward lunges': 2.0, 'jumping lunges': 2.0, 'squat jump': 2.0, 'squat kick': 2.5,
                  'squat punch': 2.5, 'good morning': 2.5, 'floor touches': 2.0, 'toe touch': 2.0, 'plank taps': 1.5, 'puddle jump': 1.5}
ELIGIBLE = set(LR2CLF.values()) | set(PLAIN.values())          # classes auto mode may declare (--auto-classes all)
# Auto mode sums the exercise-head probabilities over a family before the 0.5 / 3 s rule. The head confuses
# variants of one movement (pushups seen from the side read as "elbow plank", walking lunges as "low lunge pose",
# squats as "squat kick"), so the demo default folds those into the four demo exercises; "all" keeps every
# LR2CLF class separate and only folds the obvious variants.
FAMILY_DEMO = {'squats': ['squats', 'squat kick', 'squat punch', 'squat jump', 'plie squat', 'squat jabs'],
               'pushups': ['pushups', 'pushups (on knees)', 'spider man pushup', 'yoga pushup', 'high plank', 'elbow plank', 'plank preparation'],
               'jumping jacks': ['jumping jacks', 'squat jacks'],
               'alternating forward lunges': ['alternating forward lunges', 'jumping lunges', 'lunges', 'lunges (left leg out in front)',
                                              'lunges (right leg out in front)', 'lunges stance', 'low lunge pose (left leg back)',
                                              'low lunge pose (right leg back)', 'curtsy lunges', 'alternating lateral lunge']}
FAMILY_ALL = {'pushups': ['pushups (on knees)', 'spider man pushup', 'yoga pushup'], 'squats': ['plie squat'], 'jumping jacks': ['squat jacks'],
              'alternating forward lunges': ['lunges', 'lunges (left leg out in front)', 'lunges (right leg out in front)', 'lunges stance',
                                             'low lunge pose (left leg back)', 'low lunge pose (right leg back)']}
def class_map(mode):
    """classifier class -> declared class for auto mode."""
    m = {} if mode == 'demo' else {c: c for c in ELIGIBLE}
    for dst, members in (FAMILY_DEMO if mode == 'demo' else FAMILY_ALL).items():
        for c in members: m[c] = dst
    return m

def parse_plan(s):
    """'squats:10, pushups:8' -> [('squats', 10), ('pushups', 8)]"""
    out = []
    for item in s.split(','):
        item = item.strip()
        if not item: continue
        name, _, n = item.rpartition(':')
        if not name or not n.strip().isdigit(): raise SystemExit(f'plan item {item!r} must be name:count, e.g. squats:10; accepted names: {", ".join(accepted_names())}')
        out.append((name.strip(), int(n)))
    return out

class Session:
    def __init__(self, pipe, say, fps=10, plan=None, exercise=None, announce=True, rest_s=5.0, set_timeout_s=75.0,
                 wait_s=3.0, tempo=None, tempo_every_s=15.0, detect_hold_s=3.0, detect_every_s=10.0, detect_margin=0.3, auto_classes='demo', wait_here_s=30.0, hold_s=0.7, clip_s=None,
                 gesture_next=False, intake=False, intake_timeout_s=60.0, lost_close_s=15.0, rest_autostart=True, rest_min_s=3.0):
        self.pipe = pipe; self.say_cb = say; self.fps = fps
        self.intake = bool(intake) and bool(plan); self.intake_timeout_s = intake_timeout_s; self.pending_plan = None
        self.lost_close_s = lost_close_s; self.lost_n = 0; self.next_requested = False; self.n_lost_closed = 0
        self.rest_autostart = rest_autostart; self.rest_min_s = rest_min_s; self.preloaded = None; self.n_autostart = 0
        self.lesson_requested = False; self.lesson_finished = False
        self.gesture_next = gesture_next; self.n_gesture = 0; self._gesture_auto_logged = False
        self.plan = parse_plan(plan) if plan else None; self.single = exercise; self.announce = announce
        self.mode = 'plan' if self.plan else ('single' if exercise else 'auto')
        for name in ([n for n, _ in self.plan] if self.plan else []) + ([exercise] if exercise else []):
            try: map_exercise(name, pipe.clf)                 # fail here, before any room is joined, not in the worker thread
            except ValueError as e: raise SystemExit(f'{e}') from None
        if self.plan and any(n <= 0 for _, n in self.plan): raise SystemExit(f'plan targets must be positive: {plan!r}')
        self.rest_s = rest_s; self.set_timeout_s = set_timeout_s; self.wait_s = wait_s
        self.tempo = dict(TEMPO_TARGET_S, **(tempo or {})); self.tempo_every_s = tempo_every_s
        self.detect_hold_s = detect_hold_s; self.every_s = detect_every_s; self.margin = detect_margin; self.cmap = class_map(auto_classes)
        self.i = -1; self.state = 'init'; self.k = -1; self.target = None; self.name = None
        self.t_state = 0.0; self.said_rest = False; self.last_count = 0; self.rep_t = []; self.last_tempo = -1e9
        self.ema = None; self.hold = (None, 0.0); self.chal = (None, 0.0); self.last_eval = 0.0; self.current = None
        self.done = False; self.timeline = []; self.new_rep = False; self.wait_here_s = wait_here_s; self.said_wait_here = False; self.t = 0.0
        self.hold_s = hold_s; self.held = None; self.count_said_t = -1.0; self.n_held = 0; self.n_dropped_held = 0; self.clip_s = clip_s; self.last_count_t = None

    # ---- helpers ----
    def say(self, text, intent, meta=None, **extra):
        m = dict(meta or {}); m.update(extra); m.update(session_state=self.state, session_exercise=self.pipe.exercise, count=self.last_count, t_session=round(self.t, 2))
        self.timeline.append((round(self.t, 2), intent, text)); self.say_cb(text, intent, m)

    def _enter(self, state):
        self.state = state; self.t_state = self.t

    def _start_exercise(self, name, line, meta):
        """Fresh Coach for `name` in warm-up (counter only) until the wait ends; the session script replaces the Coach's
        own start line, and nothing the Coach would say during the wait is generated or recorded. A counter already warmed up
        during the rest (auto-start) is kept with the reps it counted."""
        self.say(line, 'announce', meta)
        if self.preloaded == name and self.pipe.coach is not None: self.name = name          # auto-start: keep the counter and its reps
        else:
            self.pipe.set_exercise(name, params=dict(start_line=False), armed=False); self.name = name
            self.last_count = 0; self.rep_t = []
        self.preloaded = None; self._drop_held('exercise changed'); self._enter('wait')

    def _track_reps(self, r, meta):
        c = r['count']
        if c > self.last_count:
            # rep time = the counter's turning point when it exposes one (RepCounter2), else the frame the count landed on
            self.last_count = c; self.rep_t.append(r['turning_t'] if r.get('turning_t') is not None else self.t); self.new_rep = True; self.last_count_t = self.t

    def _tempo(self, meta):
        """Called after the frame's own cues so the count is spoken before "Slow down" / "Pick up the pace"."""
        if not self.new_rep: return
        self.new_rep = False
        if len(self.rep_t) < 4: return
        per = float(np.median(np.diff(self.rep_t[-4:])))
        key = spoken_name(self.pipe.exercise); target = self.tempo.get(key, self.tempo.get(self.pipe.exercise))
        if not target or self.t - self.last_tempo < self.tempo_every_s: return
        if per < 0.6 * target: self.say('Slow down', 'correction', meta, why=f'tempo {per:.2f}s/rep < 0.6x{target}'); self.last_tempo = self.t
        elif per > 1.8 * target: self.say('Pick up the pace', 'correction', meta, why=f'tempo {per:.2f}s/rep > 1.8x{target}'); self.last_tempo = self.t

    def _fits_before_next_count(self, text):
        """False when the clip could not finish (plus 0.15 s) before the next count is expected at the current rep cadence."""
        if self.clip_s is None or self.last_count_t is None or len(self.rep_t) < 3: return True
        per = float(np.median(np.diff(self.rep_t[-4:]))); d = self.clip_s(text)
        if not d or per <= 0: return True
        return (per - (self.t - self.last_count_t)) >= d + 0.15

    def _emit(self, r, meta, skip_count_at=None):
        """Counts go out at once. A correction or praise is held (up to `hold_s`) while the counter says a rep completion is
        imminent, or while the clip would not finish before the next count at the current cadence, so the count is spoken
        first and the cue right after it; a held cue is released when the count lands, when the reason clears, or when the
        hold expires (with a known cadence the hold may last up to one rep period, at most 2.5 s: a 1 s cue at a 1 s cadence only fits right after a count)."""
        # before the set's first count nothing is said about form: a correction waits for "One" (at most 2.5 s) so the first
        # count is immediate and the cue follows it, the way a coach says "One. Squat deeper."; if no count lands within the
        # hold the correction is DROPPED (a person still getting into position reads as a shallow squat: without the hold,
        # a caller still standing heard "Squat deeper"), and praise before the first rep is never spoken. Later reps use
        # the cadence rules below.
        first_rep = self.hold_s > 0 and self.state == 'active' and self.last_count == 0
        for ev in r['events']:
            if ev['intent'] == 'count':
                if skip_count_at is not None and r['count'] >= skip_count_at: continue
                self.say(ev['text'], ev['intent'], meta, why=ev['why'], coach_t=ev['t']); self.count_said_t = self.t
            elif ev['intent'] == 'positive' and first_rep:
                self.n_dropped_held += 1                                           # no praise for a set that has not started
            elif ev['intent'] in ('correction', 'positive') and self.hold_s > 0 and (first_rep or r.get('rep_imminent') or not self._fits_before_next_count(ev['text'])):
                if self.held is not None: self._drop_held('replaced by a newer cue')   # a newer cue replaces an older held one (the older is dropped, not spoken)
                self.held = (ev, dict(meta), self.t, first_rep); self.n_held += 1
            else:
                self.say(ev['text'], ev['intent'], meta, why=ev['why'], coach_t=ev['t'])
        if self.held is not None:
            ev, m0, t0, pre_first = self.held; fits = self._fits_before_next_count(ev['text'])
            per = float(np.median(np.diff(self.rep_t[-4:]))) if len(self.rep_t) >= 3 else None
            cap = 2.5 if pre_first else (max(self.hold_s, min(2.5, per)) if per else self.hold_s)   # wait at most one rep period for the count
            early = (not pre_first) and ((not r.get('rep_imminent') and fits) or (self.t - t0 >= self.hold_s and fits))
            if self.count_said_t == self.t or early:
                self._release(meta)
            elif self.t - t0 >= cap - 1e-6:
                if pre_first and self.last_count == 0: self._drop_held('no rep yet')   # held before the first count and none came: not spoken
                else: self._release(meta)

    def _drop_held(self, why):
        """A held cue that no longer belongs (exercise changed, set closed, replaced) is dropped, never spoken late."""
        if self.held is not None:
            self.held = None; self.n_dropped_held += 1

    def _release(self, meta):
        ev, m0, t0, _pre = self.held; self.held = None
        if ev['intent'] == 'positive' and self.t - t0 > 1.5:                 # praise is filler: late praise is dropped, not spoken
            self.n_dropped_held += 1; return
        self.say(ev['text'], ev['intent'], m0, why=ev['why'], coach_t=ev['t'], held_s=round(self.t - t0, 2))

    # ---- per frame ----
    def on_frame(self, r, meta=None):
        """r: FramePipeline.process() result. Call once per processed frame, in order. Frames with the body not fully
        visible do not advance the session clock (set timeouts, rests and waits pause with the Coach)."""
        meta = meta or {}
        if self.lesson_requested and self.state in ('init', 'intake'):   # a lesson instead of a workout (any visibility)
            self.lesson_requested = False; self._enter('lesson')
        if self.lesson_finished and self.state == 'lesson':
            self.lesson_finished = False; self.done = True; self._enter('done')
        for text in r.get('vis_cues', []):
            if not self.done: self.say(text, 'visibility', meta, why='visibility')   # after "All done" nobody needs to step back
        if not r.get('visible', True):
            self._drop_held('not visible')                                 # never replay a cue after a visibility pause
            self.lost_n += 1
            if self.mode == 'plan' and self.state == 'active' and self.lost_close_s and self.lost_n >= self.lost_close_s * self.fps:
                self.n_lost_closed += 1; c = self.last_count                # out of view too long: close the set, the rest lets the voice help
                self._close_set(meta, f"That's {number_word(min(c, 30))}, nice work" if c >= 1 else 'Nice work', f'out of view {self.lost_close_s:.0f}s')
            if self.mode == 'plan' and r.get('no_pose_s', 0.0) >= self.wait_here_s and not self.said_wait_here:
                self.said_wait_here = True; self.say(WAIT_HERE, 'visibility', meta, why=f'no pose for {self.wait_here_s:.0f} s')
            return
        if r.get('ok', True): self.said_wait_here = False
        self.lost_n = 0
        self.i += 1; self.t = self.i / self.fps
        if r.get('hold') and self.state in ('wait', 'rest', 'intake'): self.t_state += 1.0 / self.fps   # conversation hold (voice layer): the rest or wait does not run out while the caller and the coach are talking
        if self.gesture_next and r.get('gesture_next') and self._gesture(r, meta): return   # the command frame replaces the frame's own cues
        if self.state == 'lesson': return                                  # the worker runs the lesson; nothing per frame
        if self.mode == 'plan': self._plan(r, meta)
        elif self.mode == 'single': self._single(r, meta)
        else: self._auto(r, meta)

    def _gesture(self, r, meta):
        """The T-pose command (--gesture-next). Returns True when it acted (the frame's own events are then not processed):
        plan 'active' closes the set with "Okay, moving on", 'rest' goes straight to the next exercise, 'wait' arms the Coach;
        single mode says the line, detaches and ends the session; auto mode (and init/done) ignores it."""
        if self.mode == 'auto':
            if not self._gesture_auto_logged:
                self._gesture_auto_logged = True; log.warning('gesture-next: T-pose seen in auto mode, ignored (no plan to move along)')
            return False
        if self.mode == 'plan':
            if self.state == 'active':
                self.n_gesture += 1; self._close_set(meta, MOVE_ON, 'gesture'); return True
            if self.state == 'rest':
                self.n_gesture += 1; self._drop_held('gesture'); self._next_exercise(meta); return True
            if self.state == 'wait':
                self.n_gesture += 1; self._drop_held('gesture'); self.pipe.arm(); self._enter('active'); return True
            return False
        if self.mode == 'single' and self.state == 'active':
            self.n_gesture += 1; self.say(MOVE_ON, 'milestone', meta, why='gesture')
            self._drop_held('gesture'); self.pipe.detach(); self._enter('done'); self.done = True; return True
        return False

    def _single(self, r, meta):
        if self.state == 'done': return                                    # after the T-pose command (--gesture-next): nothing more to say
        if self.state == 'init':
            if self.announce: self.say(f"Let's do {self.single}", 'announce', meta)
            self.pipe.set_exercise(self.single); self._enter('active')      # Coach keeps its own "Let's go" start line
        self._track_reps(r, meta); self._emit(r, meta); self._tempo(meta)

    def _plan(self, r, meta):
        if self.state == 'init':
            if not r.get('ok', True) or r.get('vis_cues'): return   # start when someone is in frame; one frame after "Great, I can see you"
            if self.intake: self._enter('intake'); return
            self.k = 0; name, self.target = self.plan[0]; self._start_exercise(name, f"Let's do {name}", meta); return
        if self.state == 'intake':
            pending = self.pending_plan
            if pending is not None and not r.get('hold'):                  # chosen, and nobody is talking: start it
                self.pending_plan = None; self.plan, rest_s = pending
                if rest_s is not None: self.rest_s = rest_s
            elif self.t - self.t_state < self.intake_timeout_s: return
            else: log.warning('intake: no plan chosen in %.0f s, running the configured plan', self.intake_timeout_s)
            self.k = 0; name, self.target = self.plan[0]; self._start_exercise(name, f"Let's do {name}", meta); return
        if self.next_requested and self.state in ('wait', 'rest') and not r.get('hold'):    # the caller said they are ready
            self.next_requested = False; self._drop_held('next requested')
            if self.state == 'rest': self._next_exercise(meta); return
            self.pipe.arm(); self._enter('active'); self._track_reps(r, meta); return
        if self.state == 'wait':
            moving = r['moving'] or (not self.pipe.countable and r['ok'])
            if moving or self.t - self.t_state >= self.wait_s: self.pipe.arm(); self._enter('active')
            self._track_reps(r, meta)                 # a rep completed in warm-up is counted silently (the counter ran)
            return
        if self.state == 'active':
            self._track_reps(r, meta); self._emit(r, meta, skip_count_at=self.target); self._tempo(meta); self._check_target(meta); return
        if self.state == 'rest':
            if not self.said_rest and self.t - self.t_state >= 1.5:
                self.said_rest = True; self.say('Rest for a moment', 'positive', meta)
            nxt = self.plan[self.k + 1] if self.k + 1 < len(self.plan) else None
            if self.rest_autostart and nxt is not None:
                if self.preloaded is None and self.t - self.t_state >= self.rest_min_s:
                    self.pipe.set_exercise(nxt[0], params=dict(start_line=False), armed=False); self.preloaded = nxt[0]   # counter only, nothing said
                    self.last_count = 0; self.rep_t = []
                elif self.preloaded is not None:
                    self._track_reps(r, meta)
                    if self.last_count >= 1:                                        # a real rep of the next exercise: start the set now
                        self.n_autostart += 1; self._next_exercise(meta); return
            if self.t - self.t_state >= self.rest_s: self._next_exercise(meta)
            return

    def set_plan(self, plan, rest_s=None):
        """Intake: the plan the caller chose ("squats:12,pushups:10"), started on a later frame (any thread may call this; the
        frame thread picks it up). Returns None when accepted, else why not (a bad name, or the session is already past intake)."""
        if not self.intake or self.state not in ('init', 'intake'): return f'too late: the session is in {self.state}'
        try:
            items = parse_plan(plan)
            for name, _ in items: map_exercise(name, self.pipe.clf)
        except (SystemExit, ValueError) as e: return str(e)
        if not items or any(n <= 0 for _, n in items): return f'empty or non-positive plan: {plan!r}'
        self.pending_plan = (items, rest_s); return None

    def start_lesson(self):
        """The caller chose a boxing lesson in the intake (any thread). None when accepted, else why not."""
        if self.mode != 'plan' or self.state not in ('init', 'intake'): return f'too late: the session is in {self.state}'
        self.lesson_requested = True; return None

    def finish_lesson(self):
        """The lesson's timeline ended: 'done' on the next frame."""
        self.lesson_finished = True

    def request_next(self):
        """The caller is ready (any thread): skip the rest / start counting now. None when accepted, else why not."""
        if self.mode != 'plan' or self.state not in ('wait', 'rest'): return f'nothing to skip: the session is in {self.state}'
        self.next_requested = True; return None

    def _next_exercise(self, meta):
        """End of the rest (or the T-pose command during it): the next plan item, or the closing line."""
        self.k += 1
        if self.k < len(self.plan):
            name, self.target = self.plan[self.k]; self._start_exercise(name, f'Moving on to {name}', meta)
        else:
            self.say('All done, great session', 'announce', meta); self._enter('done'); self.done = True

    def _check_target(self, meta):
        c = self.last_count; timed_out = self.t - self.t_state >= self.set_timeout_s
        if c >= self.target or timed_out:
            self._close_set(meta, f"That's {number_word(min(c, 30))}, nice work" if c >= 1 else 'Nice work',
                            'target' if c >= self.target else f'set timeout {self.set_timeout_s}s')

    def _close_set(self, meta, line, why):
        """The set is over (target reached, timed out, or the T-pose command): milestone line, held cue dropped, Coach detached, rest."""
        self.say(line, 'milestone', meta, why=why)
        self._drop_held('set closed'); self.pipe.detach(); self.said_rest = False; self._enter('rest')
        if self.mode == 'plan' and self.plan and self.k + 1 >= len(self.plan): self._next_exercise(meta)   # the last set: "All done" at once, no rest

    def _auto(self, r, meta):
        if self.state == 'init':
            self.pipe.set_exercise(None, params=dict(start_line=False, praise_after=0)); self._enter('detect')
        if r['ex_probs'] is not None: self._update_ema(r['ex_probs'])
        if self.state == 'detect':
            cls, since = self.hold
            if cls is not None and self.t - since >= self.detect_hold_s: self._declare(cls, meta)
            return
        self._track_reps(r, meta); self._emit(r, meta); self._tempo(meta)
        if self.t - self.last_eval >= self.every_s:
            self.last_eval = self.t; cls, since = self.chal
            if cls is not None and self.t - since >= self.detect_hold_s: self._declare(cls, meta)

    def _update_ema(self, p):
        p = np.asarray(p, np.float32); self.ema = p if self.ema is None else 0.6 * self.ema + 0.4 * p
        fam = {}
        for j, c in enumerate(self.pipe.clf.exs):
            d = self.cmap.get(c)
            if d is not None: fam[d] = fam.get(d, 0.0) + float(self.ema[j])
        self.fam = fam; cls = max(fam, key=fam.get); top = fam[cls]
        ok = top > 0.5
        self.hold = (cls, self.hold[1] if (ok and self.hold[0] == cls) else self.t) if ok else (None, self.t)
        cur = self.current
        if cur is not None and cls != cur and top - fam.get(cur, 0.0) >= self.margin:
            self.chal = (cls, self.chal[1] if self.chal[0] == cls else self.t)
        else:
            self.chal = (None, self.t)

    def _declare(self, cls, meta):
        self.say(f'Looks like {spoken_name(cls)}', 'announce', meta, why=f'family prob {self.fam.get(cls, 0.0):.2f}')
        self.pipe.set_exercise(cls, params=dict(start_line=False)); self.current = cls; self.name = spoken_name(cls)
        self.last_count = 0; self.rep_t = []; self.chal = (None, self.t); self.last_eval = self.t; self._drop_held('exercise changed'); self._enter('active')
