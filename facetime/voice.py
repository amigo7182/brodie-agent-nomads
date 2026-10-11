"""The conversational layer of the hackathon starter: Gemini Live (voice Puck) in the SAME worker participant, gated to the gaps
between sets and mixed into the cue player's single audio track (docs/STARTER_EVENTS.md section 4, reports/STARTER_PREREQS_REVIEW.md
section 2: reflexes on the fast path, conversation on the LLM).

Two rules, enforced in code (Floor), not in the prompt:
 (a) the model may speak only while the floor is open: session state wait, rest or done, or for window_s after a gesture or probe
     event outside an active set (a gesture or probe during a set is logged, not acted on; a probe opens a window only when the
     worker runs the latency probe), or when the caller asks something during a rest (the caller's audio reaches the model only
     while the floor is open). The cut is hard (Mac test 2026-10-09, section 6: the voice was heard over the counts): when the
     floor closes the caller's audio is not forwarded (AgentSession.input.set_audio_enabled(False)), the reply in flight is
     interrupted (AgentSession.interrupt(force=True)), the sink's buffer and the MixInput queue are cleared, and MixerAudioOutput
     drops every frame that still arrives while the floor is closed ("[voice] cut: N frames dropped (floor closed)" once per reply).
     The window after a gesture or probe is re-checked on every event and on the idle tick. ask_voice() while the floor is closed is
     queued for the next opening, but a queued ask older than ASK_TTL_S is dropped at the opening and the whole queue is dropped
     when a set starts (session state active): a late answer to a stale prompt is worse than silence.
 (b) all model audio goes through the cue player's one rtc.AudioSource: MixerAudioOutput (a livekit.agents AudioOutput) is set as
     session.output.audio before start, RoomOptions(audio_output=False) publishes no second track, and the frames go into
     CuePlayer.attach_mixer()'s MixInput, which the player's pump pushes between clips and ducks by about 10 dB under a clip.
Context goes to the model as text at slot openings only (SetSummary: exercise, reps counted, cues given, the next item), never per
rep; a summary is appended to the chat context (no reply is forced), ask_voice(prompt) asks for one. Context produced before the
model session exists is not a backlog: only the latest line is sent when the session starts.
Logging: "[voice] speaking (floor open|closed, state X)" when the first audio frame of a reply reaches the sink (the transcript line
"[voice] coach: ..." arrives later, after the audio), "[voice] cut: ..." when frames were dropped, "[voice] dropped queued ask ...".

Extension points (starter/): persona.md (the prompt), plan.txt (the session script), handlers.py with
on_event(event, say_now, ask_voice): say_now(text) speaks a bank string through the fast path (about 0.2 s; counts take precedence),
ask_voice(prompt) asks the model (about 1 s to first audio; only when the floor is open, else queued for the next slot).

Beyond reps (the worker's --perception off, docs/STARTER_README.md "Your own coach: beyond reps"): Floor(always_open=True)
(--floor open, the default without perception) never closes, so the caller's audio is always forwarded, ask_voice fires at
once and nothing is cut; VideoTee (--vision) is a livekit.agents VideoInput set as session.input.video, fed by the worker's
read loop at about 1 fps (at most 2), pulled by the framework's forward task into the realtime session's push_video (JPEG).

Model turns are logged as events of type "voice" {text, t, role} (role coach or caller) through the worker's sinks.
GeminiModel wraps the livekit-agents AgentSession with google.beta.realtime.RealtimeModel; the floor logic takes any object with
set_input_enabled / generate / interrupt / add_context (the tests use a fake). Env: GEMINI_API_KEY (or GOOGLE_API_KEY),
GEMINI_LIVE_MODEL (default DEFAULT_MODEL), GEMINI_VOICE (default Puck), GEMINI_LANGUAGE (default en-US: the model answered in
German once on the 10-09 call; the persona says English too).
"""
import asyncio, importlib.util, logging, os, time
import numpy as np
from facetime import ROOT, SAMPLE_RATE

log = logging.getLogger('voice')
STARTER_DIR = os.path.join(ROOT, 'starter')
PERSONA_PATH = os.path.join(STARTER_DIR, 'persona.md')
PLAN_PATH = os.path.join(STARTER_DIR, 'plan.txt')
HANDLERS_PATH = os.path.join(STARTER_DIR, 'handlers.py')
DEFAULT_MODEL = 'gemini-3.8-live'          # measured here 0.9 to 1.3 s to first audio on a text turn; gemini-2.5-flash-native-audio-preview-12-2025 2.5 to 3.0 s
DEFAULT_VOICE = 'Puck'                     # the voice of the cue bank (data/cues_audio_gemini)
COACH_NOTE = 'Coach note (from the coaching system, not the caller; never read it aloud or mention it): '
OPEN_STATES = ('intake', 'lesson', 'wait', 'rest', 'done')     # the session states in which the model may speak (intake: --intake, before the first set)
WINDOW_S = 4.0                             # a gesture or probe opens the floor this long (never during an active set)
QUEUE_MAX = 3                              # ask_voice prompts kept while the floor is closed (oldest dropped)
ASK_TTL_S = 15.0                           # a queued ask older than this is dropped at the opening (a stale answer is worse than silence)
DEFAULT_LANGUAGE = 'en-US'                 # RealtimeModel(language=...): BCP-47; "" or env GEMINI_LANGUAGE=off leaves it to the model
DEFAULT_PERSONA = """You are Brodie, a warm, brief fitness coach on a video call.
Speak in one or two short sentences. Never count reps and never give form corrections yourself: the coach's fast path counts
and corrects in real time, and you only talk in the breaks between sets. You know the plan and the live state of the session
from the context lines you receive. Be encouraging, specific and quick; if asked a question, answer it directly."""


def read_text(path, default=''):
    try:
        with open(path) as f: return f.read().strip() or default
    except OSError:
        return default


def load_handlers(path=HANDLERS_PATH):
    """starter/handlers.py (or any path): the module's on_event(event, say_now, ask_voice); None when the file has no on_event."""
    if not path or not os.path.exists(path): return None
    spec = importlib.util.spec_from_file_location('starter_handlers', path); mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, 'on_event', None)
    if fn is None: log.warning('%s has no on_event(event, say_now, ask_voice)', path)
    return fn


def parse_plan(plan):
    """'squats:10,pushups:8' -> [('squats', 10), ...]; tolerant of spaces and a missing count."""
    out = []
    for item in (plan or '').split(','):
        item = item.strip()
        if not item: continue
        name, _, n = item.partition(':')
        try: out.append((name.strip(), int(n)))
        except ValueError: out.append((name.strip(), None))
    return out


class Floor:
    """Rule (a). on_event(ev) and tick() apply the state to the model: open -> caller audio forwarded, queued asks fired;
    close -> audio dropped, the current reply interrupted, the sink cut (sink.cut(): buffer and MixInput cleared, later frames
    dropped). The gesture/probe window is checked on EVERY event (not only on the idle tick), never opens during an active set,
    and a probe opens it only with probe=True (the worker's --latency-probe). `now` is injectable for tests."""
    def __init__(self, model, window_s=WINDOW_S, queue_max=QUEUE_MAX, ask_ttl_s=ASK_TTL_S, probe=True, now=time.monotonic, always_open=False):
        """always_open=True (the worker's --floor open, the default with --perception off): the floor is open from the start and
        never closes; a session, gesture or probe event changes nothing, every ask is spoken at once, nothing is ever cut."""
        self.model = model; self.window_s = window_s; self.queue_max = queue_max; self.ask_ttl_s = ask_ttl_s; self.probe = probe; self.now = now
        self.always_open = bool(always_open)
        self.state = None; self.open = self.always_open; self.window_until = -1.0; self.queue = []; self.n_open = 0; self.n_close = 0
        self.opened_by = 'always' if self.always_open else None
        self.sink = None; self.t_ask = None; self.n_asked = 0; self.n_dropped_asks = 0; self.n_ignored = 0
        self.player = None; self.input_on = False; self.quiet_since = None; self.n_gated = 0   # echo guard: ears off while any Brodie audio plays (2026-10-10)
        self.guard_s = 0.3

    def on_event(self, ev):
        """Returns 'open' / 'close' when the floor changed on this event, else None (always None with always_open)."""
        t = ev.get('type')
        if self.always_open:
            if t == 'session': self.state = ev.get('state')
            return None
        if t == 'session':
            state = ev.get('state')
            if state == 'active' and self.state != 'active': self._drop_queue('a set started')
            self.state = state
        elif t in ('gesture', 'probe'):
            if self.state == 'active' or (t == 'probe' and not self.probe): self.n_ignored += 1      # logged by the worker, not acted on
            else: self.window_until = self.now() + self.window_s
        return self._apply(t)

    def tick(self):
        r = self._apply('tick'); self.gate_input(); return r

    def _audio_busy(self):
        """True while a bank clip plays or is pending, in the hold after one, or while the model's own reply is still buffered."""
        pl = self.player
        if pl is not None:
            if pl.playing is not None or pl.pending is not None or pl.urgent is not None: return True
            mix = getattr(pl, 'mix', None)
            if mix is not None and (mix.pending > 0 or self.now() < mix.duck_until): return True
        return False

    def gate_input(self):
        """Echo guard: with the floor open, the caller's audio reaches the model only when nothing of Brodie's is playing and has not
        been for guard_s (a phone on speaker feeds his own counts and replies back through the mic; without this the model heard
        itself). Costs barge-in while he talks; the floor's own open/close is untouched."""
        if not self.open or self.player is None: return
        if self._audio_busy():
            self.quiet_since = None
            if self.input_on: self.input_on = False; self.n_gated += 1; self.model.set_input_enabled(False)
            return
        if self.quiet_since is None: self.quiet_since = self.now()
        if not self.input_on and self.now() - self.quiet_since >= self.guard_s:
            self.input_on = True; self.model.set_input_enabled(True)
        if self.queue: self._fire_queue()                                   # an ask that waited for Brodie to finish talking

    def _apply(self, why):
        want = self.always_open or (self.state in OPEN_STATES) or (self.now() < self.window_until)
        if want == self.open: return None
        self.open = want
        if want:
            self.n_open += 1; self.opened_by = why if why != 'session' else self.state
            if self.player is None or not self._audio_busy(): self.input_on = True; self.model.set_input_enabled(True)
            else: self.input_on = False; self.quiet_since = None                   # ears open once Brodie's audio is done (gate_input)
            self._fire_queue(); return 'open'
        self.n_close += 1
        self.input_on = False; self.model.set_input_enabled(False); self.model.interrupt()
        if self.sink is not None: self.sink.cut()
        return 'close'

    def ask(self, prompt):
        """'spoken' when the model is asked now, 'queued' when it waits for the next opening."""
        if self.open and not self._audio_busy(): self._generate(prompt); return 'spoken'
        self.queue.append((self.now(), prompt))                       # closed, or Brodie is still talking (one voice at a time): fire when quiet
        for _, old in self.queue[:-self.queue_max]: self._log_drop(old, f'queue full ({self.queue_max})')
        del self.queue[:-self.queue_max]
        return 'queued'

    def _generate(self, prompt):
        self.t_ask = self.now(); self.n_asked += 1; self.model.generate(prompt)

    def _fire_queue(self):
        if not self.queue: return
        now = self.now(); keep = []
        for t0, prompt in self.queue:
            if now - t0 > self.ask_ttl_s: self._log_drop(prompt, f'{now - t0:.0f} s old, limit {self.ask_ttl_s:.0f} s')
            else: keep.append(prompt)
        self.queue = []
        if keep: self._generate('\n'.join(keep))

    def _drop_queue(self, why):
        for _, prompt in self.queue: self._log_drop(prompt, why)
        self.queue = []

    def _log_drop(self, prompt, why):
        self.n_dropped_asks += 1
        print(f'[voice] dropped queued ask ({why}): {prompt[:80]!r}', flush=True)


class SetSummary:
    """The context lines fed at slot openings: what the fast path did in the set that just ended, and what comes next."""
    def __init__(self, plan=None):
        self.plan = parse_plan(plan); self.reset(); self.sets = []; self.item = None; self.target = None
        self.done_note = None                                              # set by the worker once the streak is known (facetime/recap.py)
        self.rest_s = 5.0                                                  # the session's rest (the intake may change it), set before each text()

    def reset(self):
        self.exercise = None; self.reps = 0; self.faults = {}; self.cues = []; self.why = None; self.open = False

    def on_event(self, ev):
        t = ev['type']
        if t == 'session':
            if ev.get('state') == 'wait' and ev.get('item') == 0: self.sets = []        # a fresh plan (the intake may have replaced the configured one)
            if ev.get('exercise'): self.exercise = ev['exercise']
            if ev.get('state') in ('wait', 'active'): self.open = True
            if ev.get('item') is not None: self.item = ev['item']
            if ev.get('target') is not None: self.target = ev['target']
            if ev.get('state') == 'rest' or ev.get('done'): self.reps = max(self.reps, ev.get('count') or 0)
        elif t == 'rep': self.reps = max(self.reps, ev['n'])
        elif t == 'fault': self.faults[ev['label']] = self.faults.get(ev['label'], 0) + 1
        elif t == 'cue' and ev.get('intent') in ('correction', 'positive', 'milestone'):
            self.cues.append((ev['intent'], ev['text']))
            if ev['intent'] == 'milestone': self.why = ev.get('why')

    def next_item(self):
        if self.item is None or not self.plan: return None
        k = self.item + 1
        return self.plan[k] if k < len(self.plan) else None

    def text(self, state):
        """The context for the opening into `state` (rest, done) or after a gesture/probe that opened the floor; closes the set on
        rest/done. None at `wait`: run B on 10-09 showed a "Starting lunges ..." line answered right before the set and heard
        under the counts (the floor stays open at wait for a caller's question, but nothing invites a line)."""
        ex = self.exercise or 'the exercise'; tgt = f' of {self.target}' if self.target else ''
        if state == 'done' and not self.open and not self.sets:            # a boxing lesson (no sets): the worker's note is the whole context
            return (self.done_note + ' ' if self.done_note else '') + 'Never read these notes aloud; speak to the caller in your own words.' if self.done_note else None
        if state == 'done' and not self.open:                             # the last set was already summed up at its rest: no empty second summary
            return ('The whole plan is done: ' + '; '.join(f'{e} {r} reps' for e, r in self.sets) + '. Close the session warmly in one or two sentences. '
                    + (self.done_note + ' ' if self.done_note else '') + 'Never read these notes, the plan or any list aloud; speak to the caller in your own words.')
        if state in ('rest', 'done'):
            corr = sorted(self.faults.items(), key=lambda kv: -kv[1])
            spoken = [txt for intent, txt in self.cues if intent == 'correction']
            parts = [f'Set finished: {ex}, {self.reps}{tgt} reps counted' + (f' ({self.why})' if self.why and self.why != 'target' else '') + '.']
            parts.append('Corrections spoken: ' + ', '.join(dict.fromkeys(spoken)) + '.' if spoken else 'No corrections were needed.')
            if corr: parts.append('Faults seen: ' + ', '.join(f'{lab} x{n}' for lab, n in corr[:3]) + '.')
            nxt = self.next_item()
            if state == 'done':
                parts.append('The whole plan is done: ' + '; '.join(f'{e} {r} reps' for e, r in self.sets + [(ex, self.reps)]) + '. Close the session warmly in one or two sentences.'
                             + (' ' + self.done_note if self.done_note else ''))
            elif nxt: parts.append(f'Next up: {nxt[0]}' + (f' x {nxt[1]}' if nxt[1] else '') + f' after a rest of up to {self.rest_s:.0f} seconds; it starts '
                                   'by itself as soon as the caller starts moving, or when they say they are ready. You may say one or two short sentences now.')
            else: parts.append(f'The rest lasts {self.rest_s:.0f} seconds. You may say one or two short sentences now.')
            parts.append('Never read these notes, the plan or any list aloud; speak to the caller in your own words.')   # 2026-10-10: he opened every rest with "Plan: squats:10,pushups:8,..."
            self.sets.append((ex, self.reps)); self.reset()
            return ' '.join(parts)
        if state == 'wait': return None                                    # no invitation right before a set (see the docstring)
        if state == 'gesture': return 'The caller held a T-pose (the "next" gesture). You have a few seconds; one short line at most.'
        if state == 'probe': return 'The caller raised both arms over the head. One short line at most.'
        return None


try:
    from livekit.agents.voice import io as _io
    _AudioOutputBase = _io.AudioOutput; _VideoInputBase = _io.VideoInput
except Exception:                      # tests of the floor and the mixer do not need livekit-agents
    _io = None; _AudioOutputBase = object; _VideoInputBase = object


def rgba_frame(frame):
    """An RGB24 rtc.VideoFrame (the worker's VideoStream format, what Perception wants) as an RGBA one: the plugin's JPEG encoder
    calls frame.convert(RGBA) and the rtc FFI has no rgb24 -> rgba conversion ("rgb24 to Rgba is not supported", found by the
    live check on 10-10), so the alpha plane is added here in numpy (under 10 ms for 1280x720, once per forwarded frame). Any
    other frame type, and anything that is not an rtc.VideoFrame (tests), passes through unchanged."""
    try:
        from livekit import rtc
    except Exception:
        return frame
    if not isinstance(frame, rtc.VideoFrame) or frame.type != rtc.VideoBufferType.RGB24: return frame
    w, h = frame.width, frame.height
    rgb = np.frombuffer(frame.data, np.uint8).reshape(h, w, 3)
    rgba = np.empty((h, w, 4), np.uint8); rgba[..., :3] = rgb; rgba[..., 3] = 255
    return rtc.VideoFrame(w, h, rtc.VideoBufferType.RGBA, rgba.tobytes())


class VideoTee(_VideoInputBase):
    """The worker's video reaching the model (--vision): a livekit.agents VideoInput set as AgentSession.input.video before
    start(), so the framework's own forward task pulls frames from here and hands each one to the realtime session's
    push_video (livekit-plugins-google 1.8.5 JPEG-encodes it, quality 75, scaled to fit 1024x1024: a 1280x720 frame goes out
    as 1024x576; no resize is needed here; an RGB24 frame is handed over as RGBA, see rgba_frame). The session is built with
    video_sampler=None, so the only sampling is this tee's:
    offer(frame) keeps at most one frame per 1/fps s (fps at most MAX_FPS) and only while enabled (the model session has
    started; the worker binds before it starts, a warm job never offers anything), the newest frame replaces an unread one.
    Counters: n_offered (frames seen), n_queued (kept by the sampler), n_sent (pulled by the framework), n_replaced,
    n_early (offered before the session existed, never sent). aclose() ends the framework's forward loop."""
    MAX_FPS = 2.0
    _END = object()

    def __init__(self, fps=1.0, label='WorkerVideoTee', now=time.monotonic):
        if _io is not None: super().__init__(label=label)
        self.fps = max(0.1, min(float(fps or 1.0), self.MAX_FPS)); self.interval = 1.0 / self.fps; self.now = now
        self.enabled = False; self.closed = False; self.t_last = None; self.size = None
        self.n_offered = 0; self.n_queued = 0; self.n_sent = 0; self.n_replaced = 0; self.n_early = 0
        self._q = None; self._loop = None

    def _queue(self):
        if self._q is None: self._q = asyncio.Queue(maxsize=1)
        return self._q

    def offer(self, frame, t=None):
        """The worker's read loop (event loop thread): True when the frame was kept for the model."""
        self.n_offered += 1
        if self.closed: return False
        if not self.enabled: self.n_early += 1; return False
        t = self.now() if t is None else t
        if self.t_last is not None and t - self.t_last < self.interval - 1e-3: return False
        self.t_last = t; q = self._queue()
        frame = rgba_frame(frame)                                             # the plugin's encoder cannot convert RGB24 (the worker's stream format)
        if q.full():
            try: q.get_nowait(); self.n_replaced += 1
            except asyncio.QueueEmpty: pass
        q.put_nowait(frame); self.n_queued += 1
        if self.size is None: self.size = (getattr(frame, 'width', None), getattr(frame, 'height', None))
        return True

    def enable(self, on=True):
        self.enabled = bool(on)

    async def __anext__(self):
        frame = await self._queue().get()
        if frame is self._END: raise StopAsyncIteration
        self.n_sent += 1; return frame

    def __aiter__(self):
        return self

    async def pump(self, model):
        """What AgentSession._forward_video_task does with no sampler: every frame to model.push_video (tests, no session)."""
        async for frame in self: model.push_video(frame)

    async def aclose(self):
        if self.closed: return
        self.closed = True; self.enabled = False; q = self._queue()
        if q.full():
            try: q.get_nowait()
            except asyncio.QueueEmpty: pass
        q.put_nowait(self._END)

    def on_attached(self): pass
    def on_detached(self): pass


class MixerAudioOutput(_AudioOutputBase):
    """Rule (b): the AgentSession's audio sink. Frames (the framework resamples to 24 kHz for us) go into the cue player's
    MixInput; flush() ends a segment (one reply) and reports playback_finished once the mixer has read it (plus the source lead);
    clear_buffer() drops what is unread and reports the segment interrupted. No track, no second AudioSource.
    The floor's hard cut: with floor=<Floor> (anything with .open and .state) every frame that arrives while the floor is closed
    is dropped (never written to the mixer) and the reply is interrupted at its first dropped frame; cut() (the Floor calls it on
    close) clears the buffer and the MixInput queue at once. A segment that was cut or dropped reports interrupted=True, so the
    framework's chat context keeps only what was heard. One "[voice] cut: N frames dropped (floor closed)" line per reply."""
    class _Segment:
        """One reply: read0 (mixer samples read when its first frame landed; None until the previous reply has played out),
        pushed samples, and the flags of the cut."""
        __slots__ = ('read0', 'pushed', 'interrupted', 'dropped', 'cleared', 'cut')
        def __init__(self): self.read0 = None; self.pushed = 0; self.interrupted = False; self.dropped = 0; self.cleared = 0; self.cut = False

    def __init__(self, mix, lead_s=0.06, floor=None):
        if _io is None: raise RuntimeError('livekit-agents is required for MixerAudioOutput')
        super().__init__(label='CueMixer', capabilities=_io.AudioOutputCapabilities(pause=False), next_in_chain=None, sample_rate=SAMPLE_RATE)
        self.mix = mix; self.lead_s = lead_s; self.floor = floor
        self.seg = None                 # the reply being captured (its first frame arrived, flush/clear_buffer not yet)
        self.playing = None             # the reply whose playout _wait_playout is waiting for
        self._flush_task = None         # finalizers run one after the other: every segment the framework counted gets one playback_finished
        self.n_segments = 0; self.n_dropped = 0; self.n_cut = 0; self.t_first_frame = None

    @property
    def floor_open(self):
        return self.floor is None or bool(self.floor.open)

    async def capture_frame(self, frame):
        await super().capture_frame(frame)                                   # the framework (and its sink proxy) count the segment from this first frame
        seg = self.seg
        if seg is None:                                                      # the first frame of a reply
            seg = self.seg = self._Segment(); self.t_first_frame = time.monotonic()
            state = getattr(self.floor, 'state', None) if self.floor is not None else None
            t_ask = getattr(self.floor, 't_ask', None) if self.floor is not None else None
            since = '' if t_ask is None else f', {1000 * (self.floor.now() - t_ask):.0f} ms after the ask'
            print(f'[voice] speaking (floor {"open" if self.floor_open else "closed"}, state {state}{since})', flush=True)
        if seg.read0 is None:
            if self._flush_task is not None and not self._flush_task.done():
                await asyncio.shield(self._flush_task)                      # the previous reply plays out first; a cancel here must not kill its finalizer
                if self.seg is not seg: return                               # finalized while waiting (an interrupt): nothing to write
            seg.read0 = self.mix.n_read
        if not self.floor_open:                                              # the hard cut: nothing reaches the mixer while a set runs
            seg.dropped += 1; self.n_dropped += 1; seg.interrupted = True
            if not seg.cut:
                seg.cut = True; self.n_cut += 1
                if self.floor is not None and seg.pushed == 0: self.floor.model.interrupt()   # a reply that started closed: stop it, do not just mute it
            return
        pcm = np.frombuffer(frame.data, np.int16)
        if frame.num_channels > 1: pcm = pcm.reshape(-1, frame.num_channels).mean(axis=1).astype(np.int16)
        if frame.sample_rate != SAMPLE_RATE:                                  # should not happen (sample_rate declared); keep the stream going
            idx = (np.arange(int(len(pcm) * SAMPLE_RATE / frame.sample_rate)) * frame.sample_rate / SAMPLE_RATE).astype(int)
            pcm = pcm[np.clip(idx, 0, len(pcm) - 1)]
        self.mix.write(pcm); seg.pushed += len(pcm)

    def flush(self):
        super().flush()
        if self.seg is None: return
        seg, self.seg = self.seg, None
        self._finalize(seg)

    def clear_buffer(self):
        """The framework's interrupt (it flushes first, then clears on a cancel): drop what is unread; the reply being captured and
        the one playing out report interrupted. The base's segment flag is reset here too, so a caller that clears without a flush
        still gets the next reply counted as a new segment."""
        _AudioOutputBase.flush(self)
        if self.seg is None and self.playing is None: return
        self._clear_mix(close=True)
        if self.seg is not None:
            seg, self.seg = self.seg, None
            self._finalize(seg)

    def cut(self):
        """The Floor closed: clear the mixer at once and mark the reply in flight cut; the segment stays open until the framework's
        clear_buffer (which follows its interrupt) or the reply's flush finalizes it, so no second segment is opened by the tail."""
        self._clear_mix(close=False)

    def _clear_mix(self, close):
        cleared = self.mix.pending; self.mix.clear()
        owner = self.playing if self.playing is not None else self.seg       # the unread audio belongs to the reply playing out, else to the one being captured
        for s in (self.playing, self.seg):
            if s is None: continue
            s.interrupted = True
            if s is owner: s.cleared += cleared
            if cleared and not s.cut and s is owner: s.cut = True; self.n_cut += 1

    def _finalize(self, seg):
        self._flush_task = asyncio.create_task(self._wait_playout(seg, self._flush_task))

    async def _wait_playout(self, seg, prev):
        if prev is not None and not prev.done():
            try: await asyncio.shield(prev)
            except Exception: pass
        self.playing = seg
        while self.mix.pending > 0 and not seg.interrupted: await asyncio.sleep(0.01)
        if not seg.interrupted: await asyncio.sleep(self.lead_s)
        played = min(max(self.mix.n_read - (seg.read0 if seg.read0 is not None else self.mix.n_read), 0), seg.pushed)
        if seg.dropped or seg.cleared:
            print(f'[voice] cut: {seg.dropped} frames dropped (floor closed)' + (f', {1000 * seg.cleared / SAMPLE_RATE:.0f} ms of buffered audio cleared' if seg.cleared else '')
                  + f'; {played / SAMPLE_RATE:.1f} s of the reply was heard', flush=True)
        self.playing = None; self.n_segments += 1
        self.on_playback_finished(playback_position=played / SAMPLE_RATE, interrupted=seg.interrupted)


class GeminiModel:
    """The livekit-agents AgentSession with Gemini Live, output through MixerAudioOutput, input from the caller's room audio.
    start() is called once a caller is bound (a warm job never starts it: no model session, no cost)."""
    def __init__(self, persona, model=None, voice=None, api_key=None, on_turn=None, temperature=None, language=None, tools=None):
        self.tools = list(tools or []); self.persona = persona; self.model_id = model or os.environ.get('GEMINI_LIVE_MODEL') or DEFAULT_MODEL
        self.voice = voice or os.environ.get('GEMINI_VOICE') or DEFAULT_VOICE
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        lang = language if language is not None else os.environ.get('GEMINI_LANGUAGE', DEFAULT_LANGUAGE)
        self.language = None if str(lang).strip().lower() in ('', 'off', 'none') else str(lang).strip()
        self.on_turn = on_turn; self.temperature = temperature
        self.last_caller_t = -1e9; self.last_coach_t = -1e9        # monotonic times of the last transcripts (the conversation hold reads them)
        self.rt = None; self.session = None; self.agent = None; self.sink = None; self.video = None; self.started = False; self.n_generate = 0; self.n_interrupt = 0

    def build(self):
        """The plugin object (google.beta.realtime.RealtimeModel); separate from start() so a test can drive it without a room.
        language (BCP-47, default en-US) goes to the Live API's speech config: livekit-plugins-google 1.8.5 accepts it."""
        if self.rt is None:
            from livekit.plugins import google
            kw = dict(model=self.model_id, voice=self.voice)
            if self.api_key: kw['api_key'] = self.api_key
            if self.temperature is not None: kw['temperature'] = self.temperature
            if self.language: kw['language'] = self.language
            base_url = (os.environ.get('GEMINI_BASE_URL') or '').strip().rstrip('/')
            if base_url:                                                   # hackathon: GEMINI_API_KEY is the team token, the relay (modal/gemini_relay.py) holds the real key
                from google.genai import types                             # the SDK builds the Live websocket URL from base_url (wss) and sends the key as x-goog-api-key
                kw['http_options'] = types.HttpOptions(base_url=base_url)
                print(f'[voice] gemini via relay {base_url}', flush=True)
            self.rt = google.beta.realtime.RealtimeModel(**kw)
        return self.rt

    async def start(self, sink, room=None, participant_identity=None, video=None):
        """video: a VideoTee (or any livekit.agents VideoInput) set as session.input.video before start(): the framework then
        forwards every frame it yields to the realtime session's push_video (JPEG); video_sampler=None leaves the rate to the tee.
        RoomIO's own video input stays off (it selects tracks by source, camera or screenshare; the bridge's track may be neither)."""
        from livekit.agents import AgentSession, Agent
        from livekit.agents.voice.room_io import RoomOptions
        self.build(); self.sink = sink; self.video = video
        self.session = AgentSession(llm=self.rt, user_away_timeout=None, **(dict(video_sampler=None) if video is not None else {}))
        self.session.output.audio = sink                                  # before start: RoomIO then publishes no audio track
        if video is not None: self.session.input.video = video            # before start: the forward task starts with the session
        self.agent = Agent(instructions=self.persona, allow_interruptions=True, **(dict(tools=self.tools) if self.tools else {}))
        self.session.on('conversation_item_added', self._on_item)
        self.session.on('user_input_transcribed', self._on_user_transcript)
        self.session.on('error', lambda ev: log.warning('voice session error: %s', getattr(ev, 'error', ev)))
        kw = dict(record=False)
        if room is not None:
            kw.update(room=room, session_host=False,
                      room_options=RoomOptions(audio_output=False, text_output=False, text_input=False, video_input=False, audio_input=True,
                                               participant_identity=participant_identity, close_on_disconnect=False))
        await self.session.start(self.agent, **kw)
        self.session.input.set_audio_enabled(False)                        # closed until the floor opens
        self.started = True

    def _on_item(self, ev):
        item = getattr(ev, 'item', None)
        if getattr(item, 'role', None) == 'assistant': self.last_coach_t = time.monotonic()
        if getattr(item, 'role', None) == 'assistant' and self.on_turn:
            text = (getattr(item, 'text_content', None) or '').strip()
            if text: self.on_turn('coach', text)

    def _on_user_transcript(self, ev):
        if (ev.transcript or '').strip(): self.last_caller_t = time.monotonic()                      # interim transcripts count: the caller is talking
        if getattr(ev, 'is_final', False) and self.on_turn and (ev.transcript or '').strip(): self.on_turn('caller', ev.transcript.strip())

    # ---- the Floor's interface ----
    def set_input_enabled(self, on):
        if self.session is None: return
        try: self.session.input.set_audio_enabled(bool(on))
        except Exception as e: log.warning('set_audio_enabled(%s): %s', on, e)

    def generate(self, prompt):
        if self.session is None: return None
        self.n_generate += 1
        # A user-role note, not instructions=: livekit-plugins-google sends instructions as a MODEL turn, so Gemini took the bracketed
        # prompt for its own words and continued in that style (2026-10-10 calls: "(No instructions in response. Only say the
        # response.)" and whole paragraphs of reasoning were spoken). The set summaries already arrive as user-role context.
        try: return self.session.generate_reply(user_input=COACH_NOTE + prompt)
        except RuntimeError as e: log.warning('generate_reply: %s', e); return None

    def interrupt(self):
        if self.session is None: return
        self.n_interrupt += 1
        try: self.session.interrupt(force=True)
        except RuntimeError: pass

    async def add_context(self, text):
        if self.agent is None or not text: return
        try:
            ctx = self.agent.chat_ctx.copy(); ctx.add_message(role='user', content=COACH_NOTE + text)
            await self.agent.update_chat_ctx(ctx)
        except Exception as e: log.warning('add_context: %s', e)

    @property
    def speaking(self):
        return self.session is not None and self.session.current_speech is not None and not self.session.current_speech.done()

    async def aclose(self):
        if self.video is not None and hasattr(self.video, 'aclose'):
            try: await self.video.aclose()
            except Exception as e: log.debug('video tee close: %s', e)
        if self.session is not None:
            try: await self.session.aclose()
            except Exception as e: log.warning('session close: %s', e)
        self.session = None


class VoiceLayer:
    """One per job: consumes job.events, runs the Floor, feeds SetSummary context at openings, dispatches to the handlers, and
    logs model turns. job must have .events (asyncio.Queue), .player (CuePlayer or None), .emit_voice(role, text). model=None
    runs the handlers alone (ask_voice then only queues and warns)."""
    def __init__(self, job, model=None, plan=None, handlers=None, window_s=WINDOW_S, probe=True, now=time.monotonic, floor_open=False, vision=None):
        """probe=False (the worker without --latency-probe): a probe event opens no floor window (it is still logged).
        floor_open=True (--floor open): the Floor is always open (no set ever cuts the voice; asks fire at once).
        vision: a VideoTee (--vision) handed to the model session at start and enabled once the session is up."""
        self.job = job; self.model = model; self.handlers = handlers; self.vision = vision
        self.floor = Floor(model, window_s=window_s, probe=probe, now=now, always_open=floor_open) if model is not None else None
        self.summary = SetSummary(plan); self.mix = None; self.sink = None; self.started = False; self.n_events = 0; self.n_context = 0
        self.pending_context = []; self._warned_no_model = False
        self._hold_opened = None; self._hold_t0 = 0.0; self._last_voice_audio_t = -1e9

    def attach(self):
        """Give the cue player the mixer input and build the sink (no model session yet); the Floor cuts through the sink."""
        player = getattr(self.job, 'player', None)
        if player is None or self.model is None: return None
        gain_db = float(os.environ.get('VOICE_GAIN_DB', '0') or 0)
        self.mix = player.attach_mixer(gain_db=gain_db); self.sink = MixerAudioOutput(self.mix, floor=self.floor); self.floor.sink = self.sink; self.floor.player = player
        return self.mix

    async def on_caller(self, identity, room=None):
        """The worker bound a caller: start the model session once (the warm job never gets here)."""
        if self.model is None or self.started: return
        self.started = True
        if self.sink is None: self.attach()
        t = time.monotonic()
        try:
            await self.model.start(self.sink, room=room, participant_identity=identity, video=self.vision)
            print(f'[voice] model session started for {identity!r}: {self.model.model_id} voice {self.model.voice} in {time.monotonic() - t:.1f} s; floor {"open" if self.floor.open else "closed"}'
                  + (f'; video input {self.vision.fps:g} fps (tee)' if self.vision is not None else ''), flush=True)
            if self.floor.open: self.model.set_input_enabled(True)
            if self.vision is not None: self.vision.enable(True)                       # frames offered before this point were never sent
            if self.pending_context: await self.model.add_context(self.pending_context[-1])   # the latest line only, never a backlog
            self.pending_context = []
            if self.floor.open and self.floor.queue and not self.floor._audio_busy(): self.floor._fire_queue()   # an ask made before the session was up
            return True
        except Exception as e:
            log.exception('voice model start failed'); print(f'[voice] model start failed: {type(e).__name__}: {e}', flush=True)
            return False

    # ---- the handlers' verbs ----
    def say_now(self, text, intent='other'):
        """A bank string through the fast path (CuePlayer.say, priority 'other': never over a count). Returns the record."""
        player = getattr(self.job, 'player', None)
        if player is None: log.warning('say_now(%r): no cue player (events-only mode)', text); return None
        return player.say(text, intent, dict(source='handler'))

    def ask_voice(self, prompt):
        """Ask the model (about 1 s to first audio) now if the floor is open, else at the next opening (dropped if that is more than
        ASK_TTL_S away or a set starts first). Returns 'spoken'|'queued'|None (None without a model: --voice off, said once)."""
        if self.floor is None:
            if not self._warned_no_model:
                self._warned_no_model = True; print(f'[voice] ask_voice ignored: no model (--voice off); first prompt {prompt[:60]!r}', flush=True)
            return None
        if self.model is not None and not self.model.started:                 # the intake greeting can come before the session is up
            self.floor.queue.append((self.floor.now(), prompt)); return 'queued'
        return self.floor.ask(prompt)

    def set_plan(self, plan, rest_s=None):
        """Intake: hand the chosen plan to the session (it starts on a later frame); the set summary's "Next up" follows it.
        Returns None when accepted, else why not."""
        per = getattr(self.job, 'per', None)
        if per is None: return 'no session (perception off)'
        why = per.session.set_plan(plan, rest_s)
        if why is None: self.summary.plan = parse_plan(plan)
        print(f'[voice] intake: plan {plan!r}, rest {rest_s} s -> ' + ('accepted' if why is None else f'refused: {why}'), flush=True)
        return why

    # ---- events ----
    def on_event(self, ev):
        """The Floor first (so the handler's ask_voice at a rest is spoken, not queued), the context at rest/done (and after a
        gesture or probe that opened the floor), then the handlers. The rest/done summary runs whether or not a model is attached
        (it closes the set in the bookkeeping) and is sent as context only when there is one."""
        self.n_events += 1
        self.summary.on_event(ev)
        change = self.floor.on_event(ev) if self.floor is not None else None
        t = ev['type']
        if t == 'session' and ev.get('state') in ('rest', 'done'):
            per = getattr(self.job, 'per', None)
            if per is not None: self.summary.rest_s = per.session.rest_s
            text = self.summary.text(ev['state'])
            if text and self.floor is not None: self._context(text)
        elif change == 'open' and t in ('gesture', 'probe'):
            text = self.summary.text(t)
            if text: self._context(text)
        if self.handlers is not None:
            try: self.handlers(ev, self.say_now, self.ask_voice)
            except Exception as e: log.warning('handler failed on %s: %r', ev.get('type'), e)

    def _context(self, text):
        self.n_context += 1
        if self.model is not None and self.model.started: asyncio.ensure_future(self.model.add_context(text))
        else: self.pending_context = [text]                                  # before the model session exists: keep the latest line only
        print(f'[voice] context: {text[:120]}', flush=True)

    HOLD_QUIET_S = 4.0      # the hold lasts this long after the last word from either side (or the last voice audio)
    HOLD_MAX_S = 25.0       # and never more than this per opening: a set must still start

    def update_hold(self):
        """Conversation hold (2026-10-10 live call: the next set started while the caller was talking to him): while the floor is open
        and either side spoke or the model's audio played within HOLD_QUIET_S, perception.hold pauses the session's rest/wait clock
        (session.py), capped at HOLD_MAX_S per opening. Logged once per hold."""
        per = getattr(self.job, 'per', None)
        if per is None or self.floor is None or self.model is None: return
        now = time.monotonic(); want = False
        if self.floor.open and self.floor.state in ('intake', 'wait', 'rest'):
            if self._hold_opened is None: self._hold_opened = now
            last = max(getattr(self.model, 'last_caller_t', -1e9), getattr(self.model, 'last_coach_t', -1e9), self._last_voice_audio_t, self.floor.t_ask or -1e9)
            capped = (now - self._hold_opened) >= self.HOLD_MAX_S and self.floor.state != 'intake'   # the intake is a conversation: no cap
            audible = (self.mix is not None and self.mix.pending > 0) or bool(getattr(self.model, 'started', False) and getattr(self.model, 'speaking', False))
            want = audible or ((now - last) < self.HOLD_QUIET_S and not capped)   # never start a set over Brodie's own sentence
        else:
            self._hold_opened = None
        if self.mix is not None and self.mix.pending > 0: self._last_voice_audio_t = now
        if want != bool(per.hold):
            per.hold = want
            if want: self._hold_t0 = now
            else: print(f'[voice] hold: session clock held {now - self._hold_t0:.1f} s for the conversation (state {self.floor.state})', flush=True)

    async def run(self):
        """The consumer task: events from job.events, a tick every 0.25 s for the gesture window and the conversation hold."""
        while True:
            try: ev = await asyncio.wait_for(self.job.events.get(), 0.25)
            except asyncio.TimeoutError:
                if self.floor is not None: self.floor.tick()
                self.update_hold(); continue
            self.on_event(ev); self.update_hold()

    async def aclose(self):
        if self.vision is not None:                                          # the layer owns the tee: end the forward loop whatever the model does
            try: await self.vision.aclose()
            except Exception as e: log.debug('vision tee close: %s', e)
        if self.model is not None: await self.model.aclose()
