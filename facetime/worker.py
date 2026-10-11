"""LiveKit Agents worker for the hackathon starter: the coach's eyes (facetime/perception.py) and, by default, its fast mouth
(facetime/audio_out.py CuePlayer, unchanged) in ONE participant with ONE audio track, dispatched by name the way the Chert CLI
dispatches agents (reports/STARTER_PREREQS_REVIEW.md section 1: `chert dev --agent NAME` creates a room with a RoomAgentDispatch for
NAME; the call is accepted once a participant of kind AGENT is in the room; the bridge joins as chert-bridge-<call_id>).

Registers with WorkerOptions(agent_name=AGENT_NAME, default "brodie-coach"). Per job: connect (subscribe all), publish the cue audio
track at once unless --events-only / EVENTS_ONLY=1 (the bridge binds to the one participant with an unmuted audio track), then bind
to the first non-agent participant that publishes video, whoever it is (chert-bridge-<call_id>, a phone page, a fake bridge), feed the
newest frame to Perception in a worker thread with the same arrival-aligned 10 fps sampling as coach_agent, and keep the job joined
until the room closes. A job with no caller (`chert doctor --agent`, prewarm: metadata {"chert_warm": true}) does nothing but stay
joined until somebody publishes video; nothing is said or started on job start. Models and the cue bank are loaded in the prewarm
hook, so a job start costs no model load (one idle preloaded process is kept, --idle-processes).
Job end (Mac test 2026-10-09, reports/MAC_TEST_2026-10-08.md section 5): livekit-agents runs shutdown callbacks only AFTER the entrypoint
returns, so run() ends itself: on the room's `disconnected` event, when the bound participant leaves and nobody rebinds within
--leave-grace-s (default 2 s: a rejoin binds again), when connect() fails (a doctor or warm room torn down before the peer connection
completes), on --max-s and on a perception error. It then cleans up with bounded waits (feed thread join 1 s off the loop, VideoStream
closed, player stopped, cue in flight at most 1.5 s and only while still connected) and calls ctx.shutdown() so the framework finalises.

Events (one JSON object each, schema in facetime/perception.py and docs/STARTER_EVENTS.md) go to
  - an in-process asyncio.Queue, job.events, for a conversational layer (register a coroutine with @consumer, below);
  - a JSONL log, --event-log / RTCOACH_EVENT_LOG (default data/tmp/worker_events.jsonl; each line also carries room, call_id, job);
  - optionally the room data channel as JSON (--publish-data, topic --data-topic, off by default: Chert rooms hold two participants,
    the data channel is for rooms we own).
Cues the player finished or dropped go to --cue-log / RTCOACH_CUE_LOG (the coach_agent --log record shape) and to stdout.

Plan: --plan, else env PLAN, else starter/plan.txt (the hackathon script; ships with the demo plan), else the demo plan (DEFAULT_PLAN);
--exercise for one exercise, --auto for auto-detect. Job metadata is read for chert_call and media and logged (never acted on).

Voice (--voice gemini, default off: the default and events-only modes are untouched): facetime/voice.py runs Gemini Live (voice Puck)
in this same participant, its audio mixed into the cue player's one track through CuePlayer.attach_mixer() (ducked about 10 dB under a
count) and gated to the gaps between sets (session state wait/rest/done, a gesture or an armed probe outside a set, a caller's question
during a rest; the caller's audio is not forwarded during a set and a reply still streaming when a set starts is cut: interrupted,
buffers cleared, later frames dropped). The model session starts when a caller is bound (never on a warm job). Persona from
--persona (starter/persona.md); handlers from --handlers, default starter/handlers.py whenever it exists, in EVERY mode (on_event(event,
say_now, ask_voice); without the voice ask_voice returns None); model turns are logged as events of type voice {text, t, role}.
Latency probe: --latency-probe / RTCOACH_LATENCY_PROBE=1 (off by default; probe events are emitted either way; never acted on during
an active set).

Beyond reps (docs/STARTER_README.md "Your own coach: beyond reps"): --perception off runs NO Perception (no pose model in prewarm,
no session script, no rep/cue/fault events): the worker still joins, publishes its audio track, binds to the first remote participant
that publishes video (or, with no video, audio), starts the model session on the bind, emits one synthetic session event
{state: "open", exercise: null, count: 0, mode: "voice"} once the session is up (handlers greet on it) and a status event (no pose
fields) every 5 s, and ends on the same paths (leave grace, disconnect, --max-s, connect failure). The Floor is then ALWAYS OPEN
(--floor open, the default without perception): the caller's audio is always forwarded, ask_voice fires at once, nothing is cut;
say_now still plays bank strings. --vision (with --voice gemini) tees the caller's video track into the model session at about
1 fps (--vision-fps, at most 2): facetime/voice.py VideoTee, a livekit.agents VideoInput set as AgentSession.input.video, pulled by
the framework into the realtime session's push_video (JPEG). A warm job never starts a session or forwards a frame. The plan file
is then appended to the persona as free text (a recipe, a checklist), not parsed as exercises.
  .venv/bin/python -m facetime.worker --perception off --vision --voice gemini --persona starter/examples/meal_prep/persona.md \
      --plan-file starter/examples/meal_prep/recipe.md --handlers starter/examples/meal_prep/handlers.py dev
Run:
  .venv/bin/python -m facetime.worker --help
  .venv/bin/python -m facetime.worker [--plan "squats:10,pushups:8"] [--events-only] [--publish-data] dev     # LIVEKIT_URL/API_KEY/API_SECRET in env
  AGENT_NAME=brodie-coach EVENTS_ONLY=1 .venv/bin/python -m facetime.worker start
  GEMINI_API_KEY=... .venv/bin/python -m facetime.worker --voice gemini dev                                      # + GEMINI_LIVE_MODEL, GEMINI_VOICE
The flags before the LiveKit subcommand are ours; `dev`, `start`, `connect`, `download-files` and their options are the LiveKit Agents CLI's.
"""
import argparse, asyncio, collections, json, logging, os, sys, threading, time
import numpy as np
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, JobProcess, WorkerOptions, cli
from facetime import DATA, AUDIO_TRACK_NAME
from facetime.pipeline import FPS
from facetime.perception import Perception, Config, add_arguments, build_pipeline, EVENT_SCHEMA_VERSION
from facetime.coach_agent import frame_to_rgb              # the same frame conversion as the demo coach

log = logging.getLogger('worker')
ARGS_ENV = 'RTCOACH_WORKER_ARGS'      # main() hands the parsed flags to the job processes as JSON (forkserver children re-import this module)
DEFAULT_PLAN = 'squats:10,pushups:8,jumping jacks:20,lunges:10'
STARTER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'starter')
AGENT_KIND = rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
BRIDGE_PREFIX = 'chert-bridge-'
LK_COMMANDS = ('dev', 'start', 'connect', 'download-files', 'console')
CONSUMERS = []                        # coroutines consumer(job) started per job; they read job.events (see docs/STARTER_EVENTS.md)


def consumer(fn):
    """Register `async def fn(job)`: started as a task once the job is connected; read events with `await job.events.get()`."""
    CONSUMERS.append(fn); return fn


def truthy(s):
    return str(s or '').strip().lower() in ('1', 'true', 'yes', 'on')


def build_parser():
    ap = argparse.ArgumentParser(prog='python -m facetime.worker', add_help=False, description=__doc__.split('\n')[0],
                                 epilog='After these flags comes the LiveKit Agents CLI subcommand: dev (registers under AGENT_NAME with debug logging; restart it '
                                        'after a code change, livekit-agents 1.8.5 has no hot reload), start (production), connect --room R (join one room directly), '
                                        'download-files. The arms-up latency probe is off unless --latency-probe (env RTCOACH_LATENCY_PROBE=1) and never acts during '
                                        'an active set. Env: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, AGENT_NAME, PLAN, EVENTS_ONLY, RTCOACH_EVENT_LOG, '
                                        'RTCOACH_CUE_LOG, RTCOACH_LATENCY_PROBE, GEMINI_API_KEY, GEMINI_LIVE_MODEL, GEMINI_VOICE, GEMINI_LANGUAGE.')
    add_arguments(ap)
    for act in ap._actions:                                                 # the perception flag, with the worker's default (env) and what the worker does with it
        if act.dest == 'latency_probe':
            act.default = truthy(os.environ.get('RTCOACH_LATENCY_PROBE'))
            act.help = ('act on the arms-up probe (env RTCOACH_LATENCY_PROBE=1): beep through the cue player and open the voice floor for a few seconds, '
                        'never during an active set; off by default (a jumping jack is an arm raise). Probe events are emitted and logged either way')
    ap.add_argument('--auto', action='store_true', help='auto-detect mode (no plan, no exercise); without it and without --plan/--exercise/PLAN the demo plan runs')
    ap.add_argument('--events-only', action='store_true', default=truthy(os.environ.get('EVENTS_ONLY')), help='publish no audio at all: events only (env EVENTS_ONLY=1)')
    ap.add_argument('--publish-data', action='store_true', help='also publish every event as JSON on the room data channel (off by default)')
    ap.add_argument('--data-topic', default='brodie.events')
    ap.add_argument('--event-log', default=os.environ.get('RTCOACH_EVENT_LOG', os.path.join(DATA, 'tmp', 'worker_events.jsonl')), help='JSONL event log (appended; env RTCOACH_EVENT_LOG)')
    ap.add_argument('--cue-log', default=os.environ.get('RTCOACH_CUE_LOG', os.path.join(DATA, 'tmp', 'worker_cues.jsonl')), help='JSONL log of cues played or dropped (env RTCOACH_CUE_LOG)')
    ap.add_argument('--agent-name', default=os.environ.get('AGENT_NAME', 'brodie-coach'), help='dispatch name (env AGENT_NAME); "" = automatic dispatch to every room (local tests)')
    ap.add_argument('--audio-queue-ms', type=int, default=100, help='AudioSource queue for this worker (100; the Mac saw an underrun at 60 during one-per-second jack counts; coach_agent keeps 60)')
    ap.add_argument('--leave-grace-s', type=float, default=2.0, help='after the bound participant leaves, end the job unless somebody rebinds within this (negative = never end on a leave)')
    ap.add_argument('--no-probe-beep', dest='probe_beep', action='store_false', help='with --latency-probe: open the voice floor on the probe but do not beep')
    ap.add_argument('--video-quality', default='low', choices=['low', 'medium', 'high', 'none'], help='simulcast layer to request when the publisher simulcasts')
    ap.add_argument('--max-s', type=float, default=None, help='leave the room this long after the first video frame (default: stay until the room closes)')
    ap.add_argument('--idle-processes', type=int, default=int(os.environ.get('RTCOACH_IDLE_PROCS', '1')), help='preloaded job processes kept ready (models loaded in prewarm)')
    ap.add_argument('--voice', default=os.environ.get('VOICE', 'off'), choices=['off', 'gemini'], help='conversational layer between sets: gemini = Gemini Live through the cue track (facetime/voice.py; env VOICE)')
    ap.add_argument('--voice-model', default=os.environ.get('GEMINI_LIVE_MODEL') or None, help='Gemini Live model id (env GEMINI_LIVE_MODEL; default facetime.voice.DEFAULT_MODEL)')
    ap.add_argument('--voice-name', default=os.environ.get('GEMINI_VOICE', 'Puck'), help='Gemini Live voice (env GEMINI_VOICE; Puck is the cue bank voice)')
    ap.add_argument('--persona', default=os.environ.get('PERSONA_FILE', os.path.join(STARTER_DIR, 'persona.md')), help='the voice persona prompt file (starter/persona.md)')
    ap.add_argument('--handlers', default=os.environ.get('HANDLERS_FILE') or None, help='a module with on_event(event, say_now, ask_voice); default starter/handlers.py whenever that file exists, in every mode '
                                                                                        '(without --voice gemini, ask_voice returns None); --handlers none loads nothing')
    ap.add_argument('--log-caller', action='store_true', help='also emit and log the caller\'s transcribed turns as voice events (role caller); off by default: the caller\'s words never leave the model session (docs/STARTER_README.md section 8)')
    ap.add_argument('--plan-file', default=os.environ.get('PLAN_FILE', os.path.join(STARTER_DIR, 'plan.txt')), help='the plan when neither --plan nor PLAN is given (starter/plan.txt); '
                                                                                                                      'with --perception off the file is appended to the persona as free text (a recipe)')
    ap.add_argument('--perception', default=os.environ.get('PERCEPTION', 'on'), choices=['on', 'off'], help='off = no pose model, no rep counter, no session script (env PERCEPTION): the worker '
                                                                                                           'joins, binds to the caller, starts the voice and emits one synthetic session {state: open} event; the floor is then always open')
    ap.add_argument('--floor', default=os.environ.get('FLOOR') or None, choices=['gated', 'open'], help='gated = the voice speaks between sets only (the default with perception on); '
                                                                                                        'open = always allowed to speak, nothing is ever cut (the default with --perception off; env FLOOR)')
    ap.add_argument('--vision', action='store_true', default=truthy(os.environ.get('VISION')), help='with --voice gemini: tee the caller\'s video into the model session at --vision-fps (env VISION=1)')
    ap.add_argument('--vision-fps', type=float, default=float(os.environ.get('VISION_FPS', '1') or 1), help='frames per second handed to the model with --vision (default 1, at most 2; JPEG per frame)')
    ap.add_argument('--mirror', action='store_true', default=truthy(os.environ.get('MIRROR')), help='publish a video track "coach-video": the caller\'s own frame with the pose, '
                    'the count and the last cue drawn on it and Brodie\'s face inset (facetime/mirror.py, as coach_agent --mirror); before the first frame, and with '
                    '--perception off, Brodie\'s face alone. The FaceTime caller sees it (env MIRROR=1)')
    ap.add_argument('--mirror-fps', type=float, default=float(os.environ.get('MIRROR_FPS', '10') or 10), help='frame rate of the --mirror track (default 10)')
    ap.add_argument('-h', '--help', action='store_true', help='show this help and exit')
    return ap


def perception_off(a):
    return str(getattr(a, 'perception', 'on') or 'on').strip().lower() == 'off'


def read_plan_text(path):
    """--perception off: the plan file as free text (a recipe, a checklist), stripped; empty or missing -> None."""
    if not path: return None
    try:
        with open(path) as f: return f.read().strip() or None
    except OSError:
        return None


def resolve_mode(a):
    """--plan, else PLAN, else starter/plan.txt, else the demo plan; --exercise and --auto take precedence over the default plan.
    Handlers: --handlers PATH, else starter/handlers.py whenever it exists (every mode, so a team's on_event runs with say_now even
    without the voice), `none` for nothing."""
    if perception_off(a):                       # no session script: the plan is free text for the persona (a recipe), kept across the two parses
        a.plan_text = getattr(a, 'plan_text', None) or a.plan or os.environ.get('PLAN') or read_plan_text(getattr(a, 'plan_file', None))
        a.plan = None; a.exercise = None; a.floor = getattr(a, 'floor', None) or 'open'
    else:
        a.plan_text = None; a.floor = getattr(a, 'floor', None) or 'gated'
        if a.exercise: a.plan = None
        elif getattr(a, 'auto', False): a.plan = None
        else: a.plan = a.plan or os.environ.get('PLAN') or read_plan_file(getattr(a, 'plan_file', None)) or DEFAULT_PLAN
    if getattr(a, 'intake', False) and (perception_off(a) or a.voice != 'gemini' or not a.plan):
        if ARGS_ENV not in os.environ: print('[worker] --intake needs plan mode with --voice gemini (the voice asks and picks the plan): intake OFF', flush=True)
        a.intake = False
    a.vision_fps = max(0.1, min(float(getattr(a, 'vision_fps', 1.0) or 1.0), 2.0))
    h = getattr(a, 'handlers', None)
    if h and str(h).strip().lower() in ('none', 'off'): a.handlers = 'none'          # kept as the string: the job process resolves the flags again
    elif not h and os.path.exists(os.path.join(STARTER_DIR, 'handlers.py')): a.handlers = os.path.join(STARTER_DIR, 'handlers.py')
    return a


SESSION_TOOLS_PROMPT = (
    "\n\nYou cannot see the caller's video: a pose tracker watches them for you. If the caller asks whether you can see them or how "
    "to stand, call the check_view tool and answer from it; never guess. When the caller says they are ready during a rest (for example "
    "\"let's go\", \"I'm ready\", \"next\"), call the start_next_set tool. Call tools only through function calling: never write JSON, "
    "code or a tool's name in what you say."
)


def load_starter(name):
    """A team-editable module from starter/ (programs.py, boxing.py)."""
    import importlib.util
    path = os.path.join(STARTER_DIR, name)
    spec = importlib.util.spec_from_file_location('starter_' + name[:-3], path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def load_programs(path=os.path.join(STARTER_DIR, 'programs.py')):
    """starter/programs.py: the intake's workout templates (GOALS, LEVELS, INTAKE_PROMPT, build)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('starter_programs', path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def handlers_path(a):
    """The handlers module to load, None for --handlers none / nothing."""
    h = getattr(a, 'handlers', None)
    return None if not h or str(h).strip().lower() in ('none', 'off') else h


def read_plan_file(path):
    """starter/plan.txt: one line like "squats:10,pushups:8" (comments with #); empty or missing -> None."""
    if not path: return None
    try:
        lines = [l.strip() for l in open(path) if l.strip() and not l.lstrip().startswith('#')]
        return ','.join(lines) if lines else None
    except OSError:
        return None


def load_args():
    """The flags main() parsed, from the environment (a job process is a fresh interpreter); defaults when run without main()."""
    a = build_parser().parse_args([])
    try: d = json.loads(os.environ.get(ARGS_ENV) or '{}')
    except ValueError: d = {}
    for k, v in d.items(): setattr(a, k, v)
    return resolve_mode(a)


def parse_metadata(md):
    """Job metadata is JSON from the dispatcher (Chert: chert_warm, chert_call, media); anything else is logged and ignored."""
    if not md: return {}
    try:
        d = json.loads(md); return d if isinstance(d, dict) else {'metadata': d}
    except ValueError:
        return {'metadata_raw': md[:200]}


def prewarm(proc: JobProcess):
    """Runs in each job process before any job: models, trigger, thresholds and (unless events-only) the cue bank."""
    a = load_args(); cfg = Config.from_args(a); t = time.time()
    proc.userdata['args'] = a; proc.userdata['cfg'] = cfg
    if perception_off(a):                                      # no pose model, no classifier: nothing to load but the bank
        proc.userdata['pipe'] = None
    else:
        proc.userdata['pipe'] = build_pipeline(cfg)
    if not a.events_only:
        from facetime.audio_out import CueBank
        proc.userdata['bank'] = CueBank()
    if perception_off(a):
        print(f'[worker] prewarm: loaded in {time.time() - t:.1f} s (perception OFF: no pose model, no session script; voice {a.voice}, vision {"on" if a.vision else "off"}, '
              f'floor {a.floor}, audio {"OFF (events only)" if a.events_only else "cue bank"})', flush=True)
    else:
        print(f'[worker] prewarm: loaded in {time.time() - t:.1f} s (pose {proc.userdata["pipe"].pose_variant}, counter {cfg.counter}, trigger {proc.userdata["pipe"].trigger_path or "off"}, '
              f'mode {cfg.mode} ({a.plan or a.exercise or "auto-detect"}), audio {"OFF (events only)" if a.events_only else "cue bank"})', flush=True)


def describe_mode(a):
    """The mode words for the console lines: plan/single/auto with perception, 'voice' without."""
    if perception_off(a): return 'voice (perception off' + (', vision on' if getattr(a, 'vision', False) else '') + f', floor {a.floor})'
    return f'{"plan" if a.plan else ("single" if a.exercise else "auto")} ({a.plan or a.exercise or "auto-detect"})'


async def entrypoint(ctx: JobContext):
    ud = ctx.proc.userdata
    a = ud.get('args') or load_args(); cfg = ud.get('cfg') or Config.from_args(a)
    pipe = ud.pop('pipe', None)
    if pipe is None and not perception_off(a): pipe = build_pipeline(cfg)   # single use: a Perception owns it until the job ends
    bank = ud.get('bank')
    if bank is None and not a.events_only:
        from facetime.audio_out import CueBank
        bank = CueBank()
    job = CoachJob(ctx, a, cfg, pipe, bank, parse_metadata(ctx.job.metadata))
    await job.run()


class CoachJob:
    """One dispatched job: the room, the optional cue player, the Perception and its feed thread, the event sinks."""
    def __init__(self, ctx, a, cfg, pipe, bank, md):
        self.ctx = ctx; self.a = a; self.cfg = cfg; self.pipe = pipe; self.bank = bank; self.md = md
        self.room = ctx.room; self.loop = None
        self.warm = bool(md.get('chert_warm'))
        self.events = asyncio.Queue()                 # every event, for an in-process conversational layer
        self.player = None; self.player_task = None; self.cue_log = None
        self.per = None; self.thread = None; self.stop = threading.Event(); self.done = asyncio.Event(); self.closed = False
        self.latest = None; self.seq = 0; self.processed = 0; self.dropped = 0; self.idle = 0; self.frame_event = threading.Event()
        self.video_task = None; self.bound = None; self.call_id = None; self.frame_size = None; self.first_frame_t = None
        self.worker_error = None; self.cues_played = 0; self.ts_mode = None
        self.voice = None; self.voice_seq = 0; self.event_writer = None; self.consumer_tasks = []
        self.end_reason = None; self.leave_task = None; self.ctx_shutdown_called = False; self.t_end = None
        self.session_state = None; self.probes_ignored = 0        # the probe (--latency-probe) never beeps or opens the floor during an active set
        self.voice_mode = perception_off(a)                       # --perception off: no Perception; the worker emits its own session/status events
        self.t0 = time.monotonic(); self.ev_seq = 0; self.vision = None; self.vision_logged = False; self.session_opened = False; self.open_task = None
        self.mirror_src = None; self.last_cue = (None, 0.0); self.mirror_task = None
        self.deferred_cues = []; self.defer_task = None             # bank lines waiting for the voice to finish (outside a set)
        from facetime.recap import CallRecap, Streaks, profile_key
        self.recap = CallRecap(); self.streaks = Streaks(); self.profile = profile_key(md); self.day = None   # the streak and the results card
        self.card_img = None; self.card_path = None                  # at "done": the card is saved and shown as the coach's video (--mirror)
        self.lesson = None; self.lesson_task = None; self.boxing = None # a boxing lesson (starter/boxing.py): {style, title, status}   # --mirror: the newest processed frame and the last cue played

    # ---- lifecycle ----
    async def run(self):
        self.loop = asyncio.get_running_loop(); ctx = self.ctx; a = self.a
        ctx.add_shutdown_callback(self.aclose)
        if not a.events_only:
            from facetime.audio_out import CuePlayer
            self.player = CuePlayer(bank=self.bank, on_record=self.log_cue, queue_ms=a.audio_queue_ms)
            os.makedirs(os.path.dirname(os.path.abspath(a.cue_log)) or '.', exist_ok=True); self.cue_log = open(a.cue_log, 'a')
        if getattr(a, 'voice', 'off') == 'gemini' or handlers_path(a):
            self.voice = self.build_voice()
        self.room.on('track_subscribed', self.on_track_subscribed)
        self.room.on('participant_connected', lambda p: print(f'[worker] participant joined: {p.identity} (kind {rtc.ParticipantKind.Name(p.kind)})', flush=True))
        self.room.on('participant_disconnected', self.on_participant_disconnected)
        self.room.on('disconnected', self.on_room_disconnected)
        try:
            await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
        except BaseException as e:
            if isinstance(e, asyncio.CancelledError): raise
            print(f'[worker] job {ctx.job.id}: connect failed ({type(e).__name__}: {str(e)[:120]}): the room went away before the peer connection completed '
                  f'(a doctor or warm room torn down early); leaving', flush=True)
            self.end_reason = 'connect failed'; await self.aclose(); self.shutdown_ctx(self.end_reason); return
        md_note = ', '.join(f'{k}={json.dumps(self.md[k])[:80]}' for k in ('chert_warm', 'chert_call', 'media') if k in self.md) or 'none'
        print(f'[worker] job {ctx.job.id}: joined room {self.room.name!r} as {self.room.local_participant.identity!r} (agent {a.agent_name!r}); '
              f'mode {describe_mode(a)}; audio {"OFF (events only)" if a.events_only else "cue player"}; '
              f'metadata: {md_note}; participants {list(self.room.remote_participants)}', flush=True)
        if self.player is not None:
            await self.room.local_participant.publish_track(self.player.track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
            self.player_task = asyncio.create_task(self.player.run())
            print(f'[worker] published audio track {AUDIO_TRACK_NAME}', flush=True)
        if getattr(a, 'mirror', False):
            self.mirror_task = asyncio.create_task(self.mirror_loop(), name='mirror')
        if self.voice_mode:                                       # no Perception: the worker's own session/status events, logged the same way
            from facetime.perception import JsonlWriter
            self.event_writer = JsonlWriter(a.event_log, extra=dict(room=self.room.name, call_id=self.call_id, job=ctx.job.id))
        else:
            self.per = Perception(self.cfg, pipe=self.pipe, say=(self.say_from_worker if self.player is not None else None), clip_s=(self.bank.duration if self.bank is not None else None))
            self.per.on_event(self.on_event)
            self.event_writer = self.per.log_to(a.event_log, extra=dict(room=self.room.name, call_id=self.call_id, job=ctx.job.id))
            if self.player is not None:
                self.per.audio_stats = lambda: dict(underruns=self.player.underruns, cues_played=self.cues_played, cues_dropped=self.player.n_dropped)
        print(f'[worker] events (schema v{EVENT_SCHEMA_VERSION}) -> {a.event_log}' + (f' and data topic {a.data_topic!r}' if a.publish_data else ''), flush=True)
        if self.warm:
            print('[worker] warm job (chert_warm): no caller; staying joined, nothing is said until a participant publishes video' + (' or audio' if self.voice_mode else ''), flush=True)
        if self.voice_mode:
            self.consumer_tasks = [asyncio.create_task(self._status_loop(), name='status-loop')]
        else:
            self.thread = threading.Thread(target=self.feed, daemon=True, name='perception-feed'); self.thread.start()
            self.consumer_tasks = []
        self.consumer_tasks += [asyncio.create_task(fn(self), name=f'consumer-{fn.__name__}') for fn in CONSUMERS]
        if self.voice is not None:
            self.consumer_tasks.append(asyncio.create_task(self.voice.run(), name='consumer-voice'))
        await self.done.wait()
        reason = self.end_reason or (f'perception error: {self.worker_error!r}' if self.worker_error is not None else 'done')
        await self.aclose(); self.shutdown_ctx(reason)

    def end(self, reason):
        """End the job (event loop): run() wakes, cleans up and returns, then tells the framework (shutdown_ctx)."""
        if self.end_reason is None: self.end_reason = reason
        if self.t_end is None: self.t_end = time.monotonic()
        self.done.set()

    def shutdown_ctx(self, reason):
        if self.ctx_shutdown_called: return
        self.ctx_shutdown_called = True
        try: self.ctx.shutdown(reason=reason)
        except Exception as e: log.debug('ctx.shutdown: %r', e)

    def on_room_disconnected(self, *args):
        try: why = rtc.DisconnectReason.Name(args[0]) if args else 'unknown'
        except Exception: why = str(args[0]) if args else 'unknown'
        print(f'[worker] room disconnected ({why}): leaving', flush=True); self.end(f'room disconnected ({why})')

    async def _leave_grace(self):
        await asyncio.sleep(max(0.0, self.a.leave_grace_s))
        self.leave_task = None
        if self.bound is None and not self.closed:
            print(f'[worker] caller left and nobody rebound within {self.a.leave_grace_s:g} s: leaving', flush=True); self.end('caller left')

    def room_connected(self):
        try: return self.room.connection_state == rtc.ConnectionState.CONN_CONNECTED
        except Exception: return True

    async def aclose(self, *_):
        """Idempotent cleanup (run() at its end, and the framework's shutdown callback): every wait is bounded, nothing blocks the loop."""
        if self.closed: return
        self.closed = True; self.stop.set(); self.done.set(); t0 = time.monotonic()
        if self.lesson_task is not None: self.lesson_task.cancel()
        if (self.recap.sets or self.recap.lesson) and not self.warm and self.card_path is None: self.make_card()       # a call that ended before "All done"
        if self.t_end is None: self.t_end = t0
        if self.leave_task is not None: self.leave_task.cancel()
        if self.open_task is not None and not self.open_task.done(): self.open_task.cancel()
        for t in getattr(self, 'consumer_tasks', []): t.cancel()
        if self.mirror_task is not None: self.mirror_task.cancel()
        if self.defer_task is not None: self.defer_task.cancel()
        if self.video_task is not None and not self.video_task.done():                 # closes the VideoStream (read_video's finally)
            self.video_task.cancel()
            try: await asyncio.wait_for(asyncio.gather(self.video_task, return_exceptions=True), 1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError): pass
        if self.thread is not None and self.thread.is_alive():                            # the feed loop polls stop every 50 ms
            await asyncio.get_running_loop().run_in_executor(None, self.thread.join, 1.0)
        if self.voice is not None:
            try: await asyncio.wait_for(self.voice.aclose(), 3)
            except Exception as e: log.warning('voice close: %r', e)
        if self.player is not None:
            if self.room_connected():
                for _ in range(15):                                                       # let the cue in flight finish (max 1.5 s)
                    if not self.player.playing and self.player.pending is None: break
                    await asyncio.sleep(0.1)
            self.player.stop()
            if self.player_task is not None:
                try: await asyncio.wait_for(asyncio.gather(self.player_task, return_exceptions=True), 1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError): self.player_task.cancel()
            src = getattr(self.player, 'source', None)                                    # drop the AudioSource's FFI handle here, not in a __del__
            if hasattr(src, 'aclose'):
                try: await asyncio.wait_for(src.aclose(), 1.0)
                except Exception as e: log.debug('audio source close: %r', e)
        if self.per is not None: s = self.per.summary(); self.per.close()
        else:
            s = {}
            if self.voice_mode:
                s = dict(mode='voice', events=self.ev_seq, session_opened=self.session_opened)
                if self.vision is not None: s.update(vision_offered=self.vision.n_offered, vision_sent=self.vision.n_sent, vision_early=self.vision.n_early)
                if self.event_writer is not None:
                    try: self.event_writer.close()
                    except Exception as e: log.debug('event writer close: %r', e)
            if self.pipe is not None:                                                   # connect failed before Perception took the pipe: close the
                try: self.pipe.close()                                                  # pose landmarker here (an unclosed one hangs in __del__)
                except Exception as e: log.debug('pipe close: %r', e)
        if self.cue_log: self.cue_log.close()
        print(f'[worker] job {self.ctx.job.id} done ({self.end_reason or "shutdown"}; cleanup {1000 * (time.monotonic() - t0):.0f} ms, '
              f'{1000 * (time.monotonic() - self.t_end):.0f} ms after the end trigger): frames received {self.seq}, processed {self.processed}, dropped {self.dropped}, idle ticks {self.idle}; {s}', flush=True)

    # ---- video ----
    def on_track_subscribed(self, track, pub, participant):
        own = participant.kind == AGENT_KIND or participant.identity == self.room.local_participant.identity
        if track.kind != rtc.TrackKind.KIND_VIDEO:
            # --perception off: a caller with audio and no video (an audio-only call) is still a caller; the model session starts on it
            if track.kind == rtc.TrackKind.KIND_AUDIO and self.voice_mode and self.bound is None and not own and not self.closed:
                self.bind(participant, f'audio {pub.name!r} (no video yet)')
            return
        if own:
            print(f'[worker] video from agent participant {participant.identity!r} ignored', flush=True); return
        if self.video_task is not None and not self.video_task.done():
            if self.bound is not None and self.bound != participant.identity:
                print(f'[worker] video track {pub.name!r} from {participant.identity!r} ignored: already bound to {self.bound!r}', flush=True); return
            # the bound caller left (or republished) and its old stream has not ended yet: a fast rejoin, a duplicate identity
            # kicking the first, a camera switch. Drop the old stream and bind the new track, or the leave grace ends the job.
            print(f'[worker] video track {pub.name!r} from {participant.identity!r} replaces the previous stream (rejoin)', flush=True)
            self.video_task.cancel()
        self.bind(participant, f'video {pub.name!r} {pub.width}x{pub.height}, simulcast {getattr(pub, "simulcasted", False)}')
        if self.a.video_quality != 'none' and getattr(pub, 'simulcasted', False):
            try: pub.set_video_quality(rtc.VideoQuality.Value('VIDEO_QUALITY_' + self.a.video_quality.upper()))
            except Exception as e: print(f'[worker] set_video_quality failed: {e}', flush=True)
        self.video_task = asyncio.create_task(self.read_video(track))

    def bind(self, participant, what):
        """A caller: remember it (the leave grace is keyed on it), start the model session once (never on a warm job: this only runs
        when somebody publishes), and in voice mode emit the synthetic session {state: open} event once the session is up."""
        self.bound = participant.identity
        if participant.identity.startswith(BRIDGE_PREFIX): self.call_id = participant.identity[len(BRIDGE_PREFIX):]
        extra = getattr(self.event_writer, 'extra', None)                   # the JSONL lines carry the call id from the bind on (null before it)
        if isinstance(extra, dict): extra['call_id'] = self.call_id
        print(f'[worker] bound to {participant.identity!r} (call id {self.call_id or "n/a"}): {what}', flush=True)
        if self.voice_mode:
            if not self.session_opened and self.open_task is None:
                self.open_task = asyncio.create_task(self._open_session(participant.identity), name='voice-start')
        elif self.voice is not None and self.voice.model is not None and not self.voice.started:
            asyncio.create_task(self.voice.on_caller(participant.identity, room=self.room), name='voice-start')

    async def _open_session(self, identity):
        """Voice mode: the model session (when there is one), then ONE session event {state: open, mode: voice} for the handlers."""
        try:
            if self.voice is not None and self.voice.model is not None:
                await self.voice.on_caller(identity, room=self.room)
            if self.closed or self.session_opened: return
            self.session_opened = True
            self.emit('session', state='open', exercise=None, target=None, count=0, item=None, done=False, mode='voice')
        finally:
            self.open_task = None

    def video_frames(self, track):
        """The VideoStream of the bound track (RGB24 for Perception; the tee hands the same frames to the model). Tests replace it."""
        return rtc.VideoStream(track, format=rtc.VideoBufferType.RGB24)

    async def read_video(self, track):
        stream = self.video_frames(track)
        try:
            async for ev in stream:
                t_mono = time.monotonic()
                self.seq += 1; self.latest = (ev.frame, ev.timestamp_us, t_mono, time.time(), self.seq); self.frame_event.set()
                if self.vision is not None and self.vision.offer(ev.frame, t_mono) and not self.vision_logged:
                    self.vision_logged = True
                    print(f'[vision] forwarding video to the model at {self.vision.fps:g} fps ({ev.frame.width}x{ev.frame.height})', flush=True)
                if self.voice_mode:                                                      # no feed thread: first frame, size and --max-s here
                    if self.first_frame_t is None:
                        self.first_frame_t = t_mono; self.frame_size = (ev.frame.width, ev.frame.height)
                        print(f'[worker] first frame {ev.frame.width}x{ev.frame.height}', flush=True)
                    if self.a.max_s and t_mono - self.first_frame_t >= self.a.max_s:
                        print(f'[worker] --max-s {self.a.max_s} reached', flush=True); self.end('max_s reached'); break
        finally:
            await stream.aclose()
        print('[worker] video stream ended (a republished track is picked up again)', flush=True)
        if self.video_task is asyncio.current_task(): self.video_task = None     # a rejoin may already have started the next one

    # ---- voice mode (--perception off): the worker's own events ----
    def emit(self, type_, **fields):
        """A synthetic event in the perception schema (v1, t on this job's clock, own seq), through the same sinks as Perception's."""
        ev = dict(v=EVENT_SCHEMA_VERSION, t=round(time.monotonic() - self.t0, 3), t_wall=round(time.time(), 3), seq=self.ev_seq, type=type_)
        if type_ != 'status': ev['frame'] = self.seq
        ev.update(fields); self.ev_seq += 1
        if self.event_writer is not None:
            try: self.event_writer(ev)
            except Exception as e: log.warning('event log failed: %r', e)
        self.on_event(ev)
        return ev

    async def _status_loop(self):
        """Every status_every_s (5 s) while a caller is bound: frames received, the vision tee's counters, the player's stats."""
        every = float(getattr(self.cfg, 'status_every_s', 5.0) or 5.0); last_seq = self.seq; last_t = time.monotonic()
        while True:
            await asyncio.sleep(every)
            now = time.monotonic(); el = max(now - last_t, 1e-6); recv = (self.seq - last_seq) / el; last_seq = self.seq; last_t = now
            if self.bound is None and self.video_task is None: continue
            ev = dict(fps_recv=round(recv, 1), state='open' if self.session_opened else 'init', exercise=None, count=0, mode='voice',
                      voice=('on' if self.voice is not None and self.voice.model is not None and self.voice.started else 'off'))
            if self.vision is not None: ev.update(vision_frames=self.vision.n_sent, vision_fps=self.vision.fps, vision_replaced=self.vision.n_replaced)
            if self.player is not None: ev.update(underruns=self.player.underruns, cues_played=self.cues_played, cues_dropped=self.player.n_dropped)
            self.emit('status', **ev)

    def on_participant_disconnected(self, p):
        print(f'[worker] participant left: {p.identity}', flush=True)
        if p.identity == self.bound:
            self.bound = None
            if self.a.leave_grace_s is not None and self.a.leave_grace_s >= 0 and self.leave_task is None and not self.closed:
                self.leave_task = asyncio.create_task(self._leave_grace(), name='leave-grace')

    def capture_time(self, ts_us, t_wall):
        if self.ts_mode is None:
            self.ts_mode = 'epoch' if abs(ts_us / 1e6 - t_wall) < 5 else 'recv'
        return ts_us / 1e6 if self.ts_mode == 'epoch' else t_wall

    # ---- worker thread: arrival-aligned, >= 90 ms apart, newest frame only (coach_agent._worker) ----
    def feed(self):
        try:
            self._feed()
        except BaseException as e:
            log.exception('perception thread died: %r', e)
            print(f'[worker] perception thread died: {type(e).__name__}: {e}', flush=True)
            self.worker_error = e; self.loop.call_soon_threadsafe(self.done.set)

    def _feed(self):
        last_seq = 0; last_proc = -1e9
        while not self.stop.is_set():
            if not self.frame_event.wait(timeout=0.05): slot = None
            else:
                self.frame_event.clear(); slot = self.latest
                if slot is not None and slot[4] != last_seq:
                    since = time.monotonic() - last_proc
                    if since < 0.09:                              # too soon after the last one: wait for the 90 ms mark, then take the newest frame
                        time.sleep(0.09 - since); slot = self.latest; self.frame_event.clear()
            if slot is None or slot[4] == last_seq:
                self.idle += 1; continue
            frame, ts_us, t_mono, t_wall, seq = slot
            self.dropped += seq - last_seq - 1; last_seq = seq; last_proc = time.monotonic()
            if self.first_frame_t is None:
                self.first_frame_t = t_mono; print(f'[worker] first frame {frame.width}x{frame.height}', flush=True)
            if self.frame_size != (frame.width, frame.height):
                if self.frame_size is not None: print(f'[worker] received resolution changed {self.frame_size} -> {(frame.width, frame.height)}', flush=True)
                self.frame_size = (frame.width, frame.height)
            rgb = frame_to_rgb(frame)
            self.per.frames_received = self.seq; self.per.frames_dropped = self.dropped
            self.per.process_frame(rgb, t_recv=t_mono, meta=dict(frame_seq=seq, ts_us=ts_us, t_capture=self.capture_time(ts_us, t_wall), t_recv=t_wall))
            self.processed += 1
            if self.mirror_task is not None: self.mirror_src = (rgb, self.pipe.last_landmarks)   # one tuple, swapped whole: the mirror reads it on the loop
            if self.a.max_s and t_mono - self.first_frame_t >= self.a.max_s:
                print(f'[worker] --max-s {self.a.max_s} reached', flush=True); self.loop.call_soon_threadsafe(self.end, 'max_s reached'); break

    # ---- sinks ----
    DEFER_INTENTS = ('visibility', 'positive', 'announce', 'other')
    DEFER_MAX_S = 8.0

    def say_from_worker(self, text, intent, meta):
        self.loop.call_soon_threadsafe(self._say_or_defer, text, intent, meta)

    def _voice_audible(self):
        """The conversational voice is talking (a reply in progress or its audio still in the mixer)."""
        mix = getattr(self.player, 'mix', None) if self.player is not None else None
        model = getattr(self.voice, 'model', None) if self.voice is not None else None
        return (mix is not None and mix.pending > 0) or bool(model is not None and getattr(model, 'started', False) and getattr(model, 'speaking', False))

    def _say_or_defer(self, text, intent, meta):
        """Outside a set, a bank line ("Step back ...", "Rest for a moment", "Let's do X") waits while the voice is talking instead of
        chopping its sentence (2026-10-10: the player paused the reply for each clip). Counts, corrections and milestones go at once."""
        if intent == 'visibility' and self.session_state == 'lesson':        # shadowboxing flickers in and out of frame: no "Step back" chatter
            print(f'[cue] skipped during the lesson: {text!r}', flush=True); return
        if intent in self.DEFER_INTENTS and self.session_state != 'active' and self._voice_audible():
            self.deferred_cues.append((time.monotonic(), text, intent, meta))
            print(f'[cue] waiting for the voice to finish: {text!r} ({intent})', flush=True)
            if self.defer_task is None: self.defer_task = asyncio.create_task(self._flush_deferred(), name='cue-defer')
            return
        self.player.say(text, intent, meta)

    async def _flush_deferred(self):
        """Play the waiting lines once the voice has been quiet for 0.3 s; a set start or DEFER_MAX_S drops the stale ones
        (an announce always plays: the set needs it)."""
        quiet_since = None
        try:
            while self.deferred_cues and not self.closed:
                await asyncio.sleep(0.05); now = time.monotonic()
                audible = self._voice_audible(); quiet_since = None if audible else (quiet_since or now)
                oldest = now - self.deferred_cues[0][0]
                if (quiet_since is not None and now - quiet_since >= 0.3) or self.session_state == 'active' or oldest >= self.DEFER_MAX_S:
                    cues, self.deferred_cues = self.deferred_cues, []
                    for t0, text, intent, meta in cues:
                        stale = self.session_state == 'active' or now - t0 >= self.DEFER_MAX_S
                        if intent == 'announce' or not stale: self.player.say(text, intent, meta)
                        else: print(f'[cue] dropped (stale after waiting for the voice): {text!r}', flush=True)
        finally:
            self.defer_task = None

    def on_event(self, ev):
        """Perception thread -> event loop: the queue, the console, the optional data channel. The arms-up probe beeps here (say_now
        from the loop, as coach_agent does) only with --latency-probe, the cue player attached and no set active (run B on 10-09: a
        jumping jack is an arm raise), and the event says so (beep) before it is logged; a probe during a set is emitted, not acted on."""
        if ev['type'] == 'session': self.session_state = ev.get('state')
        self.recap.on_event(ev)
        if ev['type'] == 'session' and ev.get('state') in ('rest', 'done') and (ev.get('count') or 0) > 0 and self.day is None and not self.warm:
            try:
                self.day = self.streaks.record(self.profile)                 # the first finished set of the call counts the day
                print(f"[recap] streak: day {self.day['current']} (best {self.day['best']}) for profile {self.profile}", flush=True)
                if self.voice is not None:
                    self.voice.summary.done_note = (f"This was day {self.day['current']} in a row for the caller" + (f" (their best is {self.day['best']})" if self.day['best'] > self.day['current'] else ' (a new best)') +
                                                    '; mention it in the close.' if self.day['current'] > 1 else 'This was their first day: invite them to come back tomorrow and start a streak.')
            except Exception as e: log.warning('streak: %r', e)
        if ev['type'] == 'session' and ev.get('state') == 'done' and (self.recap.sets or self.recap.lesson) and self.card_path is None and not self.warm:
            self.make_card()
        if ev['type'] == 'probe':
            armed = bool(getattr(self.a, 'latency_probe', False)); active = self.session_state == 'active'
            if active: self.probes_ignored += 1
            beep = armed and not active and self.player is not None and getattr(self.a, 'probe_beep', True)
            ev['beep'] = bool(beep)
            if beep:
                from facetime.audio_out import BEEP
                t_first = ev.get('t_first_up'); now_m = time.monotonic(); now_w = time.time()
                meta = dict(t_recv=(now_w - (now_m - t_first)) if t_first is not None else None, t_detect=now_w, on_box_ms=ev.get('on_box_ms'))
                self.loop.call_soon_threadsafe(self.player.say_now, BEEP, 'probe', meta)
        self.loop.call_soon_threadsafe(self._deliver, ev)

    def _deliver(self, ev):
        self.events.put_nowait(ev)
        t = ev['type']
        if t == 'rep':
            c = ev.get('confidence') or {}
            print(f"[event] rep {ev['exercise']} {ev['n']} (agree {c.get('agree')}, amp {c.get('amp')}) t={ev['t']:.2f}", flush=True)
        elif t in ('session', 'gesture', 'visibility', 'fault', 'probe'):
            note = ''
            if t == 'probe' and not getattr(self.a, 'latency_probe', False): note = '  (probe off: --latency-probe to beep and open the floor)'
            elif t in ('probe', 'gesture') and self.session_state == 'active': note = '  (during a set: no beep, no floor window)'
            print(f"[event] {t} " + ' '.join(f'{k}={v}' for k, v in ev.items() if k not in ('v', 't_wall', 'seq', 'type', 'frame')) + note, flush=True)
        elif t == 'status' and ev.get('mode') == 'voice':
            vis = f"vision {ev['vision_frames']} frames sent ({ev['vision_fps']:g} fps, {ev.get('vision_replaced', 0)} replaced)" if 'vision_frames' in ev else 'vision off'
            print(f"[status] recv {ev.get('fps_recv')} fps, {vis}, voice {ev.get('voice')}, underruns {ev.get('underruns')}, cues {ev.get('cues_played')}, "
                  f"session {ev['state']} (voice mode, no perception)", flush=True)
        elif t == 'status':
            print(f"[status] recv {ev.get('fps_recv')} fps, processed {ev['fps_processed']} fps, pose {ev['pose_ms']} ms ({ev['pose_model']}), clf {ev['clf_ms']} ms, "
                  f"recv->decision p50 {ev.get('recv_to_decision_p50_ms')} ms, dropped {ev.get('dropped')}, underruns {ev.get('underruns')}, session {ev['state']}/{ev['exercise'] or '-'} count {ev['count']}", flush=True)
        if self.a.publish_data:
            asyncio.create_task(self._publish(ev))

    async def _publish(self, ev):
        try:
            await self.room.local_participant.publish_data(json.dumps(ev).encode(), reliable=True, topic=self.a.data_topic)
        except Exception as e:
            log.warning('publish_data failed: %r', e)

    # ---- the conversational layer (facetime/voice.py) ----
    def build_voice(self):
        """--voice gemini: the VoiceLayer with a GeminiModel (session started on bind, never on a warm job) and the mixer input on the
        cue player; --handlers alone: the handlers with say_now only. Returns None (with a message) when the layer cannot run."""
        from facetime.voice import VoiceLayer, GeminiModel, VideoTee, load_handlers, read_text, DEFAULT_PERSONA
        a = self.a; model = None
        if a.voice == 'gemini':
            if self.player is None:
                print('[voice] --voice gemini needs the cue player track (not --events-only): the model speaks through that one AudioSource; voice OFF', flush=True)
            elif not (os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')):
                print('[voice] GEMINI_API_KEY is MISSING: voice OFF', flush=True)
            else:
                persona = read_text(a.persona, DEFAULT_PERSONA); tools = []
                try:
                    so_far = self.streaks.peek(self.profile)
                    if so_far['current'] > 1: persona += (f"\n\nThis caller has worked out {so_far['current']} day(s) in a row so far (their best is {so_far['best']}); "
                                                          "welcome them back and mention the streak once in your greeting.")
                except Exception as e: log.warning('streak peek: %r', e)
                if getattr(a, 'mirror', False) and not self.voice_mode:
                    persona += ("\n\nWhen the whole workout is done, the caller's results card appears on their screen: in your closing line, "
                                "tell them to take a screenshot and share it.")
                if not self.voice_mode:                                   # the session tools: "I'm ready" and "can you see me?"
                    persona += SESSION_TOOLS_PROMPT; tools += self.session_tools()
                if getattr(a, 'intake', False):
                    programs = load_programs(); persona += programs.INTAKE_PROMPT; tools.append(self.intake_tool(programs))
                    self.boxing = load_starter('boxing.py'); persona += self.boxing.BOXING_PROMPT; tools.append(self.boxing_tool(self.boxing))
                    plan_line = f'\n\nFor your own reference only, never to be read aloud: if the caller chooses nothing, the plan is {a.plan}.'
                elif a.plan: plan_line = f'\n\nFor your own reference only, never to be read aloud: the session plan, in order, is {a.plan}.'
                elif getattr(a, 'plan_text', None): plan_line = '\n\nThe plan you are coaching, step by step (follow it in order, one step at a time; never read it aloud):\n' + a.plan_text
                else: plan_line = ''
                model = GeminiModel(persona + plan_line, model=a.voice_model, voice=a.voice_name, on_turn=self.emit_voice, tools=tools)
        if getattr(a, 'vision', False):
            if model is None: print('[vision] --vision needs --voice gemini with a model session: no video is forwarded', flush=True)
            else: self.vision = VideoTee(fps=a.vision_fps)
        hpath = handlers_path(a); handlers = load_handlers(hpath) if hpath else None
        if model is None and handlers is None: return None
        floor_open = getattr(a, 'floor', None) == 'open'
        if floor_open and not self.voice_mode: print('[voice] --floor open with perception on: the voice is never cut by a set (it will talk over the counts)', flush=True)
        v = VoiceLayer(self, model=model, plan=a.plan, handlers=handlers, probe=bool(getattr(a, 'latency_probe', False)), floor_open=floor_open, vision=self.vision)
        if model is not None:
            v.attach()
            print(f'[voice] gemini {model.model_id} voice {model.voice} language {model.language or "model default"}; persona {a.persona} ({len(model.persona)} chars'
                  + (f', plan text {len(a.plan_text)} chars from {a.plan_file}' if getattr(a, 'plan_text', None) else '') + f'); '
                  f'handlers {hpath or "none"}; mixer on the cue track (duck {20 * __import__("math").log10(self.player.mix.duck):.0f} dB); '
                  f'floor {"ALWAYS OPEN" if floor_open else "gated to the gaps between sets"}; probe window {"on" if v.floor.probe else "off"}; '
                  f'vision {f"on ({self.vision.fps:g} fps tee)" if self.vision is not None else "off"}; session starts when a caller is bound', flush=True)
        else:
            print(f'[voice] handlers {hpath} (say_now only; no model: ask_voice returns None)', flush=True)
        return v

    def make_card(self):
        """The results card: saved under data/cards/ and, with --mirror, shown on the caller's screen from now on."""
        try:
            from facetime.recap import card_image, card_frame, CARDS_DIR
            import datetime as _dt
            img = card_image(self.recap, self.day); now = _dt.datetime.now()
            path = os.path.join(CARDS_DIR, f'brodie-{now:%Y%m%d-%H%M%S}.png'); os.makedirs(CARDS_DIR, exist_ok=True); img.save(path)
            self.card_path = path; self.card_img = card_frame(img)
            print(f'[recap] results card saved: {path}' + (' (shown on the call)' if self.mirror_task is not None else ''), flush=True)
        except Exception as e: log.warning('results card: %r', e)

    def _speaking(self):
        """Brodie is audible: a bank clip playing, or the voice's reply still in the mixer."""
        pl = self.player
        if pl is None: return False
        mix = getattr(pl, 'mix', None)
        return pl.playing is not None or (mix is not None and mix.pending > 0)

    def _mirror_frame(self, av, compose, W, H):
        """One RGB24 frame (runs in a thread): the newest caller frame composed with the session state, else the face alone."""
        src = self.mirror_src; s = self.per.session if self.per is not None else None
        face = av.frame(speaking=self._speaking())
        if self.card_img is not None: img = self.card_img                    # the workout is done: the results card, for a screenshot
        elif src is None or s is None: img = np.asarray(face.resize((W, H)))
        elif self.lesson is not None and s.state == 'lesson':
            rgb, lm = src                                                    # a boxing lesson: style and round clock in the top bar
            img = compose(rgb, lm, 0, None, f"{self.lesson['style']} boxing", self.lesson['status'], self.last_cue[0], self.last_cue[1], face)
        else:
            rgb, lm = src
            img = compose(rgb, lm, s.last_count, s.target, self.pipe.exercise, s.state, self.last_cue[0], self.last_cue[1], face)
        return rtc.VideoFrame(W, H, rtc.VideoBufferType.RGB24, np.ascontiguousarray(img).tobytes()).convert(rtc.VideoBufferType.I420)

    async def mirror_loop(self):
        """--mirror: the coach's video track (as coach_agent --mirror): published at join, like the audio, so a FaceTime video call has
        the agent's video from the start (Chert: the speaking participant publishes audio and video). Rendering runs off the loop.
        Any failure is logged and leaves the call alone."""
        try:
            from facetime.avatar import Avatar
            from facetime.mirror import compose, W, H
            av = Avatar(); fps = max(1.0, min(float(getattr(self.a, 'mirror_fps', 10.0) or 10.0), 15.0))
            source = rtc.VideoSource(W, H)
            track = rtc.LocalVideoTrack.create_video_track('coach-video', source)
            await self.room.local_participant.publish_track(track, rtc.TrackPublishOptions(
                source=rtc.TrackSource.SOURCE_CAMERA, simulcast=False, video_encoding=rtc.VideoEncoding(max_framerate=int(fps), max_bitrate=1_500_000)))
            print(f'[worker] published video track coach-video {W}x{H} at {fps:g} fps (mirror)', flush=True)
            period = 1.0 / fps; next_t = self.loop.time()
            while not self.stop.is_set():
                source.capture_frame(await asyncio.to_thread(self._mirror_frame, av, compose, W, H))
                next_t = max(next_t + period, self.loop.time()); await asyncio.sleep(max(0.0, next_t - self.loop.time()))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f'[worker] mirror stopped: {e!r}', flush=True)

    def session_tools(self):
        """start_next_set (the caller is ready: skip the rest) and check_view (what the pose tracker sees: the model sees no video)."""
        from livekit.agents import function_tool

        @function_tool(description='The caller says they are ready (for example "let\'s go", "I\'m ready", "next"): skip the rest and start the '
                                   'next exercise now (in a boxing lesson: end the current rest or round and move to the next round). Only during a rest, right before a '
                                   'set, or in a boxing lesson.')
        async def start_next_set() -> str:
            if self.per is None: return 'No workout is running.'
            if self.lesson is not None and self.lesson_task is not None and not self.lesson_task.done():   # a boxing lesson: end the rest or the round
                self.lesson['skip'] = True; print('[voice] start_next_set -> lesson: moving on', flush=True)
                return 'Moving on now: say one short encouraging line; the next part of the lesson starts as soon as you finish.'
            why = self.per.session.request_next()
            print(f'[voice] start_next_set -> ' + ('accepted' if why is None else f'refused: {why}'), flush=True)
            return 'Starting the next exercise as soon as you finish this sentence; say one short encouraging line.' if why is None else f'Not possible now ({why}).'

        @function_tool(description='What the camera currently shows of the caller, from the pose tracker. Call it whenever the caller asks '
                                   'whether you can see them or how they should stand. You cannot see the video yourself.')
        async def check_view() -> str:
            per = self.per
            if per is None or self.first_frame_t is None: return 'No video from the caller yet.'
            state = per._vis_emitted or 'none'
            return {'full': 'The whole body is in view: head to feet.',
                    'partial': 'Only part of the body is in view (usually the feet or the head are cut off): ask them to step back or move the phone further away.',
                    'none': 'Nobody is in view right now.'}.get(state, 'Unknown.') + (' Counting is paused until the whole body is back in view.' if per._paused else '')

        @function_tool(description='How the caller\'s guard and stance look right now, measured by the pose tracker (hand height, feet width). '
                                   'Call it when the caller asks how their form, guard or stance looks.')
        async def check_form() -> str:
            if self.per is None or self.pipe is None: return 'No video from the caller yet.'
            if self.boxing is None: self.boxing = load_starter('boxing.py')
            text, _ = self.boxing.form_report(self.pipe.last_landmarks)
            return text or 'The tracker cannot see the arms and shoulders clearly right now: ask them to step back into view.'

        return [start_next_set, check_view, check_form]

    def boxing_tool(self, boxing):
        """start_boxing_lesson: the intake's other branch. The session leaves the intake for 'lesson'; the timeline runs here."""
        from typing import Literal
        from livekit.agents import function_tool
        styles = tuple(boxing.STYLES)

        @function_tool(description=f'Start a shadowboxing lesson in one style ({", ".join(styles)}) instead of a workout. Call it during the intake '
                                   'as soon as the caller picks a boxing style.')
        async def start_boxing_lesson(style: Literal[styles]) -> str:
            if self.per is None: return 'No session is running.'
            if self.lesson is not None: return 'A lesson is already running.'
            why = self.per.session.start_lesson()
            print(f'[voice] boxing lesson: {style} -> ' + ('accepted' if why is None else f'refused: {why}'), flush=True)
            if why is not None: return f'The lesson cannot start now ({why}).'
            st = boxing.STYLES[style]; self.lesson = dict(style=style, title=st['title'], status='getting ready')
            self.recap.plan_label = f"{st['title']} boxing"
            self.lesson_task = asyncio.create_task(self.run_lesson(boxing, style), name='boxing-lesson')
            return (f"Lesson set: {st['title']}, {len(st['drills'])} rounds of {boxing.ROUND_S} seconds. Now, in three or four short sentences: "
                    f"the style's big idea ({st['idea']}), the stance ({st['stance']}) and the guard ({st['guard']}); ask them to clear some "
                    "space and get into the stance. Round one comes after that.")
        return start_boxing_lesson

    async def run_lesson(self, boxing, style):
        """The lesson's timeline (starter/boxing.py run_lesson) with the worker's verbs; at the end: the streak, the card, 'done'."""
        job = self
        class Coach:
            round_s = boxing.ROUND_S; rest_s = boxing.REST_S
            def ask(self, prompt):
                r = job.voice.ask_voice(prompt) if job.voice is not None else None
                print(f'[lesson] ask ({r}): {prompt[:70]!r}', flush=True); return r
            def say(self, text):
                if job._voice_audible() or (job.player is not None and job.player.playing is not None): return None   # never over Brodie's voice
                print(f'[lesson] cue: {text!r}', flush=True)
                return job.voice.say_now(text) if job.voice is not None else None
            def form_data(self): return boxing.form_report(job.pipe.last_landmarks if job.pipe is not None else None)[1]
            def status(self, text): job.lesson['status'] = text
            def skip(self):
                if job.lesson.get('skip'): job.lesson['skip'] = False; return True
                return False
            def form(self): return boxing.form_report(job.pipe.last_landmarks if job.pipe is not None else None)[0]
            def log(self, text): print(f'[lesson] {text}', flush=True)
            async def wait_quiet(self, max_s):
                t0 = time.monotonic(); quiet = None
                while time.monotonic() - t0 < max_s:
                    await asyncio.sleep(0.1); now = time.monotonic()
                    quiet = None if job._voice_audible() else (quiet or now)
                    if quiet is not None and now - quiet >= 0.8 and now - t0 > 1.5: return
        try:
            taught = await boxing.run_lesson(style, Coach())
            st = boxing.STYLES[style]; self.recap.lesson = dict(style=style, title=st['title'], drills=taught)
            try: self.day = self.streaks.record(self.profile)
            except Exception as e: log.warning('streak: %r', e)
            if self.voice is not None:
                streak = (f" This was day {self.day['current']} in a row for the caller; mention it." if self.day and self.day['current'] > 1 else
                          ' This was their first day: invite them back tomorrow to start a streak.')
                self.voice.summary.done_note = (f"The {st['title']} boxing lesson is over ({len(taught)} rounds: {', '.join(taught)}). In the close, name the "
                                                f"style's one key idea ({st['idea']}) and suggest practising round one again tomorrow." + streak)
            print(f'[lesson] {st["title"]} done: {len(taught)} rounds', flush=True)
            self.per.session.finish_lesson()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.warning('boxing lesson failed: %r', e); print(f'[lesson] stopped: {e!r}', flush=True)

    def intake_tool(self, programs):
        """--intake: the choose_workout function tool. The model passes the caller's goal and level; starter/programs.py turns them
        into a plan from its templates; the session starts it once nobody is talking. The tool's answer is what the model says next."""
        from typing import Literal
        from livekit.agents import function_tool
        goals = tuple(programs.GOALS); levels = tuple(programs.LEVELS)

        @function_tool(description='Set today\'s workout from the caller\'s goal and level. Call it once, during the intake, as soon as '
                                   f'you know both (goal one of {", ".join(goals)}; level one of {", ".join(levels)}).')
        async def choose_workout(goal: Literal[goals], level: Literal[levels]) -> str:
            plan, rest_s, said = programs.build(goal, level); self.recap.plan_label = f'{level} {goal}'
            why = self.voice.set_plan(plan, rest_s) if self.voice is not None else 'no voice layer'
            if why is not None: return f'The workout could not be changed ({why}); carry on with the current plan.'
            return f'Workout set: {said}. Tell the caller in one short sentence, then ask them to step back so their whole body is in view.'
        return choose_workout

    def emit_voice(self, role, text):
        """A model (or caller) turn as an event of type voice {text, t, role}: the queue, the JSONL log, the console (own seq counter)."""
        if role == 'caller' and not getattr(self.a, 'log_caller', False):
            print(f'[voice] caller spoke ({len(text.split())} words; not logged, --log-caller to keep transcripts)', flush=True); return
        t = round(time.monotonic() - (self.per.t0 if self.per is not None else self.t0), 3)
        ev = dict(v=EVENT_SCHEMA_VERSION, t=t, t_wall=round(time.time(), 3), seq=self.voice_seq, type='voice', text=text, role=role, producer='voice')
        self.voice_seq += 1
        print(f'[voice] {role}: {text}', flush=True)
        self.events.put_nowait(ev)
        if self.event_writer is not None:
            try: self.event_writer(ev)
            except Exception as e: log.warning('voice event log failed: %r', e)
        if self.a.publish_data: asyncio.create_task(self._publish(ev))

    def log_cue(self, rec):
        rec = dict(rec, room=self.room.name, call_id=self.call_id)
        if self.cue_log: self.cue_log.write(json.dumps(rec) + '\n'); self.cue_log.flush()
        if rec['status'] == 'played':
            self.cues_played += 1
            if rec.get('intent') not in ('probe', 'beep'): self.last_cue = (rec.get('text'), time.time())   # the mirror's bottom bar
            lat = '' if rec.get('t_decision') is None else f"  decision->push {1000 * (rec['t_push'] - rec['t_decision']):.0f} ms"
            if rec['intent'] == 'probe' and rec.get('t_recv') is not None:
                lat = f"  first up frame recv -> beep pushed {1000 * (rec['t_push'] - rec['t_recv']):.0f} ms (detect -> push {1000 * (rec['t_push'] - rec['t_detect']):.0f}, on-box {rec.get('on_box_ms')} ms)"
            print(f"[cue] {time.strftime('%H:%M:%S')} [{rec['intent']:10}] {rec['text']!r}{lat}", flush=True)
        else:
            print(f"[cue] {rec['status']} [{rec['intent']}] {rec['text']!r} {rec.get('why_dropped', '')}", flush=True)


def worker_options(a):
    return WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm, agent_name=a.agent_name, num_idle_processes=max(0, int(a.idle_processes)),
                         initialize_process_timeout=90.0, job_memory_warn_mb=3000)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    ap = build_parser(); a, rest = ap.parse_known_args(argv)
    if a.help or (not rest and '--help' in argv):
        ap.print_help(); return 0
    resolve_mode(a)
    if rest and rest[0] not in LK_COMMANDS and rest[0].startswith('-'):
        print(f'[worker] unknown flag {rest[0]!r} (our flags go before the LiveKit subcommand: {", ".join(LK_COMMANDS)})', file=sys.stderr); return 2
    os.environ[ARGS_ENV] = json.dumps({k: v for k, v in vars(a).items() if k != 'help'})
    if a.vision and a.voice != 'gemini':
        print('[worker] --vision only does something with --voice gemini (the video goes to the model session); nothing is forwarded', flush=True)
    print(f'[worker] agent {a.agent_name!r}; mode {describe_mode(a)}; '
          f'audio {"OFF (events only)" if a.events_only else "cue player"}; voice {a.voice}{" (" + (a.voice_model or "default model") + ", " + a.voice_name + ")" if a.voice != "off" else ""}; '
          f'handlers {a.handlers or "none"}; data channel {"on" if a.publish_data else "off"}; events -> {a.event_log}', flush=True)
    sys.argv = [sys.argv[0]] + rest                           # the LiveKit Agents CLI parses the rest (dev | start | connect | download-files)
    cli.run_app(worker_options(a))
    return 0


if __name__ == '__main__':
    sys.exit(main())
