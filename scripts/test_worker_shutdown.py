"""facetime/worker.py job end with a fake room and job context (no LiveKit server): the entrypoint must return on its own, promptly,
because livekit-agents runs shutdown callbacks only after the entrypoint returns (Mac test 2026-10-09: "entrypoint did not exit in
time", the job process killed after 10 s, on every call end and on `chert doctor --agent` rooms).

Scenarios (each with a fresh pipeline; the cue player uses a real rtc.AudioSource, no room needed):
 A. connect() raises (a doctor or warm room torn down before the peer connection completes): run() returns at once, ctx.shutdown called.
 B. warm job, connected, then the room's `disconnected` event: run() returns within 1.5 s, feed thread stopped, player stopped.
 C. a bound caller leaves and nobody rebinds within --leave-grace-s (0.5 s here): run() returns about 0.5 s later, reason "caller left".
 D. the caller leaves and rebinds within the grace: the job keeps running; then a disconnect ends it. aclose() is idempotent.
 E. --perception off (voice mode): no Perception and no pipeline are built, no feed thread; a fake video track binds the caller
    (the VideoStream is replaced by a fake 10 fps source), ONE synthetic session {state: open, mode: voice} event is emitted,
    a status event (no pose fields) follows, a VideoTee without a model session forwards nothing (frames counted as early);
    the caller's leave ends the job within the grace, the event log has the session line.
 F. --perception off, a caller with audio only (no video): the audio track binds it, the session event is emitted, a disconnect
    ends the job. A warm job with nobody publishing offers no frame to the tee.

Run:  .venv/bin/python scripts/test_worker_shutdown.py
"""
import asyncio, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from livekit import rtc
from facetime import worker
from facetime.perception import Config, build_pipeline

PASS = []


def check(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg, flush=True); PASS.append(bool(cond)); return cond


class FakeLocal:
    identity = 'brodie-coach'
    async def publish_track(self, track, opts): return None
    async def publish_data(self, *a, **k): return None


class FakeRoom:
    def __init__(self):
        self.handlers = {}; self.name = 'chert-warm-fake'; self.local_participant = FakeLocal(); self.remote_participants = {}
        self.connection_state = rtc.ConnectionState.CONN_CONNECTED
    def on(self, name, fn=None):
        if fn is None: return lambda f: self.on(name, f)
        self.handlers.setdefault(name, []).append(fn); return fn
    def emit(self, name, *args):
        for fn in self.handlers.get(name, []): fn(*args)


class FakeJobInfo:
    id = 'job-fake'; metadata = json.dumps({'chert_warm': True})


class FakeCtx:
    def __init__(self, room, connect_error=None):
        self.room = room; self.job = FakeJobInfo(); self.callbacks = []; self.shutdowns = []; self.connect_error = connect_error; self.connected = False
    def add_shutdown_callback(self, fn): self.callbacks.append(fn)
    async def connect(self, auto_subscribe=None):
        await asyncio.sleep(0.05)
        if self.connect_error: raise self.connect_error
        self.connected = True
    def shutdown(self, reason='user requested'): self.shutdowns.append(reason)


class P:
    def __init__(self, identity): self.identity = identity; self.kind = rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD


def make_job(ctx, cfg, events_only=False, leave_grace_s=0.5, perception='on', event_log=None):
    a = worker.load_args(); a.events_only = events_only; a.leave_grace_s = leave_grace_s; a.plan = 'squats:3'
    a.perception = perception; a.voice = 'off'; a.vision = False; a.floor = None; worker.resolve_mode(a)     # voice off: no model session in these tests
    a.event_log = event_log or os.path.join(ROOT, 'data', 'tmp', 'test_shutdown_events.jsonl'); a.cue_log = os.path.join(ROOT, 'data', 'tmp', 'test_shutdown_cues.jsonl')
    pipe = None if perception == 'off' else build_pipeline(cfg)
    bank = None
    if not events_only:
        from facetime.audio_out import CueBank
        bank = CueBank()
    return worker.CoachJob(ctx, a, cfg, pipe, bank, worker.parse_metadata(ctx.job.metadata))


class FakeFrame:
    width = 1280; height = 720


class FakeFrameEvent:
    def __init__(self, i): self.frame = FakeFrame(); self.timestamp_us = int(time.time() * 1e6)


class FakeVideoStream:
    """Stands in for rtc.VideoStream: one frame event every 1/fps s until aclose()."""
    def __init__(self, fps=10.0): self.fps = fps; self.closed = False; self.n = 0
    def __aiter__(self): return self
    async def __anext__(self):
        if self.closed: raise StopAsyncIteration
        await asyncio.sleep(1.0 / self.fps); self.n += 1
        if self.closed: raise StopAsyncIteration
        return FakeFrameEvent(self.n)
    async def aclose(self): self.closed = True


class FakeTrack:
    def __init__(self, kind): self.kind = kind


class FakePub:
    def __init__(self, name): self.name = name; self.width = 1280; self.height = 720; self.simulcasted = False


def drain(q):
    out = []
    while not q.empty(): out.append(q.get_nowait())
    return out


async def run_with_trigger(job, trigger, limit):
    """Run the job; `trigger(job)` fires after 0.3 s; returns (seconds from the trigger to run() returning, exception)."""
    t = asyncio.create_task(job.run()); await asyncio.sleep(0.3)
    t0 = time.monotonic(); trigger(job)
    try: await asyncio.wait_for(t, limit); exc = None
    except asyncio.TimeoutError: exc = 'TIMEOUT'; t.cancel()
    except Exception as e: exc = e
    return time.monotonic() - t0, exc


async def main():
    cfg = Config(plan='squats:3'); keep = []                       # jobs stay referenced: no mid-test garbage collection of MediaPipe objects
    print('[A] connect() raises (doctor/warm room torn down early)', flush=True)
    ctx = FakeCtx(FakeRoom(), connect_error=RuntimeError('wait_pc_connection timed out')); job = make_job(ctx, cfg, events_only=True); keep.append(job)
    t0 = time.monotonic()
    try: await asyncio.wait_for(job.run(), 5); exc = None
    except Exception as e: exc = e
    dt = time.monotonic() - t0
    check(exc is None and dt < 1.0 and ctx.shutdowns == ['connect failed'] and job.closed and job.pipe.pose.lm._handle is None, f'run() returned in {dt * 1000:.0f} ms, no exception, ctx.shutdown("connect failed"), closed, pose landmarker closed')

    print('[B] warm job, room disconnected (cue player attached)', flush=True)
    ctx = FakeCtx(FakeRoom()); job = make_job(ctx, cfg); keep.append(job)
    dt, exc = await run_with_trigger(job, lambda j: ctx.room.emit('disconnected', rtc.DisconnectReason.ROOM_DELETED), 5)
    check(exc is None and dt < 1.5, f'run() returned {dt * 1000:.0f} ms after the disconnect (limit 1500)')
    check(not job.thread.is_alive() and job.player._stop and job.player_task.done() and job.per.closed and job.closed, 'feed thread stopped, player stopped and its task done, perception closed')
    check(ctx.shutdowns and ctx.shutdowns[0].startswith('room disconnected') and ctx.connected, f'ctx.shutdown called: {ctx.shutdowns}')
    t1 = time.monotonic(); await job.aclose(); check(time.monotonic() - t1 < 0.01, 'aclose() again returns at once (idempotent)')

    print('[C] bound caller leaves, nobody rebinds within the grace (0.5 s)', flush=True)
    ctx = FakeCtx(FakeRoom()); job = make_job(ctx, cfg); keep.append(job)
    def leave(j): j.bound = 'chert-bridge-abc'; j.on_participant_disconnected(P('chert-bridge-abc'))
    dt, exc = await run_with_trigger(job, leave, 5)
    check(exc is None and 0.4 < dt < 1.6 and job.end_reason == 'caller left', f'run() returned {dt * 1000:.0f} ms after the leave, reason {job.end_reason!r}')
    check(ctx.shutdowns == ['caller left'] and not job.thread.is_alive() and job.player._stop, 'ctx.shutdown("caller left"), thread and player stopped')

    print('[D] caller leaves and rebinds within the grace: the job stays; a disconnect then ends it', flush=True)
    ctx = FakeCtx(FakeRoom()); job = make_job(ctx, cfg, events_only=True); keep.append(job)
    t = asyncio.create_task(job.run()); await asyncio.sleep(0.3)
    job.bound = 'chert-bridge-abc'; job.on_participant_disconnected(P('chert-bridge-abc')); await asyncio.sleep(0.2); job.bound = 'chert-bridge-abc'   # rebound
    await asyncio.sleep(0.8)
    check(not t.done() and job.end_reason is None and job.leave_task is None, 'still running 1 s after the leave (rebound within the grace)')
    t0 = time.monotonic(); ctx.room.emit('disconnected', rtc.DisconnectReason.CLIENT_INITIATED)
    try: await asyncio.wait_for(t, 5); exc = None
    except Exception as e: exc = e
    check(exc is None and time.monotonic() - t0 < 1.5 and job.end_reason.startswith('room disconnected'), f'ended {1000 * (time.monotonic() - t0):.0f} ms after the disconnect: {job.end_reason}')

    print('[E] --perception off: no Perception, binds on a video track, synthetic session open event, status, ends on the leave', flush=True)
    from facetime.voice import VideoTee
    elog = os.path.join(ROOT, 'data', 'tmp', f'test_shutdown_voice_{int(time.time())}.jsonl')
    ctx = FakeCtx(FakeRoom()); ctx.job.metadata = json.dumps({'chert_call': 'fake'}); job = make_job(ctx, cfg, perception='off', event_log=elog); keep.append(job)
    job.cfg.status_every_s = 0.6                                           # a status event within the scenario
    job.vision = VideoTee(fps=1.0)                                         # as --vision would, but no model session ever starts here: nothing may be forwarded
    stream = FakeVideoStream(fps=10.0); job.video_frames = lambda track: stream
    seen = []; orig = job._deliver; job._deliver = lambda ev: (seen.append(ev), orig(ev))
    check(job.per is None and job.pipe is None and job.voice_mode and job.a.floor == 'open', 'no pipeline, no Perception object, floor resolves to open')
    t = asyncio.create_task(job.run()); await asyncio.sleep(0.3)
    check(job.per is None and job.thread is None and job.event_writer is not None and job.session_opened is False and seen == [], 'after connect: still no Perception, no feed thread, no event yet')
    job.on_track_subscribed(FakeTrack(rtc.TrackKind.KIND_VIDEO), FakePub('facetime-remote-video'), P('chert-bridge-e1'))
    await asyncio.sleep(1.0)
    sess = [e for e in seen if e['type'] == 'session']; stat = [e for e in seen if e['type'] == 'status']
    check(job.bound == 'chert-bridge-e1' and job.call_id == 'e1' and job.video_task is not None and job.seq >= 5, f'bound on the video track, {job.seq} frames received')
    check(len(sess) == 1 and sess[0]['state'] == 'open' and sess[0]['exercise'] is None and sess[0]['count'] == 0 and sess[0]['mode'] == 'voice' and sess[0]['v'] == '1' and 'frame' in sess[0],
          f'ONE synthetic session event: {dict((k, sess[0][k]) for k in ("state", "exercise", "count", "mode")) if sess else None}')
    check(len(stat) >= 1 and stat[-1]['mode'] == 'voice' and 'pose_ms' not in stat[-1] and stat[-1]['fps_recv'] > 5 and stat[-1]['state'] == 'open' and stat[-1]['vision_frames'] == 0,
          f'status event without pose fields: {stat[-1] if stat else None}')
    check(job.vision.n_offered >= 5 and job.vision.n_sent == 0 and job.vision.n_queued == 0 and job.vision.n_early == job.vision.n_offered,
          f'tee without a model session: {job.vision.n_offered} frames offered, 0 forwarded ({job.vision.n_early} early)')
    job.on_track_subscribed(FakeTrack(rtc.TrackKind.KIND_VIDEO), FakePub('other'), P('chert-bridge-e2'))
    await asyncio.sleep(0.1)
    check(job.bound == 'chert-bridge-e1' and len([e for e in seen if e['type'] == 'session']) == 1, 'a second video track is ignored while bound, no second session event')
    t0 = time.monotonic(); job.on_participant_disconnected(P('chert-bridge-e1'))
    try: await asyncio.wait_for(t, 5); exc = None
    except Exception as e: exc = e
    dt = time.monotonic() - t0
    check(exc is None and 0.4 < dt < 1.6 and job.end_reason == 'caller left' and ctx.shutdowns == ['caller left'] and stream.closed and job.closed, f'ended {dt * 1000:.0f} ms after the leave: {job.end_reason!r}, stream closed')
    lines = [json.loads(l) for l in open(elog)] if os.path.exists(elog) else []
    check(any(l['type'] == 'session' and l['state'] == 'open' and l['call_id'] == 'e1' for l in lines) and all('pose_ms' not in l for l in lines), f'{len(lines)} JSONL lines, the session open line carries call_id')
    try: os.remove(elog)
    except OSError: pass

    print('[F] --perception off: a caller with audio only binds too; a warm job offers nothing to the tee', flush=True)
    ctx = FakeCtx(FakeRoom()); job = make_job(ctx, cfg, perception='off', event_log=elog); keep.append(job); job.vision = VideoTee(fps=1.0)
    seen = []; orig = job._deliver; job._deliver = lambda ev: (seen.append(ev), orig(ev))
    t = asyncio.create_task(job.run()); await asyncio.sleep(0.3)
    job.on_track_subscribed(FakeTrack(rtc.TrackKind.KIND_AUDIO), FakePub('facetime-remote-audio'), P('chert-bridge-f1'))
    await asyncio.sleep(0.2)
    check(job.bound == 'chert-bridge-f1' and job.video_task is None and sum(1 for e in seen if e['type'] == 'session') == 1, 'audio-only caller: bound, session event emitted, no video task')
    t0 = time.monotonic(); ctx.room.emit('disconnected', rtc.DisconnectReason.ROOM_DELETED)
    try: await asyncio.wait_for(t, 5); exc = None
    except Exception as e: exc = e
    check(exc is None and time.monotonic() - t0 < 1.5 and job.end_reason.startswith('room disconnected') and job.vision.n_offered == 0, f'ended on the disconnect ({job.end_reason}), no frame offered')
    ctx = FakeCtx(FakeRoom()); job = make_job(ctx, cfg, perception='off', event_log=elog); keep.append(job); job.vision = VideoTee(fps=1.0)
    seen = []; orig = job._deliver; job._deliver = lambda ev: (seen.append(ev), orig(ev))
    dt, exc = await run_with_trigger(job, lambda j: ctx.room.emit('disconnected', rtc.DisconnectReason.ROOM_DELETED), 5)
    check(exc is None and job.warm and not job.session_opened and seen == [] and job.vision.n_offered == 0 and job.vision.n_sent == 0, 'warm job: no session event, nothing offered or forwarded')
    print('[G] a fast rejoin: the caller leaves and republishes while its old stream is still open; the new track binds, the job stays', flush=True)
    ctx = FakeCtx(FakeRoom()); ctx.job.metadata = json.dumps({'chert_call': 'fake'}); job = make_job(ctx, cfg, perception='off', event_log=elog); keep.append(job)
    streams = []; job.video_frames = lambda track: (streams.append(FakeVideoStream(fps=10.0)), streams[-1])[1]
    t = asyncio.create_task(job.run()); await asyncio.sleep(0.3)
    job.on_track_subscribed(FakeTrack(rtc.TrackKind.KIND_VIDEO), FakePub('facetime-remote-video'), P('chert-bridge'))
    await asyncio.sleep(0.3); first = job.video_task
    job.on_participant_disconnected(P('chert-bridge'))                     # the old stream is still running (as on the 10-10 phone page)
    job.on_track_subscribed(FakeTrack(rtc.TrackKind.KIND_VIDEO), FakePub('facetime-remote-video'), P('chert-bridge'))
    await asyncio.sleep(1.0)                                               # past the 0.5 s grace
    check(not t.done() and job.end_reason is None and job.bound == 'chert-bridge' and job.video_task is not first and first.done()
          and len(streams) == 2 and streams[0].closed and not streams[1].closed, 'rebound on the new track, old stream closed, job still running past the grace')
    job.on_track_subscribed(FakeTrack(rtc.TrackKind.KIND_VIDEO), FakePub('facetime-remote-video'), P('someone-else'))
    check(len(streams) == 2, 'a second caller while one is bound is still ignored')
    ctx.room.emit('disconnected', rtc.DisconnectReason.ROOM_DELETED)
    try: await asyncio.wait_for(t, 5); exc = None
    except Exception as e: exc = e
    check(exc is None and job.end_reason.startswith('room disconnected'), f'ended on the disconnect ({job.end_reason})')
    print('[H] --mirror: a coach-video track is published at join and fed (the face alone, then the composed caller frame); the job still ends promptly', flush=True)
    import numpy as np
    ctx = FakeCtx(FakeRoom()); ctx.job.metadata = json.dumps({'chert_call': 'fake'}); job = make_job(ctx, cfg); keep.append(job); job.a.mirror = True; job.a.mirror_fps = 10.0
    pubs = []; orig_pub = ctx.room.local_participant.publish_track
    async def pub(track, opts): pubs.append(track); return await orig_pub(track, opts)
    ctx.room.local_participant.publish_track = pub
    frames = []; orig_frame = job._mirror_frame
    job._mirror_frame = lambda *a: (frames.append(job.mirror_src is not None), orig_frame(*a))[1]
    t = asyncio.create_task(job.run()); await asyncio.sleep(1.0)
    n_face = len(frames)
    job.mirror_src = (np.zeros((720, 1280, 3), np.uint8), None); job.last_cue = ('Sink lower', time.time())   # as the feed thread sets it after a frame
    await asyncio.sleep(0.6)
    check(any(getattr(p, 'name', '') == 'coach-video' for p in pubs) and job.mirror_task is not None and not job.mirror_task.done(), f'coach-video published, mirror running ({len(pubs)} tracks)')
    check(n_face >= 5 and any(frames[n_face:]), f'{n_face} face-only frames before a caller frame, then {sum(frames[n_face:])} composed frames')
    t0 = time.monotonic(); ctx.room.emit('disconnected', rtc.DisconnectReason.ROOM_DELETED)
    try: await asyncio.wait_for(t, 5); exc = None
    except Exception as e: exc = e
    check(exc is None and time.monotonic() - t0 < 1.5 and job.mirror_task.done(), f'ended {1000 * (time.monotonic() - t0):.0f} ms after the disconnect, mirror stopped')
    print('[I] outside a set, bank lines wait while the voice talks (no chopped sentence); counts never wait; stale lines are dropped', flush=True)
    ctx = FakeCtx(FakeRoom()); job = make_job(ctx, cfg); keep.append(job); job.loop = asyncio.get_running_loop()
    said = []
    class FakePlayer:
        mix = type('M', (), {'pending': 0})()
        def say(self, text, intent, meta=None): said.append(text)
    job.player = FakePlayer(); talking = {'on': True}
    job._voice_audible = lambda: talking['on']
    job.session_state = 'rest'
    job._say_or_defer('Step back so I can see your whole body', 'visibility', {}); job._say_or_defer('Rest for a moment', 'positive', {})
    await asyncio.sleep(0.3)
    check(said == [] and len(job.deferred_cues) == 2, f'two lines wait while the voice talks (said {said})')
    talking['on'] = False; await asyncio.sleep(0.6)
    check(said == ['Step back so I can see your whole body', 'Rest for a moment'] and not job.deferred_cues, f'played in order once the voice was quiet: {said}')
    said.clear(); talking['on'] = True; job.session_state = 'active'
    job._say_or_defer('Three', 'count', {}); job._say_or_defer('Step back so I can see your whole body', 'visibility', {})
    check(said == ['Three', 'Step back so I can see your whole body'], f'during a set nothing waits: {said}')
    said.clear(); job.session_state = 'rest'
    job._say_or_defer('Rest for a moment', 'positive', {}); job._say_or_defer("Let's do pushups", 'announce', {})
    await asyncio.sleep(0.2); job.session_state = 'active'; await asyncio.sleep(0.3)
    check(said == ["Let's do pushups"], f'a set started while they waited: the stale praise is dropped, the announce still plays ({said})')
    job.player = None; await job.aclose()                                  # the fake player has no audio to drain
    try: os.remove(elog)
    except OSError: pass
    n = len(PASS); ok = sum(PASS); print(f'\n{ok}/{n} checks passed' + ('' if ok == n else '  <-- FAILURES'), flush=True)
    return 0 if ok == n else 1


if __name__ == '__main__':
    rc = asyncio.run(main()); sys.stdout.flush(); os._exit(rc)            # three MediaPipe pipelines in one process: skip the interpreter's __del__ pass
