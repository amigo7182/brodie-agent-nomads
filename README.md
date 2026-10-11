# Brodie starter

A real-time video coach in one Python process. Your phone joins a LiveKit room, the coach watches through the camera, counts reps, corrects form out loud, and talks between sets.

You write the coaching. Perception is done for you.

## What you need

- macOS or Linux, Python 3.12, [uv](https://docs.astral.sh/uv/) (`brew install uv`)
- A free LiveKit Cloud project ([cloud.livekit.io](https://cloud.livekit.io))
- A phone, or Safari on a Mac
- Your team card (Gemini team token)

## 1. Install (about 30 minutes, mostly waiting)

```
git clone https://github.com/seraphimlabs-dev/brodie-starter.git brodie-rt-coach && cd brodie-rt-coach
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
mkdir -p models && for m in lite full heavy; do curl -sSfL -o models/pose_landmarker_$m.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_$m/float16/latest/pose_landmarker_$m.task; done
```

On Linux, run `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` first to skip 3 GB of CUDA wheels.

## 2. Get the data bundle

The models, voice clips and sample recordings are not in git. Download them from this repo's release and unpack them at the repo root:

```
for p in part1_models_recordings part3_bank_puck; do curl -sSfLO https://github.com/seraphimlabs-dev/brodie-starter/releases/download/data-2026-10-09/rtcoach_starter_${p}_2026-10-09.tgz; done
tar xzf rtcoach_starter_part1_models_recordings_2026-10-09.tgz && tar xzf rtcoach_starter_part3_bank_puck_2026-10-09.tgz
```

Check it worked:

```
ls data/models/clf_pose.pt data/models/trigger_lr.json data/cues_audio_gemini/manifest.json models/pose_landmarker_full.task
```

## 3. Prove the install (2 minutes, no phone)

```
python -m facetime.replay_local --video data/tmp/mac_handoff/crops/run3.mp4 --plan "squats:10,pushups:8,jumping jacks:20,lunges:10" --fast --events data/tmp/run3.events.jsonl
python scripts/test_voice.py
```

## 4. Configure

Copy `.env.example` to `.env.local` and fill it in. The LiveKit values come from your project's Settings, then Keys. The Gemini token is on your team card.

## 5. First call

Terminal A, the coach:

```
source .venv/bin/activate && set -a && source .env.local && set +a
python -m facetime.worker --voice gemini dev
```

Terminal B, the link (use a new room name for every attempt):

```
source .venv/bin/activate && set -a && source .env.local && set +a
python -m facetime.mint_token --room team1-try1 --url
```

Open the third printed line on your phone, tap to enable audio, prop the phone up and step back. You should hear "Let's do squats" and then the counts.

Always start Terminal A before the phone joins.

## What you change

Only these three files:

| File | What it is |
| --- | --- |
| `starter/persona.md` | Who Brodie is and how it talks between sets |
| `starter/plan.txt` | The workout, e.g. `squats:10,pushups:8,jumping jacks:20,lunges:10` |
| `starter/handlers.py` | `on_event(event, say_now, ask_voice)`, called for every rep, fault, cue and set change |

- `say_now(text)` plays a pre-recorded line about 0.2 s later. Only strings in `data/cues_audio_gemini/manifest.json` work.
- `ask_voice(prompt)` asks the conversational voice, which answers 1 to 2 s later. Only call it when `session.state` is `rest` or `done`, never during a set.

## Docs

- [docs/STARTER_README.md](docs/STARTER_README.md): full setup, troubleshooting, FaceTime lines, licence
- [docs/STARTER_EVENTS.md](docs/STARTER_EVENTS.md): every event and its fields
- [docs/BUILD_ON_BRODIE.md](docs/BUILD_ON_BRODIE.md): what you can build and how

The same pages are at [seraphimlabs.dev/hackathon](https://seraphimlabs.dev/hackathon).

## What Nomads built (IvyHacks: Beat Muse)

A FaceTime fitness coach that sets up your workout by talking to you, shows you your own form on the call, and hands you a
card to share at the end. No app: call the number.

Run it (phone page or FaceTime, same command; add `--agent-name <name>` for the Chert line):
```
python -m facetime.worker --voice gemini --intake --mirror --log-caller dev
```

| Feature | What the caller gets | Where |
|---|---|---|
| Voice intake | Brodie asks your goal (strength, cardio, full body) and level (easy, medium, hard), then builds the workout from fixed templates via a `choose_workout` tool (the model never invents reps) | `--intake`, `starter/programs.py`, `facetime/session.py` |
| Camera view | Your own camera image with the pose skeleton, the rep count, Brodie's last line and his animated face, as the coach's FaceTime video | `--mirror`, `facetime/worker.py` (renderer `facetime/mirror.py`) |
| "I'm ready" | Say "let's go" during a rest and the next exercise starts (`start_next_set` tool) | `facetime/worker.py`, `facetime/session.py` |
| Honest about the camera | Ask "can you see me?" and Brodie answers from the pose tracker (`check_view` tool), not a guess | `facetime/worker.py` |
| Streak | "Day 3 in a row": counted per caller (Chert's hashed handle when present), mentioned in the greeting and the close | `facetime/recap.py` |
| Results card | A 1080x1350 PNG after every call: sets, reps, streak and next-time focus, ready to post | `facetime/recap.py`, saved to `data/cards/` |
| Boxing lessons | Pick a Soviet, Mexican or Cuban style lesson in the intake: Brodie teaches the stance, guard and big idea, then runs 3 timed shadowboxing rounds; mid-round cues come from the pose tracker's guard and stance measurements (`check_form`), and the screen shows the style and the round clock | `starter/boxing.py`, `start_boxing_lesson` in `facetime/worker.py` |
| Motivation and tips | "Halfway there", "Two more", "Last one" on the fast path; a short recovery or hydration tip every other rest; a stretch at the end | `starter/handlers.py`, `src/rtcoach/cues.json` |

Fixes to the starter we needed on the way (they help every team): the phone page now sends the caller's microphone and
joins once; a fast rejoin no longer ends the call; Brodie's prompts are sent as user-role notes, so the model no longer reads
instructions aloud; pre-recorded lines wait for Brodie to finish a sentence outside a set; a set no longer starts over his
own sentence; a set closes after 15 s out of view instead of waiting forever (`--lost-close-s`); no double "great session".

Tests: `scripts/test_intake.py`, `scripts/test_recap.py`, `scripts/test_boxing.py`, and new scenarios in `scripts/test_worker_shutdown.py`, next to the
starter's own suites. The licensed model files are untouched.

## Licence

The models and the thresholds (`clf_pose.pt`, `trigger_lr.json`, `src/rtcoach/thresholds_e3c.json`, `src/rtcoach/cue_prior.json`) come from a research evaluation licence. Use them for this hackathon only and do not redistribute them.
