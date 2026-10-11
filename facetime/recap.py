"""After the call: the streak (days in a row with a workout) and a shareable results card (PNG), from the event stream.

    recap = CallRecap()                    # one per call; recap.on_event(ev) for every event (any thread, events arrive in order)
    streaks = Streaks()                    # data/brodie/streaks.json, keyed by profile (the caller's hashed handle when Chert gives one)
    day = streaks.record(profile)          # once per call, at the first finished set: {'current', 'best', 'new_day', 'last'}
    render_card(recap, day, path)          # 1080x1350 PNG (4:5, fits LinkedIn, Instagram and TikTok covers)
    card_frame(card_image(recap, day))     # the same card on a 720x1280 frame: the coach's video shows it at the end of the call

The card is text only: the BRODIE wordmark, no drawing of the character (the hackathon's character kit governs any Brodie art).
Nothing here leaves the laptop: the card and the streak file are written under data/ (never in git).
"""
import datetime, json, os, threading
from PIL import Image, ImageDraw, ImageFont
from facetime import DATA

STREAKS_PATH = os.path.join(DATA, 'brodie', 'streaks.json')
CARDS_DIR = os.path.join(DATA, 'cards')
BLACK = (10, 10, 10); VOLT = (217, 255, 61); WHITE = (245, 245, 245); GREY = (150, 150, 150)
W, H = 1080, 1350
FONTS_BOLD = ['/System/Library/Fonts/Supplemental/Arial Black.ttf', '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf']
FONTS_REG = ['/System/Library/Fonts/Supplemental/Arial.ttf', '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
             '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']


def profile_key(md):
    """The caller's stable key: Chert's hashed handle when the job metadata carries one, else one local profile."""
    call = md.get('chert_call') if isinstance(md, dict) else None
    if isinstance(call, str):
        try: call = json.loads(call)
        except ValueError: call = None
    h = (call or {}).get('from_handle_hash') if isinstance(call, dict) else None
    return f'handle:{h}' if h else 'default'


class Streaks:
    """Days in a row with at least one finished set, per profile, in local dates."""
    def __init__(self, path=STREAKS_PATH):
        self.path = path; self.lock = threading.Lock()

    def _load(self):
        try:
            with open(self.path) as f: return json.load(f)
        except (OSError, ValueError): return {}

    def peek(self, profile, today=None):
        """The streak as it stands before today's workout: {'current', 'best', 'last'} (current 0 when it has lapsed)."""
        today = today or datetime.date.today()
        with self.lock: d = self._load().get(profile) or {}
        last = d.get('last'); cur = d.get('current', 0)
        if last and (today - datetime.date.fromisoformat(last)).days > 1: cur = 0
        return dict(current=cur, best=d.get('best', 0), last=last)

    def record(self, profile, today=None):
        """Today counts: +1 when the last workout was yesterday, unchanged when it was today, 1 after a gap."""
        today = today or datetime.date.today()
        with self.lock:
            data = self._load(); d = data.get(profile) or {}
            last = datetime.date.fromisoformat(d['last']) if d.get('last') else None
            if last == today: cur = d.get('current', 1); new_day = False
            else: cur = d.get('current', 0) + 1 if last and (today - last).days == 1 else 1; new_day = True
            d = dict(current=cur, best=max(cur, d.get('best', 0)), last=today.isoformat()); data[profile] = d
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + '.tmp'
            with open(tmp, 'w') as f: json.dump(data, f, indent=1)
            os.replace(tmp, self.path)
        return dict(d, new_day=new_day)


class CallRecap:
    """What the call achieved, from the events: the sets (exercise, reps, target), the corrections spoken, the plan."""
    def __init__(self):
        self.sets = []; self.corrections = {}; self._ex = None; self._target = None; self.plan_label = None; self.started = datetime.datetime.now()
        self.lesson = None                  # a boxing lesson: {style, title, drills}

    def on_event(self, ev):
        t = ev.get('type')
        if t == 'session':
            st = ev.get('state')
            if st in ('wait', 'active') and ev.get('exercise'): self._ex = ev['exercise']; self._target = ev.get('target')
            elif st in ('rest', 'done') and self._ex is not None:
                self.sets.append(dict(exercise=self._ex, reps=int(ev.get('count') or 0), target=self._target)); self._ex = None
        elif t == 'cue' and ev.get('intent') == 'correction' and ev.get('text'):
            self.corrections[ev['text']] = self.corrections.get(ev['text'], 0) + 1

    @property
    def total_reps(self): return sum(s['reps'] for s in self.sets)

    @property
    def focus(self): return max(self.corrections, key=self.corrections.get) if self.corrections else None


def _font(paths, size):
    for p in paths:
        try: return ImageFont.truetype(p, size)
        except OSError: continue
    return ImageFont.load_default(size=size)


def render_card(recap, day=None, path=None, now=None):
    """The results card: returns the path written. day: Streaks.record()'s dict or None."""
    now = now or datetime.datetime.now()
    path = path or os.path.join(CARDS_DIR, f'brodie-{now:%Y%m%d-%H%M%S}.png')
    img = card_image(recap, day, now)
    os.makedirs(os.path.dirname(path), exist_ok=True); img.save(path)
    return path


def card_frame(img, w=720, h=1280):
    """The card fitted onto a portrait video frame (black bars above and below), RGB uint8 HxWx3, for the coach's video track."""
    import numpy as np
    sc = min(w / img.width, h / img.height); cw, ch = int(img.width * sc), int(img.height * sc)
    out = Image.new('RGB', (w, h), BLACK); out.paste(img.resize((cw, ch), Image.LANCZOS), ((w - cw) // 2, (h - ch) // 2))
    return np.asarray(out)


def card_image(recap, day=None, now=None):
    """The card as a PIL image (1080x1350)."""
    now = now or datetime.datetime.now()
    img = Image.new('RGB', (W, H), BLACK); d = ImageDraw.Draw(img)
    big, mid, small, tiny = _font(FONTS_BOLD, 120), _font(FONTS_BOLD, 64), _font(FONTS_REG, 40), _font(FONTS_REG, 32)
    x = 80
    d.text((x, 70), 'BRODIE', font=mid, fill=VOLT)
    d.text((W - x, 92), now.strftime('%a %d %b %Y').upper(), font=tiny, fill=GREY, anchor='ra')
    lesson = recap.lesson
    d.text((x, 190), 'LESSON' if lesson else 'WORKOUT', font=big, fill=WHITE); d.text((x, 320), 'DONE.', font=big, fill=VOLT)
    if recap.plan_label: d.text((x, 470), recap.plan_label.upper(), font=small, fill=GREY)
    y = 560
    if lesson:                                                         # the drills taught, one per round
        for i, name in enumerate(lesson['drills'][:4], 1):
            d.text((x, y), f'R{i}', font=mid, fill=VOLT)
            d.text((x + 150, y + 14), name[:34].upper(), font=small, fill=WHITE); y += 100
        d.line([(x, y + 10), (W - x, y + 10)], fill=GREY, width=2); y += 40
        d.text((x, y), f"{len(lesson['drills'])} ROUNDS", font=mid, fill=WHITE)
        if day and day.get('current'): d.text((W - x, y), f"DAY {day['current']} STREAK", font=mid, fill=VOLT, anchor='ra')
        d.rectangle([0, H - 150, W, H], fill=VOLT)
        d.text((x, H - 112), 'Coached live on FaceTime. No app.', font=small, fill=BLACK)
        d.text((x, H - 62), '#IvyHacksBeatMuse', font=tiny, fill=BLACK)
        return img
    for s in recap.sets[:5]:
        d.text((x, y), s['exercise'].upper(), font=mid, fill=WHITE)
        reps = f"{s['reps']}" + (f" / {s['target']}" if s.get('target') else '')
        d.text((W - x, y), reps, font=mid, fill=VOLT, anchor='ra'); y += 100
    d.line([(x, y + 10), (W - x, y + 10)], fill=GREY, width=2); y += 40
    d.text((x, y), f'{recap.total_reps} REPS', font=mid, fill=WHITE)
    if day and day.get('current'):
        d.text((W - x, y), f"DAY {day['current']} STREAK", font=mid, fill=VOLT, anchor='ra')
    y += 110
    if recap.focus: d.text((x, y), f'Next time: {recap.focus.lower()}', font=small, fill=GREY)
    d.rectangle([0, H - 150, W, H], fill=VOLT)
    d.text((x, H - 112), 'Coached live on FaceTime. No app.', font=small, fill=BLACK)
    d.text((x, H - 62), '#IvyHacksBeatMuse', font=tiny, fill=BLACK)
    return img


if __name__ == '__main__':   # python -m facetime.recap [events.jsonl]: a card from a call log (the last call in the file), for a look
    import sys
    r = CallRecap()
    if len(sys.argv) > 1:
        evs = [json.loads(l) for l in open(sys.argv[1])]
        rooms = [e.get('room') for e in evs if e.get('room')]
        last = rooms[-1] if rooms else None
        for e in evs:
            if last is None or e.get('room') == last: r.on_event(e)
    else:
        for e in [dict(type='session', state='wait', exercise='squats', target=7), dict(type='cue', intent='correction', text='Hips back'),
                  dict(type='session', state='rest', count=7), dict(type='session', state='wait', exercise='pushups', target=6),
                  dict(type='session', state='rest', count=6), dict(type='session', state='wait', exercise='lunges', target=7),
                  dict(type='session', state='done', count=7)]: r.on_event(e)
        r.plan_label = 'easy strength'
    print(render_card(r, dict(current=3, best=5)))
