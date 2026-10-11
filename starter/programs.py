"""Workout programs for the intake (worker --intake): at the start of the call Brodie asks the caller's goal and how hard they
want to go, and the voice calls choose_workout(goal, level), which builds the plan from these templates. Edit freely.

Templates, not model output: the voice picks a goal and a level, never reps or rest, so it cannot invent an unsafe load.
Only the four exercises counted on a device are used (squats, pushups, jumping jacks, lunges; docs/BUILD_ON_BRODIE.md
section 1). Lunges count one per leg. Reps are capped at 30: the cue bank has spoken counts up to thirty.
"""

# goal -> the exercises in order with their MEDIUM reps
GOALS = {
    'strength':   [('squats', 12), ('pushups', 10), ('lunges', 12)],
    'cardio':     [('jumping jacks', 25), ('squats', 15), ('jumping jacks', 25)],
    'full body':  [('squats', 12), ('pushups', 8), ('jumping jacks', 20), ('lunges', 10)],
}
# level -> rep multiplier and the rest between sets (seconds)
LEVELS = {
    'easy':   dict(reps=0.6, rest_s=20.0),
    'medium': dict(reps=1.0, rest_s=12.0),
    'hard':   dict(reps=1.5, rest_s=8.0),
}
DEFAULT = ('full body', 'medium')

# appended to the persona when the worker runs with --intake
INTAKE_PROMPT = (
    "\n\nAt the start of the call, before any exercise, you run a short intake. Greet the caller, say in one short sentence that this call is coached by software that reads their body pose from the video and keeps a log of rep counts and cues, not the video or the audio, then ask two things: their goal "
    "(strength, cardio or full body) and how hard they want to go (easy, medium or hard). As soon as you know both, call the "
    "choose_workout tool with them; if they are unsure, pick full body and medium and say so. After the tool answers, tell them "
    "in ONE short sentence what the workout is, then ask them to step back so their whole body is in view. Call the tool only through function calling: never write JSON, code or the tool's name in what you say. Never invent reps "
    "or exercises yourself: the tool decides them."
)


def build(goal, level):
    """(goal, level) -> (plan string for the session, rest seconds, a one-line description for the voice)."""
    goal = (goal or '').strip().lower(); level = (level or '').strip().lower()
    if goal not in GOALS: goal = DEFAULT[0]
    if level not in LEVELS: level = DEFAULT[1]
    lv = LEVELS[level]
    items = [(name, min(30, max(3, round(reps * lv['reps'])))) for name, reps in GOALS[goal]]   # the bank counts to 30
    plan = ','.join(f'{name}:{n}' for name, n in items)
    said = ', '.join(f'{n} {name}' for name, n in items)
    return plan, lv['rest_s'], f'{level} {goal}: {said}, with {lv["rest_s"]:.0f} seconds of rest between sets'
