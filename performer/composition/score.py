import csv

import numpy as np


class BaseEasing:
    def __init__(self, t0, duration, v0, v1):
        self.t0 = t0
        self.duration = duration
        self.v0 = v0
        self.v1 = v1

    def __call__(self, t):
        # normalize time to 0-1
        t_ = t - self.t0
        t_ = t_ / self.duration

        curve = np.piecewise(
            t_, [t_ < 0.0, (0.0 <= t_) & (t_ <= 1.0), t_ > 1.0], [0.0, self.func, 1.0]
        )

        # scale to range v0-v1
        curve *= self.v1 - self.v0
        curve += self.v0

        return curve

    @staticmethod
    def func(t):
        raise NotImplementedError()


class ExpOut(BaseEasing):
    @staticmethod
    def func(t):
        return 1.0 - np.power(2.0, -10.0 * t)


MIN_ATTACK = 0.1
MAX_ATTACK = 0.4
ATTACK_PERCENT = 0.25
RELEASE = 3.0
GAP_PERCENT = 0.15  # 15% of duration


class Envelope:
    def __init__(self, t0, duration, v0, v1):
        self._t0 = t0
        self._duration = duration
        self._v0 = v0
        self._v1 = v1
        dt_attack = self._duration * ATTACK_PERCENT
        dt_attack = MAX_ATTACK if dt_attack > MAX_ATTACK else dt_attack
        dt_attack = MIN_ATTACK if dt_attack < MIN_ATTACK else dt_attack
        self._attack_duration = dt_attack
        self._attack = ExpOut(t0, self._attack_duration, v0, v1)
        self._gap_percent_duration = GAP_PERCENT
        self._release = ExpOut(self.release_t0, RELEASE, v1, 0.0)
        self._sustain = v1

    @property
    def v0(self):
        return self._v0

    @v0.setter
    def v0(self, val):
        self._v0 = val
        self._attack.v0 = val

    @property
    def v1(self):
        return self._v0

    @v1.setter
    def v1(self, val):
        self._v1 = val
        self._attack.v1 = val
        self._sustain = val
        self._release.v0 = val

    @property
    def gap_duration(self):
        gap_duration = self._duration * self.gap_percent_duration
        if self._duration - gap_duration < self.attack_duration:
            raise ValueError("Note too short!")
        return gap_duration

    @property
    def gap_percent_duration(self):
        return self._gap_percent_duration

    @gap_percent_duration.setter
    def gap_percent_duration(self, val):
        self._gap_percent_duration = val
        self._release.t0 = self.release_t0

    @property
    def attack_duration(self):
        return self._attack_duration

    @attack_duration.setter
    def attack_duration(self, val):
        self._attack_duration = val
        self._attack.duration = val

    @property
    def sustain_duration(self):
        return self._duration - self.attack_duration - self.gap_duration

    @property
    def release_t0(self):
        return self._t0 + self._duration - self.gap_duration

    def __call__(self, t):
        t0 = self._t0
        t1 = t0 + self.attack_duration
        t2 = t1 + self.sustain_duration

        return np.piecewise(
            t,
            (t < t0, (t0 < t) & (t < t1), (t1 < t) & (t < t2), t2 < t),
            (self.v0, self._attack, self._sustain, self._release),
        )


class Note:
    def __init__(self, t0, duration, dynamic, f0):
        self.t0 = t0
        self.duration = duration
        self.dynamic = dynamic
        self.f0 = f0
        self.envelope = Envelope(t0, duration, 0.0, dynamic)
        self.is_accented = False

    def set_initial_loudness(self, val):
        self.envelope.v0 = val

    def set_slur_start(self):
        self.envelope.gap_percent_duration = 0.0

    def set_slur_end(self):
        self.envelope.attack_duration = 0.0

    def set_slur_mid(self):
        self.set_slur_start()
        self.set_slur_end()

    def set_sforzando(self):
        raise NotImplementedError("Do it!")

    def set_staccato(self):
        self.envelope.gap_percent_duration = 0.5

    def set_staccatissimo(self):
        self.envelope.gap_percent_duration = 0.75

    def set_accent(self):
        self.envelope.v1 = self.envelope._v1 + 0.1
        self.is_accented = True


def get_line(start, end, val0, val1):
    duration = end - start
    if duration == 0.0:
        return lambda t: val1
    slope = (val1 - val0) / duration

    def line(t):
        return slope * (t - start) + val0

    return line


class NoteList:
    def __init__(self):
        self.notes = []
        self.cresc = []

    def append(self, note):
        self.notes.append(note)
        self.link_notes()

    def link_notes(self):
        for note, next_note in zip(self.notes[:-1], self.notes[1:]):
            last_amp = note.envelope(next_note.t0)
            next_note.set_initial_loudness(last_amp)

    def curve(self, t):
        return np.piecewise(
            t,
            [
                t < self.notes[0].t0,
                *[
                    (note.t0 <= t) & (t < next_note.t0)
                    for note, next_note in zip(self.notes[:-1], self.notes[1:])
                ],
                t > self.notes[-1].t0,
            ],
            [0.0, *[note.envelope for note in self.notes[:-1]], self.notes[-1].envelope],
        ) * np.piecewise(
            t,
            [(a[0] <= t) & (t < b[0]) for a, b in self.cresc],
            [*[get_line(a[0], b[0], a[1], b[1]) for a, b in self.cresc], 1.0],
        )

    def freq(self, t):
        return np.piecewise(
            t,
            [
                *[
                    (note.t0 <= t) & (t < next_note.t0)
                    for note, next_note in zip(self.notes[:-1], self.notes[1:])
                ],
                t >= self.notes[-1].t0,
            ],
            [note.f0 for note in self.notes],
        )


def whole_note_sec(tempo):
    return 60 * 16 / tempo


def moment_to_sec(moment, tempo):
    return whole_note_sec(tempo) * moment


def midi_to_hz(midi: float) -> float:
    return 440.0 * np.power(2, ((midi - 69) / 12))


def hz_to_midi(hz: float) -> float:
    return 12 * np.log2(hz / 440) + 69


def parse_note(row):
    return float(row[0]), float(row[2]), float(row[4])


dynamics_map = {
    "ppp": 0.2,
    "pp": 0.3,
    "p": 0.4,
    "mp": 0.5,
    "mf": 0.6,
    "f": 0.7,
    "ff": 0.8,
    "fff": 0.9,
}


def parser(path: str):
    notes = NoteList()
    with open(path) as csvfile:
        current_tempo = None
        current_note = None
        is_in_slur = False
        current_dynamic = 0.6
        is_in_cresc = False
        cresc_start = None

        for row in csv.reader(csvfile, delimiter="\t"):
            if current_tempo is not None:
                time = moment_to_sec(float(row[0]), current_tempo)
                if (current_note is not None) and (time > current_note.t0):
                    notes.append(current_note)
                    current_note = None

            match row:
                case [_, "tempo", tempo]:
                    current_tempo = float(tempo)
                case [_, "note", pitch, _, duration, *_]:
                    pitch = float(pitch)
                    duration = float(duration)
                    f0 = midi_to_hz(pitch)
                    duration = moment_to_sec(duration, current_tempo)
                    current_note = Note(time, duration, current_dynamic, f0)
                    if is_in_slur:
                        current_note.set_slur_mid()
                case [_, "rest", *_]:
                    continue
                case [_, "slur", value]:
                    if int(value) == -1:
                        current_note.set_slur_start()
                    else:
                        current_note.set_slur_end()
                case [_, "script", "accent"]:
                    current_note.set_accent()
                case [_, "script", "staccato"]:
                    current_note.set_staccato()
                case [_, "script", "staccatissimo"]:
                    current_note.set_staccatissimo()
                case [_, "dynamic", value]:
                    dynamic = dynamics_map[value]
                    if is_in_cresc:
                        notes.cresc.append((cresc_start, (time, dynamic / current_dynamic)))
                        is_in_cresc = False
                    current_dynamic = dynamic
                    if current_note.is_accented:
                        current_note.envelope.v1 = current_dynamic + 0.1
                    else:
                        current_note.envelope.v1 = current_dynamic
                case [_, "cresc" | "decresc", value]:
                    cresc_start = (time, 1.0)
                    is_in_cresc = True
                case _:
                    print(f'<NA>\ttime: {time:.2f} kind: {row[1]} values: {" - ".join(row[2:])}')

    return notes
