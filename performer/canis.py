"""Notes:
    a, d, r times are constant for the instrument
    peak = 1, sustain = 0.95
    envelope generator takes start and duration
    notes can be:
    - individual: above described
    - slur begin: no release
    - slur middle: no attack, no release
    - slur end: no attack
    pre-attack, post-release (-80, -100)
    accent adds a constant dB (?)
    sfz changes sustain/peak ratio
    staccato halves duration

    fff, ff, f, mf, mp, p, pp, ppp = map flute dB range to these (-80, -10) increments of 10
    cresc., decresc. create line from d0 to d1 with duration of dynamic change

    times are in whole notes (lilypond moments).
    tempo indication create a `seconds_to_moments` function
    rit., accel. modifies this mapping, linearly or exponentially
"""
import numpy as np

# TODO: rit, accel. curves
# TODO: dB range
# TODO: vibrato
# TODO: humanize (add noise)

ATTACK = 0.1
DECAY = 0.4
RELEASE = 0.04
TAIL = 1.5
SUSTAIN_AMP = 0.97
ACCENT = 0.1
NIENTE = 0.0
PPP = 0.2


def get_line(start, duration, val0, val1):
    if duration == 0.0:
        return lambda t: (val0 + val1) / 2
    slope = (val1 - val0) / duration

    def line(t):
        return slope * (t - start) + val0

    return line


class ADSR:
    def __init__(self, start, duration):
        self.start = start
        self.duration = duration
        self.peak = 1.0
        self.sustain = SUSTAIN_AMP
        self.attack = ATTACK
        self.decay = DECAY
        self.release = RELEASE
        self.val_start = 0.0
        self.val_end = 0.0
        self.staccato = 1.0

    def set_slur_start(self):
        self.release = 0.0
        self.val_end = self.sustain

    def set_slur_end(self):
        self.attack = 0.0
        self.decay = 0.0
        self.val_start = self.sustain
        self.peak = self.sustain

    def set_slur_mid(self):
        self.set_slur_start()
        self.set_slur_end()

    def set_accent(self):
        self.peak += ACCENT
        self.sustain += ACCENT

    def set_sforzando(self):
        # self.peak += ACCENT
        self.sustain -= ACCENT

    def set_staccato(self):
        self.staccato = 2

    def set_staccatissimo(self):
        self.staccato = 4

    def get_envelope_func(self):
        if self.duration < self.attack + self.decay + self.release:
            raise ValueError("Note too short!")

        sustain_duration = self.duration / self.staccato - self.attack - self.decay - self.release
        t0 = self.start
        t1 = self.start + self.attack
        t2 = t1 + self.decay
        t3 = t2 + sustain_duration

        a = get_line(t0, self.attack, self.val_start, self.peak)
        d = get_line(t1, self.decay, self.peak, self.sustain)
        r = get_line(t3, TAIL, self.sustain, self.val_end)

        def rr(t):
            return np.maximum(r(t), 0.0)

        def envelope(t):
            return np.piecewise(
                t,
                (
                    t < t0,
                    (t0 <= t) & (t < t1),
                    (t1 <= t) & (t < t2),
                    (t2 <= t) & (t < t3),
                    t3 <= t,
                ),
                (0, a, d, self.sustain, rr),
            )

        return envelope

    def get_boundaries(self):
        return self.start, self.start + self.duration

    def __ge__(self, other):
        return self.start >= other.start
