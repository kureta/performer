% Utilities to re-tune regular tunings.

% Use these functions to choose a different regular
% tuning, with fifths taken from some equal temperament,
% and accidentals scaled accordingly.

% Only equal temperaments are supported because Lilypond uses
% rational numbers to denote pitch alterations.
% If you don't want an equal temperament you can probably
% find one anyway that's close enough to what you want.

% to use, add
%
%     tuning = #31
%     \include "regular.ly"
%
% somewhere near the top of the file.  (After any language setting.)

% Resets Lilypond's "default scale" containing the pitch of each
% unaltered note (the C major scale).
#(define (retune-nominals ET)
    (ly:set-default-scale (ly:make-scale (list->vector
        (map (lambda (fifths octaves) (* 6
                  (+ (* fifths (best-fifth ET)) octaves)))
           '(0 2 4 -1 1 3 5) '(0 -1 -2 1 0 -1 -2))))))

% Finds the best size of fifth in the equal temperament with
% the given number of fifths to the octave.
% Note: the effective equal temperament may end up larger.
% For example, ask for 12 and "quartertones" will give you 24.
% If ET is a ratio, assume it describes the fifth, and return it.
#(define (best-fifth ET)
    (if (integer? ET)
        (/ (inexact->exact (round (* ET 0.5849625007))) ET)
        ET))

% Takes the association of pitch names and returns a
% new copy where each alteration has the correct value
% relative to fifths in the given equal temperament.
#(define (retuned-pitchnames pitchnames ET)
    (map (lambda (pitchname)
             (let ((pitch (cdr pitchname)))
                 (cons (car pitchname)
                     (ly:make-pitch
                         (ly:pitch-octave pitch)
                         (ly:pitch-notename pitch)
                         (scale-alteration (ly:pitch-alteration pitch) ET)))))
         pitchnames))

% Takes a list mapping alterations to glyphs
% and re-tunes the alterations according to the size of fifth
% in the given equal temperament.
#(define (retune-glyphs glyphs ET)
    (map (lambda (glyph) (cons (scale-alteration (car glyph) ET) (cdr glyph)))
         glyphs))

% Converts an alteration from the initial alteration size
% (that would give 12-equal) to the given equal temperament.
#(define (scale-alteration alteration ET)
    (* 12 alteration (- (* 7 (best-fifth ET)) 4)))

% Scale an explicit scale
#(define (scale-scale keysig ET)
    (map (lambda (note) (cons (car note) (scale-alteration (cdr note) ET)))
        keysig))

% Retune the standard scales (major/ionian has no alterations)
minor = #(scale-scale minor tuning)
locrian = #(scale-scale locrian tuning)
aeolian = #(scale-scale aeolian tuning)
mixolydian = #(scale-scale mixolydian tuning)
lydian = #(scale-scale lydian tuning)
phrygian = #(scale-scale phrygian tuning)
dorian = #(scale-scale dorian tuning)

% Set the innards
newglyphs = #(begin
    (retune-nominals tuning)
    (ly:parser-set-note-names (retuned-pitchnames pitchnames tuning))
    (retune-glyphs standard-alteration-glyph-name-alist tuning))

% Apply the new glyphs.
\layout {
  \context {
    \Score
    \override Accidental.#'alteration-glyph-name-alist = \newglyphs
    \override KeySignature.#'alteration-glyph-name-alist = \newglyphs
    \override AccidentalCautionary.#'alteration-glyph-name-alist = \newglyphs
    \override TrillPitchAccidental.#'alteration-glyph-name-alist = \newglyphs
    \override AmbitusAccidental.#'alteration-glyph-name-alist = \newglyphs
  }
}

% Null function to work with scordablature
retune = #(define-music-function (parser location ET midc m)
    (integer? integer? ly:music?) m)
