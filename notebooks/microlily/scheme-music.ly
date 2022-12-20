% Scheme music functions

% Uses tunegrid: the equal temperament mapping for tuning

% Apply a function to all pitches.
% This is copied from "Transposing music with minimum accidentals"
% In the "Pitches" snippets in the documentation.
#(define (apply-to-pitches fn music)
  (let ((es (ly:music-property music 'elements))
        (e (ly:music-property music 'element))
        (p (ly:music-property music 'pitch)))
    (if (pair? es)
       (ly:music-set-property!
         music 'elements
         (map (lambda (x) (apply-to-pitches fn x)) es)))
    (if (ly:music? e)
       (ly:music-set-property!
         music 'element
         (apply-to-pitches fn e)))
    (if (ly:pitch? p)
       (begin
         (set! p (fn p))
         (ly:music-set-property! music 'pitch p)))
    music))

tonic = #(define-music-function (parser location m) (ly:music?)
    (begin
        (apply-to-pitches (lambda (p) (begin (set! tonic-pitch p) p))
            (ly:music-deep-copy m))
        (make-music 'SequentialMusic 'void #t)))

% Set the pitch(es) to the given ratio relative to the tonic
tune = #(define-music-function (parser location ratio m)
  (rational? ly:music?)
  (apply-to-pitches (lambda (p) (tune-pitch tunegrid ratio p))
      (ly:music-deep-copy m)))

% Shift pitch(es) by the desired ratio
raise = #(define-music-function (parser location ratio m)
  (rational? ly:music?)
  (apply-to-pitches (lambda (p) (raise-by-ratio tunegrid ratio p))
      (ly:music-deep-copy m)))

lower = #(define-music-function (parser location ratio m)
  (rational? ly:music?)
  (apply-to-pitches (lambda (p) (raise-by-ratio tunegrid (/ ratio) p))
      (ly:music-deep-copy m)))

%
% Cents display
%

centsUp = #(define-music-function (parser location m) (ly:music?)
    (add-cents-to-music 1 m))

centsDown = #(define-music-function (parser location m) (ly:music?)
    (add-cents-to-music -1 m))

% Return the cents deviations from 12-equal for a list of pitches.
#(define (cents-for-chord es)
    (if (null? es) '()
        (let* ((p (ly:music-property (car es) 'pitch))
               (others (cents-for-chord (cdr es)))
               (octave (ly:pitch-octave p))
               (degree (ly:pitch-notename p))
               (alter (ly:pitch-alteration p))
               ; ET alteration corresponding to the conventional part
               ; of mixed sagittal
               (etalter (et-alteration alter))
               (altcents (* 1200 (- (pitch->octaves p tunegrid)
                                    (pitch12->octaves octave degree etalter))))
               )
            (cons altcents others))))

#(define (add-cents-to-music direction m)
    (let ((es (ly:music-property m 'elements))
          (e  (ly:music-property m 'element)))
            (if (pair? es)
                (ly:music-set-property! m 'elements
                    (if (eq? (ly:music-property m 'name) 'EventChord)
                        (add-cents-to-chord direction es)
                        (map (lambda (more)
                                (add-cents-to-music direction more))
                             es))))
            (if (ly:music? e)
                (ly:music-set-property! m 'element
                    (add-cents-to-music direction e)))
        m))

% Add cents markup to a list including pitches
#(define (add-cents-to-chord direction es)
    (cons (make-music 'TextScriptEvent 'direction direction 'text
            (markup
                #:tiny
                (#:override '(baseline-skip . 1.5)
                    (make-center-column-markup
                        (map format-cents (cents-for-chord
                            (sort
                                (filter (lambda (e) (ly:pitch?
                                        (ly:music-property e 'pitch)))
                                    es)
                                pitch-less)))))))
            es))

% Format a number of cents (string of nearest integer)
#(define (format-cents c)
    (string-append (if (< c 0) "" "+")
        (number->string (inexact->exact (round c)))))

% Decide which pitch is lesser for purposes of sorting.
% Takes staff position first (to match visual appearance)
% then alteration as a tie breaker.
#(define (pitch-less ea eb)
    (let* ((pa (ly:music-property ea 'pitch))
           (pb (ly:music-property eb 'pitch))
           (steps-diff (- (ly:pitch-steps pa) (ly:pitch-steps pb))))
        (cond ((< 0 steps-diff) #t)
              ((> 0 steps-diff) #f)
              (<= (ly:pitch-alteration pa) (ly:pitch-alteration pb)))))
