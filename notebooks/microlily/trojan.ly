% Code for Sagittal Trojan notations
tuning = #184858

\include "scheme-et.ly"

%
% Retune the pitch names (similar to regular.ly)
%

% Keep the default scale in 12-equal
#(ly:set-default-scale (ly:make-scale etNominals))

tunegrid = #(prime-et tuning)
tunedNominals = #etNominals

\include "scheme-music.ly"

#(define (retuned-pitchnames pitchnames ET)
    (let ((apotome-size (* 6 (ratio->octaves ET 2187/2048)))
          (fifth-correction (- (* 6 (ratio->octaves ET 3/2)) 7/2))
          (by-fifths #(0 2 4 -1 1 3 5)))
        (map (lambda (pitchname)
                (let* ((pitch (cdr pitchname))
                       (notename (ly:pitch-notename pitch))
                       (correction (* fifth-correction
                                      (vector-ref by-fifths notename)))
                       (old-alteration (ly:pitch-alteration pitch))
                       (alteration (+ correction
                                      (* old-alteration apotome-size 2))))
                    (cons (car pitchname)
                        (ly:make-pitch (ly:pitch-octave pitch)
                                       notename alteration))))
            pitchnames)))

#(ly:parser-set-note-names parser (retuned-pitchnames pitchnames tunegrid))

%
% Set the accidentals
%

#(define (entry-for-alteration strings alter)
         (entry-for-rounded-alteration
                (alteration->octaves grid) strings alter))

pureStrings = #(map (lambda (entry)
            (cons (car entry) (sagittalString (cdr entry))))
            pureLookup)

pureXExtents = #(map (lambda (entry)
            (cons (car entry) (sagittalXExtent (cdr entry))))
            pureLookup)

pureYExtents = #(map (lambda (entry)
            (cons (car entry) (sagittalYExtent (cdr entry))))
            pureLookup)

mixedStrings = #(map (lambda (entry)
            (cons (car entry) (sagittalString (cdr entry))))
            mixedLookup)

mixedXExtents = #(map (lambda (entry)
            (cons (car entry) (sagittalXExtent (cdr entry))))
            mixedLookup)

mixedYExtents = #(map (lambda (entry)
            (cons (car entry) (sagittalYExtent (cdr entry))))
            mixedLookup)

pure = {
    \override Staff.Accidental #'stencil = #ly:text-interface::print
    \override Staff.Accidental #'text = #(lambda (grob)
        (entry-for-alteration pureStrings
            (ly:grob-property grob 'alteration)))
    \override Staff.Accidental #'font-name = #"Sagittal"
    \override Staff.Accidental #'font-size = #4
    % c.f. lilypond-user 2014-03-06 or
    % http://code.google.com/p/lilypond/issues/detail?id=2811
    \override Staff.Accidental #'horizontal-skylines = #'()
    \override Staff.Accidental #'X-extent = #(lambda (grob)
        (entry-for-alteration pureXExtents
            (ly:grob-property grob 'alteration)))
    \override Staff.Accidental #'Y-extent = #(lambda (grob)
        (entry-for-alteration pureYExtents
            (ly:grob-property grob 'alteration)))
}

mixed = {
    \override Staff.Accidental #'stencil = #ly:text-interface::print
    \override Staff.Accidental #'text = #(lambda (grob)
        (entry-for-alteration mixedStrings
            (ly:grob-property grob 'alteration)))
    \override Staff.Accidental #'font-name = #"Sagittal"
    \override Staff.Accidental #'font-size = #4
    % c.f. lilypond-user 2014-03-06 or
    % http://code.google.com/p/lilypond/issues/detail?id=2811
    \override Staff.Accidental #'horizontal-skylines = #'()
    \override Staff.Accidental #'X-extent = #(lambda (grob)
        (entry-for-alteration mixedXExtents
            (ly:grob-property grob 'alteration)))
    \override Staff.Accidental #'Y-extent = #(lambda (grob)
        (entry-for-alteration mixedYExtents
            (ly:grob-property grob 'alteration)))
}
