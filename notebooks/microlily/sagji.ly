% Code for Sagittal notations with just intonation

\include "regular.ly"

%
% JI support
%

% This uses equal temperaments because notation is more consistent
% when snapped to an equal temperament and the Lilypond tuning
% model means everything is equally tempered.
% The trick is to find sufficiently large equal temperaments
% that the inaccuracy doesn't matter.

tunedNominals = #(pythag-nominals tunegrid)

% convert from divisions of a 200 cent whole tone
% to steps in the notation grid
tuningFactor = #(/ (ratio->steps (prime-et grid) 2187/2048)
                   (octaves->alteration (ratio->octaves tunegrid 2187/2048)))

\include "scheme-music.ly"

%
% Set the accidentals
%

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
