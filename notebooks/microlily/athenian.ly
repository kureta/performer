\include "sagittal.ly"

grid = 224

mixedLookup = \mixedAthenianStrings
pureLookup = \pureAthenianStrings

\include "scheme-et.ly"
tuning = #184858
tunegrid = #(prime-et tuning)

% convert from divisions of a 200 cent whole tone
% to steps in the notation grid
tuningFactor = #(/ (ratio->steps (prime-et grid) 2187/2048)
                   (octaves->alteration (ratio->octaves tunegrid 2187/2048)))

#(define (entry-for-alteration strings alter)
         (entry-for-rounded-alteration tuningFactor strings alter))

\include "sagji.ly"
