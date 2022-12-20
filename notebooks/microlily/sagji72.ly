\include "sagittal.ly"
grid = 72

mixedLookup = \mixedSextulaStrings
pureLookup = \pureSextulaStrings

\include "scheme-et.ly"
tuning = #184858
tunegrid = #(prime-et tuning)
notegrid = #(prime-et grid)

% convert from divisions of a 200 cent whole tone
% to steps in the notation grid
tuningFactor = #(/ (ratio->steps (prime-et grid) 2187/2048)
                   (octaves->alteration (ratio->octaves tunegrid 2187/2048)))

#(define (entry-for-alteration strings alter)
         (entry-for-known-alteration tunegrid notegrid
                                     tuningFactor strings alter))

\include "sagji.ly"
