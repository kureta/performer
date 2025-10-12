\include "sagittal.ly"

grid = 12

mixedLookup = #'((0 natural) (1 sharp) (-1 flat)
    (2 doublesharp) (-2 doubleflat))
pureLookup = #'((0 natural) (1 doublebarbdoubleup) (-1 doublebarbdoubledown)
    (2 doublebarbexup) (-2 doublebarbexdown))

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
