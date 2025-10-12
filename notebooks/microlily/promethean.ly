\include "sagittal.ly"

grid = 494

mixedLookup = \mixedPrometheanStrings
pureLookup = \purePrometheanStrings

\include "scheme-et.ly"
tuning = #184858
tunegrid = #(prime-et tuning)

% convert from divisions of a 200 cent whole tone
% to steps in the notation grid
tuningFactor = #(/ (ratio->steps (prime-et grid) 2187/2048)
                   (octaves->alteration (ratio->octaves tunegrid 2187/2048)))

#(define (quantize alteration)
    (let ((cents (* 1200 tuningFactor alteration)))
        (cond
               ((and (> cents -18.784861516) (< cents -17.321106073)) -75/10)
               ((and (< cents 18.784861516) (> cents 17.321106073)) 75/10)
               ((and (> cents -39.765356196) (< cents -38.061940350)) -163/10)
               ((and (< cents 39.765356196) (> cents 38.061940350)) 163/10)
               ((and (> cents -57.81833999) (< cents -55.866666067)) -235/10)
               ((and (< cents 57.81833999) (> cents 55.866666067)) 235/10)
               ((round (* sharpness grid))))))

#(define (entry-for-alteration strings alter)
    (cdr (assoc (quantize alter) strings)))

\include "sagji.ly"
