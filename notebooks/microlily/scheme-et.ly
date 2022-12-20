% Lilypond-escaped Scheme code for helping with equal temperaments
% and other bloat as deemed convenient

#(define primes '(2 3 5 7 11 13 17 19 23
                  29 31 37 41 43 47 53 59 61))

#(define (ratio->steps et ratio)
    (dotprod et (factorize ratio)))

#(define (ratio->octaves et ratio)
    (/ (ratio->steps et ratio) (car et)))

#(define (ratlist->octaves et ratlist)
         (/ (dotprod et ratlist) (car et)))

%
% Help for factorizing ratios
%

#(define (factorize ratio)
    (map - (factorize-int (numerator ratio))
           (factorize-int (denominator ratio))))

#(define (factorize-int n)
    (map (lambda (prime) (factors-of n prime)) primes))

#(define (factors-of n divisor)
    (if (= (modulo n divisor) 0)
        (+ 1 (factors-of (/ n divisor) divisor))
        0))

#(define (add-octaves n ratlist)
         (cons (+ n (car ratlist)) (cdr ratlist)))

%
% Making and altering Lilypond pitches
%

#(define (entry-for-rounded-alteration
          tuningFactor strings alter)
    (cdr (assoc
            (round (* alter tuningFactor))
            strings)))

#(define (entry-for-known-alteration
          tuning-et notation-et tuningFactor strings alter)
    (let ((ratlist (alteration->ratlist alter tuning-et)))
        (if ratlist (cdr (assoc (dotprod ratlist notation-et)
                                strings))
            ; fall back to rounding
            (begin
                (display "Unable to find ratiolist for ")
                (display alter)
                (newline)
                (entry-for-rounded-alteration tuningFactor strings alter)
                ))))

nominalRatios = #'(1/1 9/8 81/64 4/3 3/2 27/16 243/128)
etNominals = ##(0 2/12 4/12 5/12 7/12 9/12 11/12)
#(define (pythag-nominals ET)
    (list->vector (map
        (lambda (rat) (ratio->octaves ET rat))
        nominalRatios)))
nominalLists = #(list->vector (map factorize nominalRatios))

#(define tonic-pitch (ly:make-pitch -1 0 0))

#(define (tune-pitch ET ratio p)
    (let* ((relative-tuning (- (pitch->octaves p ET)
                               (pitch->octaves tonic-pitch ET)))
           (toniclist (pitch->ratlist tonic-pitch ET))
           (pitchlist (pitch->ratlist p ET))
           (ratlist (factorize ratio))
           (full-shift (- (ratlist->octaves ET ratlist) relative-tuning))
           (octave-correction (round full-shift))
           (oe-shift (- full-shift octave-correction))
           (alteration (ly:pitch-alteration p))
           (alt-list (alteration->ratlist alteration ET))
           (new-alteration (+ (octaves->alteration oe-shift)
                              alteration))
           (new-pitch (ly:make-pitch (ly:pitch-octave p)
                                     (ly:pitch-notename p)
                                     new-alteration))
           )
        (if (and toniclist pitchlist alt-list)
            ; All the reverse-lookups succeeded, so update the cache
            (let* ((relative-list (map - pitchlist toniclist))
                   (full-ratshift (map - ratlist relative-list))
                   (shift-list (add-octaves (- octave-correction)
                                            full-ratshift))
                   (new-alt-list (map - shift-list alt-list))
                   )
                (update-ratlist-cache! new-alteration new-alt-list)
                (shift-notename new-pitch new-alt-list ET)
                )
            new-pitch)))

#(define (shift-notename p alt-list ET)
    (let ((diatonic-shift (adjust-notename alt-list)))
        (if (= diatonic-shift 0)
            p
            (let* ((notename (+ (ly:pitch-notename p) diatonic-shift))
                   (old-octave (ly:pitch-octave p))
                   (old-notename (ly:pitch-notename p))
                   (shifted-notename (+ old-notename diatonic-shift))
                   (octave-shift (floor (/ shifted-notename 7)))
                   (new-notename (modulo shifted-notename 7))
                   (shift-list (add-octaves octave-shift
                                (map - (vector-ref nominalLists new-notename)
                                       (vector-ref nominalLists old-notename))))
                   (new-alt-list (map - alt-list shift-list))
                   (new-alteration (octaves->alteration
                                        (ratlist->octaves ET new-alt-list)))
                   )
                (update-ratlist-cache! new-alteration new-alt-list)
                (ly:make-pitch (+ old-octave octave-shift)
                               new-notename
                               new-alteration))
        )))

#(define (raise-by-ratio ET ratio p)
   (let* ((reg (ly:pitch-octave p))
          (degree (ly:pitch-notename p))
          (alteration (ly:pitch-alteration p))
          (altlist (alteration->ratlist alteration ET))
          (new-alt (+ alteration
                      (octaves->alteration (ratio->octaves ET ratio))))
          (new-pitch (ly:make-pitch reg degree new-alt))
         )
    (if altlist
        (let ((new-altlist (map + altlist (factorize ratio))))
            (update-ratlist-cache! new-alt new-altlist)
            (shift-notename new-pitch new-altlist ET)
            )
        new-pitch)))

#(define (pitch->octaves p ET)
    (let ((octave (ly:pitch-octave p))
          (degree (ly:pitch-notename p))
          (alter (ly:pitch-alteration p)))
        (+ octave
           (vector-ref tunedNominals degree)
           (alteration->octaves alter))))

#(define (pitch->ratlist p ET)
    (let* ((octave (ly:pitch-octave p))
           (degree (ly:pitch-notename p))
           (alter (ly:pitch-alteration p))
           (altlist (alteration->ratlist alter ET))
           )
        (if altlist
            (map + (add-octaves octave (vector-ref nominalLists degree))
                   altlist)
            #f)))

#(define (pitch12->octaves octave degree alter)
        (+ octave
           (vector-ref etNominals degree)
           (alteration->octaves alter)))

#(define (octaves->alteration oct) (* oct 6))
#(define (alteration->octaves alter) (/ alter 6))

%
% Manage the cache of known ratio lists
%

#(define ratlist-cache #f)

#(define (alteration->ratlist alteration ET)
    (begin (if (not ratlist-cache) (init-ratlist-cache ET))
        (let ((lookup (assoc alteration ratlist-cache)))
            (if lookup (cdr lookup) #f))))

#(define (init-ratlist-cache ET)
    (set! ratlist-cache
        (let ((apotome (factorize 2187/2048)))
            (map (lambda (multiple)
                (let ((ratlist (map (lambda (x) (* multiple x)) apotome)))
                (cons (octaves->alteration
                        (ratlist->octaves ET ratlist))
                      ratlist)))
                '(0 1 -1 2 -2)))))

#(define (update-ratlist-cache! alteration ratlist)
    (if (not (assoc alteration ratlist-cache))
        (set! ratlist-cache
            (cons (cons alteration ratlist) ratlist-cache))))

%
% Help for finding ET mappings
%

% Generalized patent val
#(define (prime-et n)
    (map (lambda (p) (nint (* n (/ (log p)
                                   (log 2)))))
         primes))

#(define (dotprod a b)
    (apply + (map * a b)))

#(define (nint x) (inexact->exact (round x)))
