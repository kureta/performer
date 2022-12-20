% Helmholtz-Ellis style notations, with a distinct accidental for each prime.
% For now, Extended Helmholtz-Ellis Just Intonation Pitch Notation.
% Requires the HE font to be installed.  See www.plainsound.org

% First column: code of character of lowering accidental
%               or 's for special case
% Second column: code of character of raising accidental
% Remaining columns: mapping from JI to the raising interval
#(define ehejimap '(
    (  s   d 7 11 16 20 24 26 29 30 31 34 35 36 37 38 38 40 41 41) ; limmas
    (  s   c 5  8 12 14 17 19 20 21 23 24 25 26 27 27 28 29 29 30) ; apotomes
    (  s   s 0  0 -1  0  0  0  1  0  0  1  0  0  0  0  0 -1  0 -1)
    ( 60  62 0  0  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0)
    ( 53  52 0  0  0  0  1  0  0  0  0  0 -1  1  0  0  0  0  0  0)
    ( 48  57 0  0  0  0  0 -1  0  0  0  0  0  0  0  0  0  0  0  0)
    ( 58  59 0  0  0  0  0  0 -1  0  0  0  0  0  0  0  0  0  0  0)
    ( 92  47 0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0)
    ( 54  51 0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0)
    ( 55  50 0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0)
    ( 45  43 0  0  0  0  0  0  0  0  0  0 -1  0  0  0  0  0  0  0)
    (178 179 0  0  0  0  0  0  0  0  0  0  0 -1  0  0  0  0  0  0)
    ( 38  33 0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0) ; approx
    (255 183 0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0)
    (189 191 0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0)
    (176 177 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 -1  0  0)
    (251 248 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0)
    (186 187 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1)
))

\include "scheme-et.ly"
tuning = #184858
tunegrid = #(prime-et tuning)
tunedNominals = #(pythag-nominals tunegrid)
\include "regular.ly"
\include "scheme-music.ly"

#(define (string-for-alteration alter)
    (apply string-append
        (map ly:wide-char->utf-8
             (reverse (chars-for-ratlist ehejimap
                                (alteration->ratlist alter tunegrid))))))

% The corresponding alteration for equal temperament
% (used for calculating cents offsets)
#(define (et-alteration alter)
    (/ (dotprod (cddadr ehejimap)
                (alteration->ratlist alter tunegrid))
       2))

% The amount by which the notename is wrong
#(define (adjust-notename ratlist)
    (dotprod (cddar ehejimap) ratlist))

#(define (chars-for-ratlist mapping ratlist)
    (if (null? mapping)
        '()
        (let* ((lower-glyph (caar mapping))
               (raise-glyph (cadar mapping))
               (steps (dotprod ratlist (cddar mapping)))
               (more (chars-for-ratlist (cdr mapping) ratlist))
               )
            (cond ((eqv? raise-glyph 'd) (diatonic-chars steps more))
                  ((eqv? raise-glyph 'c) (chromatic-chars steps more))
                  ((eqv? raise-glyph 's) (syntonic-chars steps more))
                  (else
                    (cond ((and (> steps 1) (= raise-glyph 62))
                            ; special 7-limit case
                            (cons 46 (rep-cons (- steps 2) 62 more)))
                          ((and (< steps -1) (= lower-glyph 60))
                            ; special 7-limit case
                            (cons 44 (rep-cons (- (- steps) 2) 60 more)))
                          ((< steps 0)
                            (rep-cons (- steps) lower-glyph more))
                          (else
                            (rep-cons steps raise-glyph more))))
                  ))))

#(define (diatonic-chars steps more)
    (if (= steps 0)
        more
        (begin
            (display "Alteration assigned to wrong degree.")
            (newline)
            '()
        )))

#(define (chromatic-chars steps more)
    ; chromatic steps
    ; Only double flats to double sharps supported
    (let* ((symbol (cdr (assoc steps '((0 . 110)
                                       (-1 . 101) (-2 . 69)
                                       (1 . 118) (2 . 86)))))
           ; Now deal with syntonic commas
           (full-symbol (+ symbol (car more)))
           )
        (if (and (not (null? (cdr more))) (= full-symbol 110))
            ; suppress a simple natural
            ; when other symbols apply
            (cdr more)
            (cons full-symbol (cdr more))
        )))

#(define (syntonic-chars steps more)
    ; syntonic comma steps
    ; return an integer to alter the previous accidental for a +/- 3 range
    ; then the other symbols
    (cond
        ((> steps 3)
            ; add another glyph for a +/-6 range
            (cons 3 (cons (vector-ref #(33 34 35) (- steps 4))
                          more)))
        ((< steps -3)
            ; add another glyph for a +/-6 range
            (cons -3 (cons (vector-ref #(38 37 36) (- (- steps) 4))
                           more)))
        (else
            (cons steps more))))

% repeated-cons for building strings of integers
#(define (rep-cons n symbol more)
         (if (= n 0)
             more
             (cons symbol (rep-cons (- n 1) symbol more))))


mixed = {
    \override Staff.Accidental #'stencil = #ly:text-interface::print
    \override Staff.Accidental #'text = #(lambda (grob)
        (string-for-alteration
            (ly:grob-property grob 'alteration)))
    \override Staff.Accidental #'font-name = #"HE"
    \override Staff.Accidental #'font-size = #-1
    % c.f. lilypond-user 2014-03-06 or
    % http://code.google.com/p/lilypond/issues/detail?id=2811
    \override Staff.Accidental #'horizontal-skylines = #'()
    \override Staff.Accidental #'X-extent = #(lambda (grob)
        (let ((acc-str (string-for-alteration
                        (ly:grob-property grob 'alteration))))
                (cons (* -0.3 (string-length acc-str))
                      (* 1.4 (string-length acc-str)))))
    \override Staff.Accidental #'Y-extent = #'(-1.0 . 1.0)
}

pure = \mixed
