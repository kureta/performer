\version "2.24"

\include "./microlily/he.ly"

\language "english"

global = {
  \numericTimeSignature
}

intro_flute_one = {
  \global
  \tonic e
  \mixed
  \tuplet 3/2 { \tune 2 e'8-> \f ( [ \tune 3 b'-. ) r ] } r1 |
  \tuplet 3/2 { \tune 5 g''8-> ( [ \tune 19 g''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 5 g''8-> ( [ \tune 11 a''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 13 c'''8-> ( [ \tune 5 g''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 19 g''8-> ( [ \tune 13 c'''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 19 g''8-> ( [ \tune 17 f''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 19 g''8-> ( [ \tune 17 f''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 5 g''8-> ( [ \tune 13 c'''8-. ) r ] } r1 |
}

intro_flute_two = {
  \global
  \mixed
  \tonic e
  \tuplet 3/2 { \tune 19 g''8-> \f ( [ \tune 13 c'''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 11 a''8-> ( [ \tune 7 d'''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 19 g''8-> ( [ \tune 7 d'''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 7 d'''8-> ( [ \tune 2 e'-. ) r ] } r1 |
  \tuplet 3/2 { \tune 3 b'8-> ( [ \tune 2 e'-. ) r ] } r1 |
  \tuplet 3/2 { \tune 7 d'''8-> ( [ \tune 3 b'-. ) r ] } r1 |
  \tuplet 3/2 { \tune 11 a''8-> ( [ \tune 13 c'''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 7 d'''8-> ( [ \tune 2 e'-. ) r ] } r1 |
}

canon_one = {
  \global
  \tonic e
  \mixed
  \shape #'((-0.1 . 0.1) (0 . 2) (0 . 2) (0 . 0)) Slur
  \tune 5 g''4 \f\> (
  \tune 7 d'''4~ \tune 7 d'''4 \p )
  \shape #'((-0.1 . 0.1) (0 . 2) (0 . 2) (0 . 0)) Slur
  \tune 2 e'8-> ( \tune 5 g''8 )
  \shape #'((-0.1 . 0.1) (0 . 2) (0 . 2) (0 . 0)) Slur
  \tune 2 e'8-> \< ( \tune 3 b'8
  \tune 5 g''8 \f ) \tune 3 b'8~ \> ( \tune 3 b'8
  \tune 2 e'8
  \tune 3 b'8 \tune 2 e'8 \p )
}

canon_two = {
  \global
  \tonic e
  \mixed
  \tune 17 f''8-> \f \tune 13 c'''8 ( \p \<
  \tune 19 g''8 \tune 17 f''8 \f )
  \tune 13 c'''8 \> ( \tune 19 g''8
  \tune 17 f''8 ) \tune 19 g''8~ \p ( \tune 19 g''8
  \tune 11 a''8 -> \<
  \tune 13 c'''4 )
  \tune 11 a''8 ( \tune 17 f''8->
  \tune 11 a''4 \f )
}

harmon = {
  \global
  \tonic e
  \mixed

  \clef bass
  \ottava #-1
  \set Staff.ottavation = #"8vb"
  \tune 1 e,,4 \fff \> \tune 2 e,4 \tune 3 b,4 \tune 4 e4
  \ottava #0
  \tune 5 e4 \tune 6 e'4 \tune 7 e'4 \tune 8 e'4
  \clef treble
  \tune 9 f'4 \tune 10 g'4 \tune 11 a'4 \tune 12 b'4
}
