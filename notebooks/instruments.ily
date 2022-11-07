\version "2.22.2"

\include "microlily/he.ly"

\language "english"

\header {
  title = "Gradient Descent"
  subtitle = "Study for an Ensemble of Fictional Flutes"
  composer = "Sahin Kureta"
  tagline = #f
}

\paper {
  paper-width = 240
  paper-height = 135
}

global = {
  \numericTimeSignature
}

intro_flute_one = {
  \global
  \time 5/4 \tempo 4 = 90 \tonic e
  \clef treble
  \tuplet 3/2 { \tune 2 e'8 ( [ \tune 3 b'-. ) r ] } r1 |
  \tuplet 3/2 { \tune 5 g''8 ( [ \tune 19 g''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 5 g''8 ( [ \tune 11 a''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 13 c'''8 ( [ \tune 5 g''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 19 g''8 ( [ \tune 13 c'''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 19 g''8 ( [ \tune 17 f''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 19 g''8 ( [ \tune 17 f''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 5 g''8 ( [ \tune 13 c'''8-. ) r ] } r1 |
}

intro_flute_two = {
  \global
  \time 5/4 \tempo 4 = 90 \tonic e
  \clef treble
  \tuplet 3/2 { \tune 19 g''8 ( [ \tune 13 c'''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 11 a''8 ( [ \tune 7 d'''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 19 g''8 ( [ \tune 7 d'''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 7 d'''8 ( [ \tune 2 e'-. ) r ] } r1 |
  \tuplet 3/2 { \tune 3 b'8 ( [ \tune 2 e'-. ) r ] } r1 |
  \tuplet 3/2 { \tune 7 d'''8 ( [ \tune 3 b'-. ) r ] } r1 |
  \tuplet 3/2 { \tune 11 a''8 ( [ \tune 13 c'''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 7 d'''8 ( [ \tune 2 e'-. ) r ] } r1 |
}
