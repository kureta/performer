\version "2.16.0"
\include "microlily/he.ly"
\include "./event-listener.ly"

\header {
    title = "The Harmonic Series"
    tagline = "Music engraving by Graham Breed using Lilypond (see http://x31eq.com/lilypond/)"
    }
\language "english"

flute_one = { \time 5/4 \tempo 4 = 90 \tonic e
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

flute_two = { \time 5/4 \tempo 4 = 90 \tonic e
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

\score {
    <<
    \new Staff {\mixed \centsUp \flute_one}
    \new Staff {\mixed \centsUp \flute_two}
    >>
    \layout{}
    \midi{}
}
