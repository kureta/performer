\version "2.16.0"
\include "microlily/he.ly"

\header {
    title = "The Harmonic Series"
    tagline = "Music engraving by Graham Breed using Lilypond (see http://x31eq.com/lilypond/)"
    }

bass = { \clef "bass" \tonic a
    \transpose a a, { a,,1 a, e a }
    \tune 5 c e \tune 7 g a }
treble = { \clef "treble" \tonic a {
    \transpose c c' {
        b,1 \tune 5 c \tune 11 d e \tune 13 f \tune 14 g \tune 15 g a
        \tune 17 b b
    }
    \transpose c c'' {
        \tune 19 c \tune 5 c \tune 21 d \tune 11 d \tune 23 d e
        \tune 25 e \tune 26 f \tune 27 f \tune 28 g \tune 29 g
            \tune 30 g \tune 31 g a
        \tune 33 a \tune 17 b \tune 35 b b \tune 37 b
    }
    \transpose c c''' {
        \tune 38 c \tune 39 c \tune 40 c
        \tune 41 c \tune 42 d \tune 43 d \tune 44 d \tune 45 d
            \tune 46 d \tune 47 d e
        \tune 49 f \tune 50 e \tune 51 f \tune 52 f \tune 53 f
            \tune 54 f \tune 55 f \tune 56 g
        \tune 57 g \tune 58 g \tune 59 g \tune 60 g \tune 61 g
            \tune 62 a \tune 63 a a
    }
}}
\score {
    <<
        \override Score.BarNumber #'break-visibility = #'#(#t #t #t)
        \new Staff \centsUp {
            #(set-accidental-style 'dodecaphonic)
            \mixed \bass \mixed \treble
        }

    >>
    \layout{}
    \midi{
        \context {
            \Score
            tempoWholesPerMinute = #(ly:make-moment 180 4)
        }
    }
}
