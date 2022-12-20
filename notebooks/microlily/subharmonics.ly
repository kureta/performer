\version "2.16.0"
\include "he.ly"

\header {
    title = "Subharmonics"
    tagline = "Music engraving by Graham Breed using Lilypond (see http://x31eq.com/lilypond/)"
    }

treble = { \clef "treble" \tonic a
    \transpose c c'' {
        a1 \tune #1/33 a \tune #1/17 g \tune #1/35 g \tune #1/9 g
            \tune #1/37 g \tune #1/19 f \tune #1/39 f \tune #1/5 f
        \tune #1/41 f \tune #1/21 e \tune #1/43 e \tune #1/11 e
            \tune #1/45 e \tune #1/23 e \tune #1/47 e d
        \tune #1/49 c \tune #1/25 d \tune #1/51 c \tune #1/13 c
            \tune #1/53 c \tune #1/27 c \tune #1/55 c \tune #1/7 b,
    }
    \transpose c c' {
        \tune #1/57 b1 \tune #1/29 b \tune #1/59 b \tune #1/15 b
            \tune #1/61 b \tune #1/31 a \tune #1/63 a a
    }

}
\score {
    <<
        \new Staff \centsUp {
            #(set-accidental-style 'dodecaphonic)
            \mixed \treble
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
