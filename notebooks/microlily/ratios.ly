\version "2.16.0"
\include "athenian.ly"

\header {
    title = "Sagittal Athenian Notation for Just Intonation"
    composer = "Keenan & Secor"
    tagline = "Music engraving by Graham Breed using Lilypond (see http://x31eq.com/lilypond/)"
    }

content = \transpose c c' { \time 2/4 \tonic c
    \override Staff.TimeSignature #'stencil = #'()
    \tune #1/1 c2 \tune #3/2 g \tune #4/3 f \bar "||"
    \tune #5/3 a \tune #5/4 e \tune #6/5 e \tune #8/5 a \bar "||"
    \tune #7/4 b \tune #7/5 g \tune #7/6 e \tune #8/7 d
    \tune #10/7 f \tune #12/7 a \bar "||"
    \tune #9/5 b \tune #9/7 e \tune #9/8 d \tune #10/9 d
    \tune #14/9 a \tune #16/9 b \bar "||"
    \tune #11/6 b \tune #11/7 { a4 g } \tune #11/8 f2
    \tune #11/9 e \tune #11/10 d \tune #12/11 d
    \tune #14/11 { e4 f } \tune #16/11 g2 \tune #18/11 a
    \tune #20/11 b \bar "||"
    \tune #13/7 b \tune #13/8 a \tune #13/9 g \tune #13/10 f
    \tune #13/11 e \tune #13/12 d \tune #14/13 d \tune #16/13 e
    \tune #18/13 f \tune #20/13 g \tune #22/13 a \tune #24/13 b \bar "||"
    \tune #15/8 b \tune #15/11 f \tune #15/13 d \tune #15/14 c
    \tune #16/15 d \tune #22/15 g \tune #26/15 b \tune #28/15 c' \bar "||"
    \tune #17/16 d \tune #32/17 b \tune #25/16 { g4 a } \bar "||"
}

\score {
    <<
    \new Staff {\mixed \centsUp \content}
    \new Staff {\new Voice = "pure" \pure \content}
    \new Lyrics \lyricmode {
        \set associatedVoice = #"pure"
        "1/1"2 "3/2"2 "4/3"2 "5/3"2 "5/4"2 "6/5"2 "8/5"2
        "7/4"2 "7/5"2 "7/6"2 "8/7"2 "10/7"2 "12/7"2
        "9/5"2 "9/7"2 "9/8"2 "10/9"2 "14/9"2 "16/9"2
        "11/6"2 "11/7"2 "11/8"2 "11/9"2 "11/10"2 "12/11"2 "14/11"2
        "16/11"2 "18/11"2 "20/11"2
        "13/7"2 "13/8"2 "13/9"2 "13/10"2 "13/11"2 "13/12"2 "14/13"2
        "16/13"2 "18/13"2 "20/13"2 "22/13"2 "24/13"2
        "15/8"2 "15/11"2 "15/13"2 "15/14"2 "16/15"2 "22/15"2 "26/15"2 "28/15"2
        "17/16"2 "32/17"2 "25/16"2
        }
    >>
    \layout{}
    \midi{}
}
