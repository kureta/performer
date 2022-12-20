\version "2.16.0"
\include "sagji224.ly"

\header {
    title = "A Musical Example in Sagittal Notation"
    composer = "Keenan & Secor"
    tagline = "Music engraving by Graham Breed using Lilypond (see http://x31eq.com/lilypond/)"
    }

anchor = {\once \override NoteColumn #'force-hshift =#0 }
nudge = {\once \override NoteColumn #'force-hshift =#1.0 }

content = \transpose c c' { \tonic c
    <c \tune #5/4 e \tune #3/2 g>2 <c f \tune #5/3 a> |
    <c \tune #5/4 e g \tune #7/4 b> <c \tune #6/5 e g \tune #9/5 b> |
    <<
    { \stemUp \tune #16/9 b4 \tune #8/5 a }
    \\
    \stemDown <c f>2
    >>
    <c \tune #5/4 e g>2 |
    <d \tune #11/8 f \tune #7/4 b \tune #15/8 b>4 ~
    <d \tune #11/8 f \tune #13/8 a \tune #15/8 b>4
    <c \tune #5/4 e g c'> <c \tune #7/6 e g \tune #11/6 b> |
    <<
        { \stemUp \tune #117/64 b4 \tune #27/16 a }
    \\
    \stemDown <d \tune #45/32 f>2
    >>
    <g, \raise #64/63 f g \tune #15/8 b>2 |
    <<
    {\tieUp \anchor <g c'>1 ~ | <\tune #5/4 e g c'>1 }
    \\
    {\stemUp \nudge \tune #21/16 f2 \tune #5/4 e4 d | s1 }
    \\
    {\tieDown \anchor c1 ~ | \anchor c1 }
    >> \bar "||"
}

\score {
    <<
        \new Staff \centsUp {\mixed \content}
        \new Staff \centsDown {\pure \content}
    >>
    \layout{}
    \midi{
        \context {
            \Score
            tempoWholesPerMinute = #(ly:make-moment 180 4)
        }
    }
}
