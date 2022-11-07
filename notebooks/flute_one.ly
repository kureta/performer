\version "2.22.2"

\include "./event-listener.ly"
\include "./instruments.ily"

\score {
  <<
    \new Staff \with {instrumentName = "Flute I" } {\mixed \intro_flute_one}
  >>
  \layout {
    \context {
      \Score
      proportionalNotationDuration = #(ly:make-moment 1/3)
    }
  }
  \midi{}
}
