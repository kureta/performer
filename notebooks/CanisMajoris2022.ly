\version "2.22.2"

\include "./event-listener.ly"
\include "./instruments.ily"

\score {
  <<
    \new Staff \with {instrumentName = "Flute I" } {\mixed \intro_flute_one}
    \new Staff \with {instrumentName = "Flute II" } {\mixed \intro_flute_two}
  >>
  \layout{}
  \midi{}
}
