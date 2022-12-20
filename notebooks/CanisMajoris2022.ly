\version "2.22.2"

\include "./event-listener.ly"
\include "./instruments.ily"

\header {
  title = "Title?"
  subtitle = "Study for an Ensemble of Fictional Flutes"
  composer = "Sahin Kureta"
  tagline = #f
}

\paper {
  paper-width = 240
  paper-height = 135
}

\score {
   \layout{
      \context {
        \Score
        proportionalNotationDuration = #(ly:make-moment 1/2)
    }
  }
  <<
    \new Staff \with {instrumentName = "Flute I" } {\mixed \intro_flute_one}
    \new Staff \with {instrumentName = "Flute II" } {\mixed \intro_flute_two}
  >>
}
