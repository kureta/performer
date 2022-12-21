\version "2.24"

#(ly:set-option 'relative-includes #t)

\include "./event-listener.ly"
\include "./instruments.ily"

\header {
  title = "Title?"
  subtitle = "Study for an Ensemble of Fictional Flutes"
  composer = "Sahin Kureta"
  tagline = #f
}

\paper {
  paper-width = 400
  paper-height = 225
}

\score {
  \layout{
    \context {
      \Score
      proportionalNotationDuration = #(ly:make-moment 1/8)
      \enablePolymeter
    }
    \context {
      \Staff
      \remove "Instrument_name_engraver"
      \RemoveEmptyStaves
      \override VerticalAxisGroup.remove-first = ##t
      \override VerticalAxisGroup.staff-staff-spacing = #'((basic-distance . 17))
      \numericTimeSignature
    }
  }
  \new StaffGroup {
    <<
      \new Staff \with {instrumentName = "Flute I" } {
        \tempo 4 = 90
        \time 5/4
        \clef treble
        \intro_flute_one
      }
      \new Staff \with {instrumentName = "Flute II" } {
        \tempo 4 = 90
        \time 5/4
        \clef treble
        \intro_flute_two
      }
      \new Staff \with {instrumentName = "Flute III" } {
        \tempo 4 = 90
        \time 4/4
        \repeat unfold 5 \canon_one
	\tune 3 b'8 \tune 2 e'8 \p
      }
      \new Staff \with {instrumentName = "Flute IV" } {
        \tempo 4 = 90
        \time 4/4
        \repeat unfold 5 \canon_two
	\tune 12 b'4 \p
      }

    >>
  }
}
