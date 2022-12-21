\version "2.24"

#(ly:set-option 'relative-includes #t)

\include "./event-listener.ly"
\include "./instruments.ily"

\header {
  title = "Title?"
  subtitle = "Study for an Ensemble of Fictional Flutes"
  composer = "Şahin Kureta"
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
	\break
	\intro_flute_one
      }
      \new Staff \with {instrumentName = "Flute II" } {
        \tempo 4 = 90
        \time 5/4
        \clef treble
        \intro_flute_two
	\break
        \intro_flute_two
      }
      \new Staff \with {instrumentName = "Flute IV" } {
        \tempo 4 = 90
        \time 5/4
        \clef treble
	\repeat unfold 10 s1
	\break
	\flute_three_b
	\three_b_fast
	\break
	\flute_three_b_accent
	\three_b_slow
      }
      \new Staff \with {instrumentName = "Flute V" } {
        \tempo 4 = 90
        \time 5/4
        \clef treble
	\repeat unfold 10 s1
	\break
	\flute_four_b
	\four_b_fast
	\break
	\flute_four_b_accent
	\four_b_slow
      }
      \new Staff \with {instrumentName = "Flute III" } {
        \tempo 4 = 90
        \time 5/4
        \clef bass
	\repeat unfold 2\intro_flute_three_drone
	\break
	\intro_flute_three_drone
	\time 4/4
	\intro_flute_three_drone_b
      }
    >>
  }
}
