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
        \intro_flute_one
	\time 4/4
	\tune 7 d'2.. ( \pp\< \tune 7 d'8 \f )
      }
      \new Staff \with {instrumentName = "Flute II" } {
        \tempo 4 = 90
        \time 5/4
        \clef treble
        \intro_flute_two
        \intro_flute_two
	\time 4/4
	\tune 5 g2.. ( \pp\< \tune 5 g 8 \f )
      }
      \new Staff \with {instrumentName = "Flute III" } {
        \tempo 4 = 90
        \time 4/4
        \repeat unfold 10 s1
	\time 4/4
	\repeat unfold 5 \canon_one
	\tune 2 e'2.. ( \pp\< \tune 2 e'8 \f )
      }
      \new Staff \with {instrumentName = "Flute IV" } {
        \tempo 4 = 90
        \time 4/4
        \repeat unfold 10 s1
	\time 4/4
	\repeat unfold 5 \canon_two
	\tune 11 a''2.. ( \pp\< \tune 11 a''8 \f )
      }
    >>
  }
}
