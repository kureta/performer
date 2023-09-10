\version "2.24"

\include "oll-core/package.ily"
\loadPackage lilypond-export

#(ly:set-option 'relative-includes #t)

\include "./event-listener.ly"
\include "./instruments.ily"

\header {
  title = "Study No. 2 for AI Ensemble (2022)"
  subtitle = "Piece for a Group of Imaginary Flutes"
  composer = "Şahin Kureta"
  tagline = ##f
}

\paper {
  paper-width = 400
  paper-height = 225
}

music = \new StaffGroup {
  <<
    \new Staff \with {instrumentName = "Flute I" } {
      \tempo 4 = 90
      \time 5/4
      \clef treble
      \intro_flute_one
      \intro_flute_one_b
      s1
      s4
      \cadenzaOn
      \accidentalStyle neo-modern
      \slow_one
      s1
      \cadenzaOff
      \end_one
      \end_one
      \bar "."
    }
    \new Staff \with {instrumentName = "Flute II" } {
      \tempo 4 = 90
      \time 5/4
      \clef treble
      \intro_flute_two
      \intro_flute_two_b
      s1
      s4
      \cadenzaOn
      \accidentalStyle neo-modern
      \slow_two
      s1
      \cadenzaOff
      \end_two
      \end_two
      \bar "."
    }
    \new Staff \with {instrumentName = "Flute IV" } {
      \tempo 4 = 90
      \time 5/4
      \clef treble
      \repeat unfold 10 s1
      \flute_three_b
      \three_b_fast
      \flute_three_b_accent
      \three_b_slow
      \cadenzaOn
      \accidentalStyle neo-modern
      \slow_three
      s1
      \cadenzaOff
      \end_three
      \end_three
      \bar "."
    }
    \new Staff \with {instrumentName = "Flute V" } {
      \tempo 4 = 90
      \time 5/4
      \clef treble
      \repeat unfold 10 s1
      \flute_four_b
      \four_b_fast
      \flute_four_b_accent
      \four_b_slow
      \cadenzaOn
      \accidentalStyle neo-modern
      \slow_four
      s1
      \cadenzaOff
      \end_four
      \end_four
      \bar "."
    }
    \new Staff \with {instrumentName = "Flute III" } {
      \tempo 4 = 90
      \time 5/4
      \clef bass
      \repeat unfold 2\intro_flute_three_drone
      \intro_flute_three_drone
      \time 4/4
      \intro_flute_three_drone_b
      \cadenzaOn
      \accidentalStyle neo-modern
      \slow_five
      s1
      \cadenzaOff
      \end_bass
      \end_bass
      \bar "."
    }
  >>
}

opts.exporter = #exportMusicXML
\score {
  \music
  \layout{
    \FileExport #opts

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
}
