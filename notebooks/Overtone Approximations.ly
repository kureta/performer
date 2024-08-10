\version "2.24"

\include "oll-core/package.ily"

#(ly:set-option 'relative-includes #t)

\include "./microlily/he.ly"

\language "english"

global = {
  \numericTimeSignature
}

\header {
  title = "Overtones"
  subtitle = " "
  tagline = ##f
}

\paper {
  paper-width = 210
  paper-height = 297

  system-system-spacing =
  #'((basic-distance . 12)
     (minimum-distance . 8)
     (padding . 5)
     (stretchability . 100))
}

music = \new StaffGroup {
  <<
    \new Staff \with {instrumentName = "Overtones" } {
      \override TextScript.self-alignment-X = #CENTER
      \override TextScript.Y-offset = #-7
      \clef treble
      \cadenzaOn
      \accidentalStyle neo-modern
      \tonic c
      \mixed
      \tune 1 d'4-1 \tune 2 d'4-2 \tune 3 d'4-3_"+2" \tune 4 d'4-4 \tune 5 d'4-5_"-14"
      \tune 6 d'4-6_"+2" \tune 7 d''4-7_"-31" \tune 8 d'4-8 \tune 9 d'4-9_"+4"
      \break
      \tune 10 d'4-10_"-14" \tune 11 d'4-11_"+51" \tune 12 d'4-12_"+2" \tune 13 d''4-13_"-59"
      \tune 14 d''4-14_"-31" \tune 15 d''4-15_"-12" \tune 16 d'4-16 \tune 17 d'4-17_"+5"
      \tune 18 d'4-18_"+4"
      \break
      \tune 19 d'4-19_"-2" \tune 20 d'4-20_"-14" \tune 21 d'4-21_"-29" \tune 22 d'4-22_"+51"
      \tune 23 d'4-23_"+28"
      \tune 24 d'4-24_"+2" \tune 25 d'4-25_"-27" \tune 26 d''4-26_"-59" \tune 27 d''4-27_"+6"
      \break
      \tune 28 d''4-28_"-31" \tune 29 d''4-29_"+30" \tune 30 d''4-30_"-12" \tune 31 d'4-31_"-55"
      \tune 32 d'4-32
      \tune 33 d'4-33_"53" \tune 34 d'4-34_"+5" \tune 35 d'4-35_"-45"
      \break
      \hideNotes
      r1
      \break
      r1
      \break
      r1
      \break
      r1
      \break
      r1
      \break
      r1
      \break
      r1
      \unHideNotes
    }
  >>
}

\score {
  \music
  \layout{
    indent =0.0

    \context {
      \Score
      proportionalNotationDuration = #(ly:make-moment 1/8)
      \enablePolymeter
    }

    \context {
      \Staff
      \remove "Instrument_name_engraver"
      \remove "Time_signature_engraver"
      \override VerticalAxisGroup.remove-first = ##t
      \override VerticalAxisGroup.staff-staff-spacing = #'((basic-distance . 17))
      \numericTimeSignature
    }
  }
}