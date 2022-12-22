\version "2.24"

\include "./microlily/he.ly"

\language "english"

global = {
  \numericTimeSignature
}

intro_flute_one = {
  \global
  \tonic d
  \mixed
  \tuplet 3/2 { \tune 2 e'8-> \mf ( [ \tune 3 b'-. ) r ] } r1 |
  \tuplet 3/2 { \tune 5 g''8-> ( [ \tune 19 g''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 5 g''8-> ( [ \tune 11 a''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 13 c'''8-> ( [ \tune 5 g''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 19 g''8-> ( [ \tune 13 c'''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 19 g''8-> ( [ \tune 17 f''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 19 g''8-> ( [ \tune 17 f''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 5 g''8-> ( [ \tune 13 c'''8-. ) r ] } r1 |
}

intro_flute_one_b = {
  \global
  \tonic d
  \mixed
  \tuplet 3/2 { \tune 3 e'8-> \f ( [ \tune 2 b'-. ) r ] } r1 |
  \tuplet 3/2 { \tune 19 g''8-> ( [ \tune 5 g''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 11 g''8-> ( [ \tune 5 a''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 5 c''8-> ( [ \tune 13 g''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 13 g''8-> ( [ \tune 19 c''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 17 g''8-> ( [ \tune 19 f''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 17 g''8-> ( [ \tune 19 f''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 13 g''8-> ( [ \tune 5 c''8-. ) r ] } r1 |
}

intro_flute_two = {
  \global
  \mixed
  \tonic d
  \tuplet 3/2 { \tune 19 g''8-> \mf ( [ \tune 13 c'''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 11 a''8-> ( [ \tune 7 d'''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 19 g''8-> ( [ \tune 7 d'''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 7 d'''8-> ( [ \tune 2 e'-. ) r ] } r1 |
  \tuplet 3/2 { \tune 3 b'8-> ( [ \tune 2 e'-. ) r ] } r1 |
  \tuplet 3/2 { \tune 7 d'''8-> ( [ \tune 3 b'-. ) r ] } r1 |
  \tuplet 3/2 { \tune 11 a''8-> ( [ \tune 13 c'''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 7 d'''8-> ( [ \tune 2 e'-. ) r ] } r1 |
}

intro_flute_two_b = {
  \global
  \mixed
  \tonic d
  \tuplet 3/2 { \tune 19 g''8-> \f ( [ \tune 13 c'''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 11 a''8-> ( [ \tune 7 d'''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 19 g''8-> ( [ \tune 7 d'''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 7 d'''8-> ( [ \tune 2 e'-. ) r ] } r1 |
  \tuplet 3/2 { \tune 3 b'8-> ( [ \tune 2 e'-. ) r ] } r1 |
  \tuplet 3/2 { \tune 7 d'''8-> ( [ \tune 3 b'-. ) r ] } r1 |
  \tuplet 3/2 { \tune 11 a''8-> ( [ \tune 13 c'''-. ) r ] } r1 |
  \tuplet 3/2 { \tune 7 d'''8-> ( [ \tune 2 e'-. ) r ] } r1 |
}

intro_flute_three_drone = {
  \global
  \mixed
  \tonic d
  \repeat unfold 4 { \tune 2 d,4.-> ( \fff\> \tune 2 d,8 ) \p r2. | }
}

intro_flute_three_drone_b = {
  \global
  \mixed
  \tonic d
  \tune 2 d,4.-> ( \fff\> \tune 2 d,8 ) \p r2 |
  \tune 2 d,4.-> ( \fff\> \tune 2 d,8 ) \p r2 |
  \tune 2 d,4.-> ( \fff\> \tune 2 d,8 ) \p r2 |
  \tune 2 d,4.-> ( \fff\> \tune 2 d,8 ) \p r2 |
  r1
}

flute_three_b = {
  \global
  \mixed
  \tonic d
  \tune 2 d''16-> ( \f \> \tune 7 d'' \tune 11 d'' \tune 5 d''
  \tune 2 d''16-> \tune 7 d'' \tune 11 d'' \tune 5 d''
  \tune 2 d''16-> \tune 7 d'' \tune 11 d'' \tune 5 d'' )
  \tune 9 d'-> ( [ \tune 2 d''16  \tune 7 d'' \tune 11 d'' \tune 5 d'' ) ] \p
  \tune 9 d'-> ( [ \mf
  \tonic f
  \tune 2 d'16  \tune 7 d'' ) ] |

  \tune 2 d''16-> ( [ \f \> \tune 7 d' \tune 11 d'' ) ]
  \tune 2 d''16-> ( [ \tune 7 d' \tune 11 d'') ]
  \tune 2 d''16-> ( [ \tune 7 d' \tune 11 d'') ]
  \tune 2 d''16-> ( [ \tune 7 d' \tune 11 d'' \tune 5 d''' ) ]
  \tune 2 d''16-> ( [ \tune 7 d' \tune 11 d'' \tune 5 d''' ) ] \p
  \tonic g
  \tune 2 d'16-> ( [ \mf \tune 7 d'' \tune 11 d''' ) ] |

  \tune 2 d''16-> ( [ \f \> \tune 7 d' ) ]
  \tune 2 d''16-> ( [ \tune 7 d' ) ]
  \tune 2 d''16-> ( [ \tune 7 d' ) ]
  \tune 2 d''16-> ( [ \tune 7 d' ) ]
  \tune 2 d''16-> ( [ \tune 7 d' \tune 11 d'' ) ]
  \tune 2 d''16-> ( [ \tune 7 d' \tune 11 d'' ) ]
  \tune 2 d''16-> ( [ \tune 7 d' \tune 11 d'' \tune 5 d'' ) ] \p
  \tonic e
  \tune 5 d''16-> ( [ \mf \tune 9 d''' ] ) |
}

flute_three_b_accent = {
  \global
  \mixed
  \tonic d
  \tune 2 d''16-> ( \f \> \tune 2 d'' \tune 11 d'' \tune 5 d''->
  \tune 5 d''16 \tune 7 d'' \tune 11 d''-> \tune 11 d''
  \tune 2 d''16 \tune 7 d''-> \tune 7 d'' \tune 5 d'' )
  \tune 9 d'-> ( [ \tune 9 d'16  \tune 7 d'' \tune 11 d''-> \tune 11 d'' ) ] \p
  \tune 9 d' ( [ \mf
  \tonic f
  \tune 2 d'16->  \tune 2 d' ) ] |

  \tune 2 d''16 ( [ \f \> \tune 7 d'-> \tune 7 d' ) ]
  \tune 2 d''16 ( [ \tune 7 d'-> \tune 7 d') ]
  \tune 2 d''16 ( [ \tune 7 d'-> \tune 7 d') ]
  \tune 2 d''16 ( [ \tune 7 d'-> \tune 7 d' \tune 5 d''' ) ]
  \tune 2 d''16-> ( [ \tune 2 d'' \tune 11 d'' \tune 5 d'''-> ) ] \p
  \tonic g
  \tune 5 d'''16 ( [ \mf \tune 7 d'' \tune 11 d'''-> ) ] |

  \tune 11 d'''16 ( [ \f \> \tune 7 d' ) ]
  \tune 2 d''16-> ( [ \tune 2 d'' ) ]
  \tune 2 d''16 ( [ \tune 7 d'-> ) ]
  \tune 7 d'16 ( [ \tune 7 d' ) ]
  \tune 2 d''16-> ( [ \tune 2 d'' \tune 11 d'' ) ]
  \tune 2 d''16-> ( [ \tune 2 d'' \tune 11 d'' ) ]
  \tune 2 d''16-> ( [ \tune 2 d'' \tune 11 d'' \tune 5 d''-> ) ] \p
  \tonic e
  \tune 5 d''16 ( [ \mf \tune 9 d''' ] ) |
}

three_b_fast = {
  \global
  \mixed
  \tonic e
  \tune 2 d''16-> ( [ \f \> \tune 7 d' ) ]
  \tune 2 d''16-> ( [ \tune 7 d' \tune 11 d'' ) ]
  \tune 2 d''16-> ( [ \tune 7 d' \tune 11 d'' \tune 5 d'' ) ]
  \tune 2 d''16-> ( [ \tune 7 d' \tune 11 d'' \tune 5 d'' \tune 9 d'' ) ] \p
  \tune 2 d''16-> \mf \< ( [ \tune 7 d' \tune 11 d'' \tune 5 d'' \tune 9 d'' \tune 2 d''' ) ] \f
}

three_b_slow = {
  \global
  \mixed
  \tonic e
  \tune 2 d''16-> ( [ \f \> \tune 3 d'' \tune 5 d'' ) ]
  \tune 2 d''16-> ( [\tune 3 d'' \tune 5 d'' \tune 7 d' ) ]
  \tuplet 3/2 { \tune 2 d''8-> ( [ \tune 3 d'' \tune 5 d'' ) \mp \< ] }
  \tune 7 d''8-> ( [ \tune 9 d'' ] )
  \tune 11 d''8.-> \tune 13 d''8 ( \f |
  \tune 13 d''1 ) r4 |
}

flute_four_b = {
  \global
  \mixed
  \tonic d
  \tune 17 d'16-> ( \f \> \tune 3 d''' \tune 13 d'' \tune 19 d''
  \tune 17 d'16-> \tune 3 d''' \tune 13 d'' \tune 19 d''
  \tune 17 d'16-> \tune 3 d''' \tune 13 d'' \tune 19 d'' )
  \tune 11 d''-> ( [ \tune 17 d'16  \tune 3 d''' \tune 13 d' \tune 19 d'' ) ] \p
  \tune 11 d'-> ( [ \mf
  \tonic f
  \tune 17 d'16  \tune 3 d'' ) ] |

  \tune 17 d'16-> ( [ \f \> \tune 3 d'' \tune 19 d' ) ]
  \tune 17 d'16-> ( [ \tune 3 d'' \tune 19 d') ]
  \tune 17 d'16-> ( [ \tune 3 d'' \tune 19 d') ]
  \tune 17 d'16-> ( [ \tune 3 d'' \tune 19 d' \tune 13 d'' ) ]
  \tune 17 d'16-> ( [ \tune 3 d'' \tune 19 d' \tune 13 d'' ) ] \p
  \tonic g
  \tune 17 d'16-> ( [ \mf \tune 3 d'' \tune 19 d'' ) ] |

  \tune 17 d''16-> ( [ \f \> \tune 3 d'' ) ]
  \tune 17 d''16-> ( [ \tune 3 d'' ) ]
  \tune 17 d''16-> ( [ \tune 3 d'' ) ]
  \tune 17 d''16-> ( [ \tune 3 d'' ) ]
  \tune 17 d''16-> ( [ \tune 3 d'' \tune 19 d'' ) ]
  \tune 17 d''16-> ( [ \tune 3 d'' \tune 19 d'' ) ]
  \tune 17 d''16-> ( [ \tune 3 d'' \tune 19 d'' \tune 13 d'' ) ] \p
  \tonic e
  \tune 13 d''16-> ( [ \mf \tune 19 d''' ] ) |
}

flute_four_b_accent = {
  \global
  \mixed
  \tonic d
  \tune 17 d'16-> ( \f \> \tune 17 d' \tune 13 d'' \tune 19 d''
  \tune 17 d'16 \tune 3 d'''-> \tune 3 d''' \tune 19 d''
  \tune 17 d'16 \tune 3 d''' \tune 13 d''-> \tune 19 d'' )
  \tune 11 d'' ( [ \tune 17 d'16  \tune 3 d''' \tune 13 d'-> \tune 13 d' ) ] \p
  \tune 11 d' ( [ \mf
  \tonic f
  \tune 17 d'16  \tune 3 d'' ) ] |

  \tune 17 d'16-> ( [ \f \> \tune 17 d' \tune 19 d' ) ]
  \tune 17 d'16 ( [ \tune 3 d'' \tune 19 d'->) ]
  \tune 19 d'16 ( [ \tune 3 d'' \tune 19 d') ]
  \tune 17 d'16 ( [ \tune 3 d''-> \tune 3 d'' \tune 13 d'' ) ]
  \tune 17 d'16 ( [ \tune 3 d'' \tune 19 d'-> \tune 19 d' ) ] \p
  \tonic g
  \tune 17 d'16 ( [ \mf \tune 3 d'' \tune 19 d'' ) ] |

  \tune 17 d''16-> ( [ \f \> \tune 17 d'' ) ]
  \tune 17 d''16 ( [ \tune 3 d'' ) ]
  \tune 17 d''16 ( [ \tune 3 d''-> ) ]
  \tune 3 d''16 ( [ \tune 3 d'' ) ]
  \tune 17 d''16 ( [ \tune 3 d'' \tune 19 d''-> ) ]
  \tune 19 d''16 ( [ \tune 3 d'' \tune 19 d'' ) ]
  \tune 17 d''16 ( [ \tune 3 d''-> \tune 3 d'' \tune 13 d'' ) ] \p
  \tonic e
  \tune 13 d''16 ( [ \mf \tune 19 d''' ] ) |
}

four_b_fast = {
  \global
  \mixed
  \tonic e
  \tune 17 d''16-> ( [ \f \> \tune 3 d' ) ]
  \tune 17 d''16-> ( [ \tune 3 d' \tune 13 d'' ) ]
  \tune 17 d''16-> ( [ \tune 3 d' \tune 13 d'' \tune 19 d'' ) ]
  \tune 17 d''16-> ( [ \tune 3 d' \tune 13 d'' \tune 19 d'' \tune 11 d'' ) ] \p
  \tune 17 d''16-> \mf \< ( [ \tune 3 d' \tune 13 d'' \tune 19 d'' \tune 11 d'' \tune 17 d''' ) ] \f
}

four_b_slow = {
  \global
  \mixed
  \tonic e
  \tune 19 d''16-> ( [ \f \> \tune 17 d'' \tune 13 d'' ) ]
  \tune 19 d''16-> ( [\tune 17 d'' \tune 13 d'' \tune 11 d' ) ]
  \tuplet 3/2 { \tune 19 d''8-> ( [ \tune 17 d'' \tune 13 d'' ) \mp \< ] }
  \tune 19 d''8-> ( [ \tune 17 d'' ] )
  \tune 13 d''8.-> \tune 11 d''8 ( \f |
  \tune 11 d''1 ) r4 |
}

% =================== Slow music =========================
slow_one = {
  \global
  \mixed
  \tonic e
  \tune 31 e'''1 ( \pp \<
  \tune 31 e'''1 \mf \>
  \tune 31 e'''1 ) \pp
  \tune 31 e'''1 ( \pp \<
  \tune 31 e'''1 \mf \>
  \tune 31 e'''1 ) \pp
  \break
  \tune 30 e'''1 ( \pp \<
  \tune 30 e'''1 \mf \>
  \tune 30 e'''1 ) \pp
  s2
  \tune 31 e'''1 ( \pp \<
  \tune 31 e'''1 \mf \>
  \tune 31 e'''1 ) \pp
  \tune 31 e'''1 ( \pp \<
  \tune 31 e'''1 \mf \>
  \tune 31 e'''1 ) \pp
  \tune 30 e'''1 ( \pp \<
  \tune 30 e'''1 \mf \>
  \tune 30 e'''1 ) \pp
  \break

  \tune 30 e'''1 ( \pp \<
  \tune 30 e'''1 \mf \>
  \tune 30 e'''1 ) \pp
  \tune 26 e'''1 ( \pp \<
  \tune 26 e'''1 \mf \>
  \tune 26 e'''1 ) \pp
  \tune 22 e'''2 ( \pp \<
  \tune 22 e'''2 \mf \>
  \tune 22 e'''2 ) \pp
  \tune 18 e'''2 ( \pp \<
  \tune 18 e'''2 \mf \>
  \tune 18 e'''2 ) \pp
  \tune 18 e''2 ( \p \<
  \tune 18 e''2 \f \>
  \tune 18 e''2 ) \p

  \tune 14 e''4 ( \p \<
  \tune 14 e''4 \f \>
  \tune 14 e''4 ) \p
  \tune 10 e''4 ( \mp \<
  \tune 10 e''4 \ff \>
  \tune 10 e''4 ) \mp
  \tune 6 e''4 ( \mp \<
  \tune 6 e''4 \ff \>
  \tune 6 e''4 ) \mp

  \tune 7 e''1 \mf
  \tune 7 e''4-> \mf
  \tune 7 e''1 \mf
  \tune 7 e''4-> \mf
  \tune 7 e''1 \mf
  \tune 7 e''4-> \mf
  \tune 7 e''1 \mf
  \tune 7 e''4-> \mf
  \tune 7 e''1 \mf
  \tune 7 e''4-> \mf

  s1 s4 s4
  \tune 35 e''2. ( \ppp \< \tune 35 e''4 ) \mf
  s1
  \tune 35 e''2. ( \ppp \< \tune 35 e''4 ) \mf
}

slow_two = {
  \global
  \mixed
  \tonic e
  s4.
  \tune 29 e''1 ( \pp \<
  \tune 29 e''1 \mf \>
  \tune 29 e''1 ) \pp
  \tune 29 e'''1 ( \pp \<
  \tune 29 e'''1 \mf \>
  \tune 29 e'''1 ) \pp
  s2 s8
  \tune 29 e''1 ( \pp \<
  \tune 29 e''1 \mf \>
  \tune 29 e''1 ) \pp
  \tune 29 e'''1 ( \pp \<
  \tune 29 e'''1 \mf \>
  \tune 29 e'''1 ) \pp
  \break
  \tune 29 e'''2 ( \p \<
  \tune 29 e'''2 \f \>
  \tune 29 e'''2 ) \p
  \tune 26 e'''2 ( \p \<
  \tune 26 e'''2 \f \>
  \tune 26 e'''2 ) \p
  \tune 29 e''1 ( \pp \<
  \tune 29 e''1 \mf \>
  \tune 29 e''1 ) \pp

  \tune 29 e'''1 ( \pp \<
  \tune 29 e'''1 \mf \>
  \tune 29 e'''1 ) \pp
  \tune 25 e'''1 ( \pp \<
  \tune 25 e'''1 \mf \>
  \tune 25 e'''1 ) \pp
  \tune 21 e'''2 ( \pp \<
  \tune 21 e'''2 \mf \>
  \tune 21 e'''2 ) \pp
  \tune 17 e'''2 ( \pp \<
  \tune 17 e'''2 \mf \>
  \tune 17 e'''2 ) \pp
  \tune 17 e''2 ( \p \<
  \tune 17 e''2 \f \>
  \tune 17 e''2 ) \p

  \tune 13 e''4 ( \p \<
  \tune 13 e''4 \f \>
  \tune 13 e''4 ) \p
  \tune 9 e''4 ( \mp \<
  \tune 9 e''4 \ff \>
  \tune 9 e''4 ) \mp
  \tune 5 e''4 ( \mp \<
  \tune 5 e''4 \ff \>
  \tune 5 e''4 ) \mp

  \tune 5 e'1 \mf
  \tune 5 e'4-> \mf
  \tune 5 e'1 \mf
  \tune 5 e'4-> \mf
  \tune 5 e'1 \mf
  \tune 5 e'4-> \mf
  \tune 5 e'1 \mf
  \tune 5 e'4-> \mf
  \tune 5 e'1 \mf

  s1 s4
  \tune 28 e''2. ( \ppp \< \tune 28 e''4 ) \mf
  s1
  \tune 28 e''2. ( \ppp \< \tune 28 e''4 ) \mf
}

slow_three = {
  \global
  \mixed
  \tonic e
  s1 s1
  \tune 23 e''1 ( \pp \<
  \tune 23 e''1 \mf \>
  \tune 23 e''1 ) \pp
  \tune 21 e'1 ( \pp \<
  \tune 21 e'1 \mf \>
  \tune 21 e'1 ) \pp
  s1
  \tune 6 e'2 ( \pp \<
  \tune 6 e'2 \f \>
  \tune 6 e'2 ) \pp
  \tune 8 e'2 ( \pp \<
  \tune 8 e'2 \f \>
  \tune 8 e'2 ) \pp
  s1 s2
  \tune 23 e'''2 ( \p \<
  \tune 23 e'''2 \f \>
  \tune 23 e'''2 ) \p
  \tune 28 e'''2 ( \p \<
  \tune 28 e'''2 \f \>
  \tune 28 e'''2 ) \p
  \tune 23 e''1 ( \pp \<
  \tune 23 e''1 \mf \>
  \tune 23 e''1 ) \pp

  \tune 28 e''1 ( \pp \<
  \tune 28 e''1 \mf \>
  \tune 28 e''1 ) \pp
  \tune 24 e'''1 ( \pp \<
  \tune 24 e'''1 \mf \>
  \tune 24 e'''1 ) \pp
  \tune 20 e'''2 ( \pp \<
  \tune 20 e'''2 \mf \>
  \tune 20 e'''2 ) \pp
  \tune 16 e'''2 ( \pp \<
  \tune 16 e'''2 \mf \>
  \tune 16 e'''2 ) \pp
  \tune 16 e''2 ( \p \<
  \tune 16 e''2 \f \>
  \tune 16 e''2 ) \p

  \tune 12 e''4 ( \p \<
  \tune 12 e''4 \f \>
  \tune 12 e''4 ) \p
  \tune 8 e''4 ( \mp \<
  \tune 8 e''4 \ff \>
  \tune 8 e''4 ) \mp
  \tune 4 e''4 ( \mp \<
  \tune 4 e''4 \ff \>
  \tune 4 e''4 ) \pp

  \tune 3 e''1 \mf
  \tune 3 e''4-> \mf
  \tune 3 e''1 \mf
  \tune 3 e''4-> \mf
  \tune 3 e''1 \mf
  \tune 3 e''4-> \mf
  \tune 3 e''1 \mf
  \tune 3 e''4-> \mf

  s1 s4 s2
  \tune 21 e''2. ( \ppp \< \tune 21 e''4 ) \mf
  s1
  \tune 21 e''2. ( \ppp \< \tune 21 e''4 ) \mf
}

slow_four = {
  \global
  \mixed
  \tonic e
  s1 s2
  \tune 7 e'4 ( \pp \<
  \tune 7 e'4 \mf \>
  \tune 7 e'4 ) \pp
  \tune 17 e''2 ( \pp \<
  \tune 17 e''2 \mf \>
  \tune 17 e''2 ) \pp
  s1
  \tune 3 e'4 ( \pp \<
  \tune 3 e'4 \ff \>
  \tune 3 e'4 ) \pp
  \tune 5 e'4 ( \pp \<
  \tune 5 e'4 \ff \>
  \tune 5 e'4 ) \pp
  s2
  \tune 7 e'2 ( \pp \<
  \tune 7 e'2 \ff \>
  \tune 7 e'2 ) \pp
  s4
  \tune 5 e'2 ( \pp \<
  \tune 5 e'2 \f \>
  \tune 5 e'2 ) \pp
  \tune 7 e'2 ( \pp \<
  \tune 7 e'2 \f \>
  \tune 7 e'2 ) \pp
  s1 s4
  \tune 27 e'''2 ( \p \<
  \tune 27 e'''2 \f \>
  \tune 27 e'''2 ) \p
  \tune 24 e'''2 ( \p \<
  \tune 24 e'''2 \f \>
  \tune 24 e'''2 ) \p
  \tune 7 e''2 ( \pp \<
  \tune 7 e''2 \mf \>
  \tune 7 e''2 ) \pp

  \tune 31 e''1 ( \pp \<
  \tune 31 e''1 \mf \>
  \tune 31 e''1 ) \pp
  \tune 27 e''1 ( \pp \<
  \tune 27 e''1 \mf \>
  \tune 27 e''1 ) \pp
  \break
  \tune 23 e'''2 ( \pp \<
  \tune 23 e'''2 \mf \>
  \tune 23 e'''2 ) \pp
  \tune 19 e'''2 ( \pp \<
  \tune 19 e'''2 \mf \>
  \tune 19 e'''2 ) \pp
  \tune 15 e'''2 ( \pp \<
  \tune 15 e'''2 \mf \>
  \tune 15 e'''2 ) \pp
  \tune 15 e''2 ( \p \<
  \tune 15 e''2 \f \>
  \tune 15 e''2 ) \p

  \break
  \tune 11 e''4 ( \p \<
  \tune 11 e''4 \f \>
  \tune 11 e''4 ) \p
  \tune 7 e''4 ( \mp \<
  \tune 7 e''4 \ff \>
  \tune 7 e''4 ) \mp
  \tune 3 e''4 ( \mp \<
  \tune 3 e''4 \ff \>
  \tune 3 e''4 ) \mp

  \tune 1 e'1 \mf
  \tune 1 e'4-> \mf
  \tune 1 e'1 \mf
  \tune 1 e'4-> \mf
  \tune 1 e'1 \mf
  \tune 1 e'4-> \mf
  \tune 1 e'1 \mf
  \tune 1 e'4-> \mf
  \tune 1 e'1 \mf
  \tune 1 e'4-> \mf

  s1 s4
  \tune 14 e''2. ( \ppp \< \tune 14 e''4 ) \mf
  s1
  \tune 14 e''2. ( \ppp \< \tune 14 e''4 ) \mf
}

slow_five = {
  \repeat unfold 34 s1
  \tune 1 e1 ( \ppp \< \tune 1 e1 \tune 1 e1 \tune 1 e1 \tune 1 e1 \tune 1 e4 \ff )
  \break
  s1
  \tune 7 e,1
  s1
  \tune 7 e,1
}

end_one = {
  \tonic d
  \break
  \time 5/4
  \tuplet 3/2 { \tune 2 e''8-> \mf ( [ \tune 7 b-. ) r ] } r4
  \tuplet 3/2 { \tune 2 e''8-> \mf ( [ \tune 7 b-. ) r ] } r2 |
  \tuplet 3/2 { \tune 2 e''8-> \mf ( [ \tune 7 b-. ) r ] } r2
  \tuplet 3/2 { \tune 2 e''8-> \mf ( [ \tune 7 b-. ) r ] } r4 |
  \time 2/4
  \tune 2 e''4-> \mf \tune 7 b |
}

end_three = {
  \tonic d
  \break
  \time 5/4
  \tuplet 3/2 { \tune 11 e''8-> \mf ( [ \tune 3 b'-. ) r ] } r4
  \tuplet 3/2 { \tune 11 e''8-> \mf ( [ \tune 3 b'-. ) r ] } r2 |
  \tuplet 3/2 { \tune 11 e''8-> \mf ( [ \tune 3 b'-. ) r ] } r2
  \tuplet 3/2 { \tune 11 e''8-> \mf ( [ \tune 3 b'-. ) r ] } r4 |
  \time 2/4
  \tune 11 e''4-> \mf \tune 3 b' |
}

end_two = {
  \tonic d
  \break
  \time 5/4
  \tuplet 3/2 { \tune 5 e'8-> \mf ( [ \tune 13 b'-. ) r ] } r2
  \tuplet 3/2 { \tune 5 e'8-> \mf ( [ \tune 13 b'-. ) r ] } r4 |
  \tuplet 3/2 { \tune 5 e'8-> \mf ( [ \tune 13 b'-. ) r ] } r4
  \tuplet 3/2 { \tune 5 e'8-> \mf ( [ \tune 13 b'-. ) r ] } r2 |
  \time 2/4
  \tune 5 e'4-> \mf \tune 13 b' |
}

end_four = {
  \tonic d
  \break
  \time 5/4
  \tuplet 3/2 { \tune 7 e'8-> \mf ( [ \tune 17 b'-. ) r ] } r2
  \tuplet 3/2 { \tune 7 e'8-> \mf ( [ \tune 17 b'-. ) r ] } r4 |
  \tuplet 3/2 { \tune 7 e'8-> \mf ( [ \tune 17 b'-. ) r ] } r4
  \tuplet 3/2 { \tune 7 e'8-> \mf ( [ \tune 17 b'-. ) r ] } r2 |
  \time 2/4
  \tune 7 e'4-> \mf \tune 17 b' |
}

end_bass = {
  \tonic d
  \time 5/4
  \tune 2 e,4 \f r4 \tune 2 e,4 \tune 2 e,4 r4 |
  \tune 2 e,4 r4 \tune 2 e,4 \tune 2 e,4 r4 |
  \time 2/4
  \tune 2 e,4 e, |
}
% =================== Below is garbage ===================
flute_one_a = {
  \global
  \tonic d
  \mixed
  \tuplet 3/2 { \tune 5 d''8-> ( \tune 13 d''4~ } \tune 13 d''4 ) r2. |
  \tuplet 3/2 { \tune 13 d''8-> ( \tune 2 d''4~ } \tune 2 d''4 ) r2. |
  \tuplet 3/2 { \tune 11 d''8-> ( \tune 5 d'4~ } \tune 5 d'4 ) r2. |
  \tuplet 3/2 { \tune 17 d'8-> ( \tune 5 d''4~ } \tune 5 d''4 ) r2. |
  \tuplet 3/2 { \tune 11 d''8-> ( \tune 2 d''4~ } \tune 2 d''4 ) r2. |
  \tuplet 3/2 { \tune 3 d''8-> ( \tune 5 d''4~ } \tune 5 d''4 ) r2. |
  \tuplet 3/2 { \tune 17 d''8-> ( \tune 2 d'4~ } \tune 2 d'4 ) r2. |
  \tuplet 3/2 { \tune 5 d''8-> ( \tune 2 d''4~ } \tune 2 d''4 ) r2. |
}

flute_two_a = {
  \global
  \mixed
  \tonic f
  r4 \tuplet 3/2 { \tune 3 f''8-> ( \tune 9 f''4~ } \tune 9 f''4 ) r2 |
  r4 \tuplet 3/2 { \tune 17 f'8-> ( \tune 5 f''4~ } \tune 5 f''4 ) r2 |
  r4 \tuplet 3/2 { \tune 13 f''8-> ( \tune 3 f'4~ } \tune 3 f'4 ) r2 |
  r4 \tuplet 3/2 { \tune 19 f''8-> ( \tune 7 f'4~ } \tune 7 f'4 ) r2 |
  r4 \tuplet 3/2 { \tune 3 f'8-> ( \tune 13 f''4~ } \tune 13 f''4 ) r2 |
  r4 \tuplet 3/2 { \tune 17 f'8-> ( \tune 5 f''4~ } \tune 5 f''4 ) r2 |
  r4 \tuplet 3/2 { \tune 17 f''8-> ( \tune 7 f'4~ } \tune 7 f'4 ) r2 |
  r4 \tuplet 3/2 { \tune 13 f'8-> ( \tune 2 f''4~ } \tune 2 f''4 ) r2 |
}

flute_four_a = {
  \global
  \mixed
  \tonic g
  r4. \tuplet 3/2 { \tune 3 g''8-> ( \tune 9 g''4~ } \tune 9 g''4 ) r4. |
  r4. \tuplet 3/2 { \tune 19 g''8-> ( \tune 13 g'4~ } \tune 13 g'4 ) r4. |
  r4. \tuplet 3/2 { \tune 3 g''8-> ( \tune 11 g''4~ } \tune 11 g''4 ) r4. |
  r4. \tuplet 3/2 { \tune 7 g'8-> ( \tune 5 g''4~ } \tune 5 g''4 ) r4. |
  r4. \tuplet 3/2 { \tune 2 g''8-> ( \tune 3 g'4~ } \tune 3 g'4 ) r4. |
  r4. \tuplet 3/2 { \tune 2 g'8-> ( \tune 17 g''4~ } \tune 17 g''4 ) r4. |
  r4. \tuplet 3/2 { \tune 7 g'8-> ( \tune 5 g''4~ } \tune 5 g''4 ) r4. |
  r4. \tuplet 3/2 { \tune 11 g''8-> ( \tune 2 g'4~ } \tune 2 g'4 ) r4. |
}

flute_five_a = {
  \global
  \mixed
  \tonic e
  r2. \tuplet 3/2 { \tune 3 e''8-> ( \tune 9 e''4~ } \tune 9 e''4 ) |
  r2. \tuplet 3/2 { \tune 3 e''8-> ( \tune 11 e''4~ } \tune 11 e''4 ) |
  r2. \tuplet 3/2 { \tune 19 e''8-> ( \tune 17 e'4~ } \tune 17 e'4 ) |
  r2. \tuplet 3/2 { \tune 13 e''8-> ( \tune 5 e''4~ } \tune 5 e''4 ) |
  r2. \tuplet 3/2 { \tune 3 e''8-> ( \tune 5 e''4~ } \tune 5 e''4 ) |
  r2. \tuplet 3/2 { \tune 5 e'8-> ( \tune 11 e''4~ } \tune 11 e''4 ) |
  r2. \tuplet 3/2 { \tune 19 e''8-> ( \tune 3 e''4~ } \tune 3 e''4 ) |
  r2. \tuplet 3/2 { \tune 3 e''8-> ( \tune 17 e''4~ } \tune 17 e''4 ) |
}

flute_three_a = {
  \global
  \mixed
  \repeat unfold 2 {
    \tonic d
    \tune 2 d,4.-> ( \fff\> \tune 2 d,8 ) \p r2 |
    \tonic f
    \tune 2 d,4.-> ( \fff\> \tune 2 d,8 ) \p r2 |
    \tonic g
    \tune 2 d,4.-> ( \fff\> \tune 2 d,8 ) \p r2 |
    \tonic e
    \tune 2 d,4.-> ( \fff\> \tune 2 d,8 ) \p r2 |
  }
    \tonic d
    \tune 2 d,4.-> ( \fff\> \tune 2 d,8 ) \p r2 |
    \tonic f
    \tune 2 d,4.-> ( \fff\> \tune 2 d,8 ) \p r2 |
}

canon_one = {
  \global
  \tonic e
  \mixed
  \shape #'((-0.1 . 0.1) (0 . 2) (0 . 2) (0 . 0)) Slur
  \tune 5 g''4 \mf\> (
  \tune 7 d'''4~ \tune 7 d'''4 \pp )
  \shape #'((-0.1 . 0.1) (0 . 2) (0 . 2) (0 . 0)) Slur
  \tune 2 e'8-> ( \tune 5 g''8 )
  \shape #'((-0.1 . 0.1) (0 . 2) (0 . 2) (0 . 0)) Slur
  \tune 2 e'8-> \< ( \tune 3 b'8
  \tune 5 g''8 \mf ) \tune 3 b'8~ \> ( \tune 3 b'8
  \tune 2 e'8
  \tune 3 b'8 \tune 2 e'8 \pp )
}

canon_two = {
  \global
  \tonic e
  \mixed
  \tune 17 f''8-> \mf \tune 13 c'''8 ( \pp \<
  \tune 19 g''8 \tune 17 f''8 \mf )
  \tune 13 c'''8 \> ( \tune 19 g''8
  \tune 17 f''8 ) \tune 19 g''8~ \pp ( \tune 19 g''8
  \tune 11 a''8 -> \<
  \tune 13 c'''4 )
  \tune 11 a''8 ( \tune 17 f''8->
  \tune 11 a''4 \mf )
}

harmon = {
  \global
  \tonic e
  \mixed

  \clef bass
  \ottava #-1
  \set Staff.ottavation = #"8vb"
  \tune 1 e,,4 \fff \> \tune 2 e,4 \tune 3 b,4 \tune 4 e4
  \ottava #0
  \tune 5 e4 \tune 6 e'4 \tune 7 e'4 \tune 8 e'4
  \clef treble
  \tune 9 f'4 \tune 10 g'4 \tune 11 a'4 \tune 12 b'4
}
