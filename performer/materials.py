import abjad
from abjadext import rmakers

ABJAD_INCLUDE_DIR = abjad.configuration.Configuration().abjad_directory / "abjad/scm"
abjad_ily = f'\\include "{ABJAD_INCLUDE_DIR}/abjad.ily"\n'


def make_rhythm(durations):
    nested_music = rmakers.accelerando(
        durations, [(1, 8), (1, 20), (1, 16)], [(1, 20), (1, 8), (1, 16)]
    )
    container = abjad.Container(nested_music)
    rmakers.duration_bracket(container)
    rmakers.feather_beam(container)
    music = abjad.mutate.eject_contents(container)
    return music


def main():
    time_signatures = rmakers.time_signatures([(4, 8), (3, 8), (4, 8), (3, 8)])
    durations = [abjad.Duration(_) for _ in time_signatures]
    music = make_rhythm(durations)

    lilypond_file = abjad.illustrators.selection(
        music,
        time_signatures,
    )

    abjad.LilyPondFile([abjad_ily, lilypond_file])

    abjad.show(lilypond_file)


if __name__ == "__main__":
    main()
