"""Book media for the shared phoneme animation."""
from welch_paths import welch_path

from _2026.the_bitter_lesson.p5 import P5 as PhonemeAnimation


class P5(PhonemeAnimation):
    audio_fn=welch_path('bitter_lesson/exports/tell_me_about_china.wav')
    spectra_path=welch_path('bitter_lesson/hacking/spectral_envelopes_final.npy')
