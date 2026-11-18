"""
P48 - CameraCtrl edit where the inference camera path comes first:
the pyramid flies its whole path, and only then does the rendered video appear and play. This is to show
that the path is an input you have to hand the model before it generates anything.

    uv run manimgx render p48.py P48

  beat 1  Stable Video Diffusion: the model block draws in, with an example input image and output video
  beat 2  CameraCtrl: camera icon + conditioning arrows into every block, title
  beat 3  TRAINING: the model block shrinks; the camera icon is replaced by a live 3D camera path (upper left),
          conditioning frame on the left, ground-truth clip on the right, cycling through RealEstate10K examples
  beat 4  INFERENCE: same layout, one conditioning image after the other (outputs/ then outputs_empty_office/),
          each rendered along several requested camera paths - path first, then the video (path_first toggles)

All 2D artwork is redrawn from the p39_40-0x.svg mockups using their exact coordinates (1920x1080 viewBox mapped
onto manim's 8-unit-tall frame), so proportions match the mockups. Everything 2D is fixed in frame; only the
camera pyramids live in 3D, and the scene camera is steered so that 3D world lands in the upper-left corner.

Toggles are class attributes on P48 (show_text, video_on_pyramid, which clips / paths, timings).
"""

from camera_gx import CameraCtrlScene


class P48(CameraCtrlScene):
    train_clip_indices = [10, 3, 17, 24, 31, 45]
    inference_path_first = True
    bus_run_time = 2.0
    training_transition_time = 1.5
    end_hold = 2
