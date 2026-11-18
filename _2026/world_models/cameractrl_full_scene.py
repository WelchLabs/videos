"""
CameraCtrl - the full p39-40 scene (ManimGX), one continuous take:

    uv run manimgx render cameractrl_full_scene.py CameraCtrlFull

  beat 1  Stable Video Diffusion: the model block draws in, with an example input image and output video
  beat 2  CameraCtrl: camera icon + conditioning arrows into every block, title
  beat 3  TRAINING: the model block shrinks; the camera icon is replaced by a live 3D camera path (upper left),
          conditioning frame on the left, ground-truth clip on the right, cycling through RealEstate10K examples
  beat 4  INFERENCE: same layout, one conditioning image after the other (outputs/ then outputs_empty_office/),
          each rendered along several requested camera paths

All 2D artwork is redrawn from the p39_40-0x.svg mockups using their exact coordinates (1920x1080 viewBox mapped
onto manim's 8-unit-tall frame), so proportions match the mockups. Everything 2D is fixed in frame; only the
camera pyramids live in 3D, and the scene camera is steered so that 3D world lands in the upper-left corner.

Toggles are class attributes on CameraCtrlFull (show_text, video_on_pyramid, which clips / paths, timings).
"""

from camera_gx import CameraCtrlScene


class CameraCtrlFull(CameraCtrlScene):
    """The original CameraCtrl edit, with the inference clip and camera path in sync."""
