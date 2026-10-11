# Video Mocap to Rigify

Blender add-on that extracts human pose landmarks from a video with
[MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
and retargets them onto a Rigify armature.

## Requirements

- Blender 3.6 or newer
- Internet access on first setup
- A video containing a visible human subject
- A Rigify armature, or another armature using compatible bone names

The add-on installs `mediapipe` and `opencv-python` into Blender's bundled
Python. The installation runs in the background so Blender remains responsive.

## Installation

1. Download the release ZIP file.
2. In Blender, open **Edit > Preferences > Add-ons**.
3. Select **Install...** and choose the downloaded ZIP.
4. Enable **Video Mocap to Rigify**.
5. Open the 3D View sidebar with `N`, then select the **Mocap** tab.
6. Click **Install MediaPipe + OpenCV**. Installation progress and the latest
   pip status are shown in the panel. Press `Esc` to cancel.

If Blender cannot import the packages after installation, restart Blender and
try the add-on again.

## Basic workflow

1. Choose a source video.
2. Select the target armature.
3. Choose the model complexity:
   - `0` is fastest.
   - `2` is most accurate.
4. Adjust visibility, scale, smoothing, mirror, depth, and root-motion options
   as needed.
5. Click **Extract & Bake from Video**.
6. Review the generated animation and use **Clear Baked Animation** if needed.

The video is processed frame by frame. Longer or higher-resolution videos take
more time and memory.

## Supported bone mapping

The add-on includes aliases for common Rigify names such as `spine`,
`spine_fk.001`, `upper_arm.L`, `forearm.L`, `thigh.L`, and `shin.L`. Use
**Print Bone Map** in the panel to inspect which target bones were found.

## Troubleshooting

- **MediaPipe is not installed:** Click **Install MediaPipe + OpenCV** and wait
  for the status to finish. An internet connection is required.
- **Installation fails:** Check Blender's system console for the pip output,
  confirm that Blender can access the internet, and retry.
- **Video file not found:** Use an absolute or Blender-relative path and verify
  that the file still exists.
- **Poor tracking:** Improve lighting, keep the subject in frame, lower the
  visibility threshold, or try a higher model complexity.
- **Animation is mirrored or depth is reversed:** Toggle **Mirror** or
  **Flip Depth**.

## License

This project is licensed under the [MIT License](./LICENSE).
