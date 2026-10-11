bl_info = {
    "name": "Video Mocap to Rigify",
    "author": "Kabrownie.digital",
    "version": (0, 2, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > Mocap",
    "description": "Extract pose from a video with MediaPipe and retarget onto a Rigify rig",
    "category": "Animation",
}

import os
import sys
import subprocess
import tempfile
import time

import bpy
from mathutils import Vector, Matrix, Quaternion

# ---------------------------------------------------------------------------
#  MediaPipe availability (lazy)
# ---------------------------------------------------------------------------
_MP_STATE = None

def mediapipe_available():
    global _MP_STATE
    if _MP_STATE is None:
        try:
            modules = os.path.join(os.path.dirname(__file__), "modules")
            if os.path.isdir(modules) and modules not in sys.path:
                sys.path.insert(0, modules)
            import mediapipe  # noqa: F401
            import cv2        # noqa: F401
            _MP_STATE = True
        except Exception:
            _MP_STATE = False
    return _MP_STATE


def reset_mediapipe_cache():
    global _MP_STATE
    _MP_STATE = None


def find_blender_python():
    """Return the Python interpreter bundled with the running Blender."""
    exe = sys.executable
    if exe and os.path.basename(exe).lower().startswith("python") and os.path.isfile(exe):
        return exe

    binary_dir = os.path.dirname(os.path.abspath(bpy.app.binary_path))
    candidates = [
        os.path.join(sys.prefix, "bin", "python"),
        os.path.join(sys.prefix, "bin", "python3"),
        os.path.join(binary_dir, "python", "bin", "python3"),
        os.path.join(binary_dir, "python", "bin", "python3.11"),
        os.path.join(binary_dir, "..", "python", "bin", "python3"),
    ]
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


# ---------------------------------------------------------------------------
#  Landmark / bone tables  (unchanged from the JSON version)
# ---------------------------------------------------------------------------
LM = {
    "nose": 0, "l_ear": 7, "r_ear": 8,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16,
    "l_index": 19, "r_index": 20,
    "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26,
    "l_ankle": 27, "r_ankle": 28,
    "l_heel": 29, "r_heel": 30,
    "l_foot": 31, "r_foot": 32,
}

BONE_ALIASES = {
    "hips":        ("hips", "spine"),
    "spine1":      ("spine_fk.001", "spine.001"),
    "spine2":      ("spine_fk.002", "spine.002"),
    "spine3":      ("spine_fk.003", "spine.003"),
    "chest":       ("chest",),
    "neck":        ("neck", "spine.004"),
    "head":        ("head", "spine.005"),
    "shoulder.L":  ("shoulder.L",),
    "shoulder.R":  ("shoulder.R",),
    "upper_arm.L": ("upper_arm_fk.L", "upper_arm.L"),
    "upper_arm.R": ("upper_arm_fk.R", "upper_arm.R"),
    "forearm.L":   ("forearm_fk.L", "forearm.L"),
    "forearm.R":   ("forearm_fk.R", "forearm.R"),
    "hand.L":      ("hand_fk.L", "hand.L"),
    "hand.R":      ("hand_fk.R", "hand.R"),
    "thigh.L":     ("thigh_fk.L", "thigh.L"),
    "thigh.R":     ("thigh_fk.R", "thigh.R"),
    "shin.L":      ("shin_fk.L", "shin.L"),
    "shin.R":      ("shin_fk.R", "shin.R"),
    "foot.L":      ("foot_fk.L", "foot.L"),
    "foot.R":      ("foot_fk.R", "foot.R"),
    "toe.L":       ("toe.L",),
    "toe.R":       ("toe.R",),
}

TASKS = [
    ("hips",       "hips_c",     "chest_c",    None),
    ("spine1",     "hips_c",     "chest_c",    None),
    ("spine2",     "hips_c",     "chest_c",    None),
    ("spine3",     "hips_c",     "chest_c",    None),
    ("chest",      "hips_c",     "chest_c",    None),
    ("neck",       "chest_c",    "head_c",     None),
    ("head",       "head_c",     "head_top",   None),
    ("shoulder.L", "chest_c",    "l_shoulder", None),
    ("upper_arm.L","l_shoulder", "l_elbow",    ("l_shoulder", "l_elbow", "l_wrist")),
    ("forearm.L",  "l_elbow",    "l_wrist",    ("l_shoulder", "l_elbow", "l_wrist")),
    ("hand.L",     "l_wrist",    "l_index",    None),
    ("shoulder.R", "chest_c",    "r_shoulder", None),
    ("upper_arm.R","r_shoulder", "r_elbow",    ("r_shoulder", "r_elbow", "r_wrist")),
    ("forearm.R",  "r_elbow",    "r_wrist",    ("r_shoulder", "r_elbow", "r_wrist")),
    ("hand.R",     "r_wrist",    "r_index",    None),
    ("thigh.L",    "l_hip",      "l_knee",     ("l_hip", "l_knee", "l_ankle")),
    ("shin.L",     "l_knee",     "l_ankle",    ("l_hip", "l_knee", "l_ankle")),
    ("foot.L",     "l_ankle",    "l_foot",     None),
    ("toe.L",      "l_foot",     "l_toe_tip",  None),
    ("thigh.R",    "r_hip",      "r_knee",     ("r_hip", "r_knee", "r_ankle")),
    ("shin.R",     "r_knee",     "r_ankle",    ("r_hip", "r_knee", "r_ankle")),
    ("foot.R",     "r_ankle",    "r_foot",     None),
    ("toe.R",      "r_foot",     "r_toe_tip",  None),
]


# ---------------------------------------------------------------------------
#  Math helpers  (identical to the JSON addon)
# ---------------------------------------------------------------------------
def lm_to_blender(p, flip_depth=False):
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    if flip_depth:
        z = -z
    return Vector((x, -z, -y))


def make_matrix(head, y_axis, z_hint):
    y = y_axis.copy()
    if y.length < 1e-8:
        return None
    y.normalize()
    z = z_hint - y * z_hint.dot(y)
    if z.length < 1e-6:
        z = Vector((0.0, 0.0, 1.0)) - y * y.z
        if z.length < 1e-6:
            z = Vector((0.0, 1.0, 0.0)) - y * y.y
    z.normalize()
    x = y.cross(z)
    return Matrix((
        (x.x, y.x, z.x, head.x),
        (x.y, y.y, z.y, head.y),
        (x.z, y.z, z.z, head.z),
        (0.0, 0.0, 0.0, 1.0),
    ))


def body_backward(P):
    right = P["l_hip"] - P["r_hip"]
    up = P["chest_c"] - P["hips_c"]
    if right.length < 1e-6 or up.length < 1e-6:
        return Vector((0.0, 1.0, 0.0))
    fwd = right.normalized().cross(up.normalized())
    if fwd.length < 1e-6:
        return Vector((0.0, 1.0, 0.0))
    return -fwd.normalized()


def build_joints(raw, flip_depth=False, mirror=False):
    P = {name: lm_to_blender(raw[idx], flip_depth) for name, idx in LM.items()}
    if mirror:
        swapped = {}
        for k, v in P.items():
            if k.startswith("l_"):
                nk = "r_" + k[2:]
            elif k.startswith("r_"):
                nk = "l_" + k[2:]
            else:
                nk = k
            swapped[nk] = Vector((-v.x, v.y, v.z))
        P = swapped
    P["hips_c"] = (P["l_hip"] + P["r_hip"]) * 0.5
    P["chest_c"] = (P["l_shoulder"] + P["r_shoulder"]) * 0.5
    P["head_c"] = (P["l_ear"] + P["r_ear"]) * 0.5
    P["head_top"] = P["head_c"] + (P["head_c"] - P["chest_c"])
    for s in ("l", "r"):
        P[f"{s}_toe_tip"] = P[f"{s}_foot"] + (P[f"{s}_foot"] - P[f"{s}_ankle"])
    return P


def smooth_raw(frames, radius):
    if radius <= 0:
        return frames
    n = len(frames)
    out = []
    for i in range(n):
        lo, hi = max(0, i - radius), min(n, i + radius + 1)
        acc = [0.0] * (33 * 3)
        cnt = 0
        for j in range(lo, hi):
            f = frames[j]
            if f is None:
                continue
            for k in range(33):
                acc[k * 3 + 0] += f[k][0]
                acc[k * 3 + 1] += f[k][1]
                acc[k * 3 + 2] += f[k][2]
            cnt += 1
        out.append(None if cnt == 0 else
                   [[acc[k * 3 + c] / cnt for c in range(3)] for k in range(33)])
    return out


def fill_gaps(frames):
    filled = list(frames)
    n = len(filled)
    last = None
    for i in range(n):
        if filled[i] is not None:
            last = filled[i]
        elif last is not None:
            filled[i] = last
    nxt = None
    for i in range(n - 1, -1, -1):
        if filled[i] is not None:
            nxt = filled[i]
        else:
            filled[i] = nxt
    return filled


def resolve_bones(arm):
    out = {}
    for key, names in BONE_ALIASES.items():
        pb = None
        for n in names:
            if n in arm.pose.bones:
                pb = arm.pose.bones[n]
                break
        out[key] = pb
    return out


def rest_offset(pb):
    if pb.parent is not None:
        return pb.parent.bone.matrix_local.inverted() @ pb.bone.matrix_local
    return pb.bone.matrix_local.copy()


def retarget_frame(bones, P, scale, root_offset, prev_world, use_root_motion):
    world = {}
    default_z = body_backward(P)
    for key, ja, jb, plane in TASKS:
        pb = bones.get(key)
        if pb is None:
            continue
        a, b = P[ja], P[jb]
        if pb.parent is not None:
            parent_world = world.get(pb.parent.name)
            if parent_world is None:
                parent_world = pb.parent.bone.matrix_local.copy()
        else:
            parent_world = Matrix.Identity(4)
        base = parent_world @ rest_offset(pb)
        head = base.translation.copy()
        if key == "hips" and use_root_motion:
            head = P["hips_c"] * scale + root_offset
        y_axis = b - a
        if y_axis.length < 1e-8:
            y_axis = Vector((0.0, 0.0, 1.0))
        z_hint = default_z
        if plane is not None:
            pa, pbb, pc = (P[p] for p in plane)
            n = (pbb - pa).cross(pc - pbb)
            if n.length > 1e-6:
                z_hint = n.normalized()
        M = make_matrix(head, y_axis, z_hint)
        if M is None:
            continue
        prev = prev_world.get(pb.name)
        if prev is not None:
            if M.to_3x3().col[2].dot(prev.to_3x3().col[2]) < 0.0:
                M = make_matrix(head, y_axis, -z_hint)
                if M is None:
                    continue
        world[pb.name] = M
        pb.matrix_basis = base.inverted() @ M
    return world


# ---------------------------------------------------------------------------
#  Video decoding + MediaPipe extraction
# ---------------------------------------------------------------------------
def extract_landmarks(video_path, model_complexity=2, min_visibility=0.3,
                      progress_cb=None):
    """Decode `video_path` frame by frame and return (fps, [landmark_list|None])."""
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = pose.process(rgb)

            if result.pose_world_landmarks is None:
                frames.append(None)
            else:
                lms = result.pose_world_landmarks.landmark
                vis = sum(l.visibility for l in lms) / len(lms)
                if vis < min_visibility:
                    frames.append(None)
                else:
                    frames.append([[l.x, l.y, l.z] for l in lms])

            if progress_cb is not None and idx % 5 == 0:
                progress_cb(idx, total)
    finally:
        cap.release()
        pose.close()

    if progress_cb is not None:
        progress_cb(idx, total)

    return fps, frames


# ---------------------------------------------------------------------------
#  Shared "bake from frame list" routine
# ---------------------------------------------------------------------------
def bake_frames(arm, frames, scn):
    """Retarget `frames` (list of landmark lists) onto `arm`."""
    frames = fill_gaps(frames)
    frames = smooth_raw(frames, scn.mocap_smooth)

    bones = resolve_bones(arm)
    if bones.get("hips") is None:
        raise RuntimeError("No hips/spine bone found — is this a Rigify rig?")

    if scn.mocap_force_fk:
        for pb in arm.pose.bones:
            if "IK_FK" in pb.keys():
                try:
                    pb["IK_FK"] = 0.0
                except Exception:
                    pass

    scale = scn.mocap_scale
    use_root = scn.mocap_root_motion

    hips_pb = bones["hips"]
    rest_hips = hips_pb.bone.head_local.copy()

    P0 = build_joints(frames[0], scn.mocap_flip_depth, scn.mocap_mirror)
    root_offset = rest_hips - P0["hips_c"] * scale

    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"

    if arm.animation_data is None:
        arm.animation_data_create()

    start = scn.mocap_start_frame
    prev_world = {}
    prev_quat = {}

    for i, raw in enumerate(frames):
        P = build_joints(raw, scn.mocap_flip_depth, scn.mocap_mirror)
        world = retarget_frame(bones, P, scale, root_offset,
                               prev_world, use_root)
        prev_world = world
        frame = start + i

        for key, pb in bones.items():
            if pb is None or pb.name not in world:
                continue
            q = pb.rotation_quaternion.copy()
            pq = prev_quat.get(pb.name)
            if pq is not None and q.dot(pq) < 0.0:
                q = Quaternion((-q.w, -q.x, -q.y, -q.z))
                pb.rotation_quaternion = q
            prev_quat[pb.name] = q.copy()
            pb.keyframe_insert("rotation_quaternion", frame=frame)
            if key == "hips":
                if use_root:
                    pb.keyframe_insert("location", frame=frame)
                else:
                    pb.location = (0.0, 0.0, 0.0)

    scn.frame_start = start
    scn.frame_end = start + len(frames) - 1
    return len(frames)


# ===========================================================================
#  Operators
# ===========================================================================
class MOCAP_OT_install_mediapipe(bpy.types.Operator):
    bl_idname = "mocap.install_mediapipe"
    bl_label = "Install MediaPipe + OpenCV"
    bl_description = ("pip-install mediapipe and opencv-python into Blender's "
                      "Python. Requires an internet connection and takes a minute.")
    bl_options = {"REGISTER"}

    def execute(self, context):
        if context.scene.mocap_installing:
            self.report({"WARNING"}, "MediaPipe installation is already running.")
            return {"CANCELLED"}

        python = find_blender_python()
        if python is None:
            self.report({"ERROR"},
                        "Could not locate Blender's bundled Python interpreter.")
            return {"CANCELLED"}

        try:
            self._log_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".log", prefix="mocap-pip-", delete=False)
            self._log_path = self._log_file.name
            install_script = (
                "import ensurepip, subprocess, sys; "
                "ensurepip.bootstrap(upgrade=True); "
                "raise SystemExit(subprocess.call([sys.executable, '-m', 'pip', "
                "'install', '--upgrade', '--disable-pip-version-check', "
                "'--no-input'] + sys.argv[1:]))"
            )
            modules = os.path.join(os.path.dirname(__file__), "modules")
            os.makedirs(modules, exist_ok=True)
            self._process = subprocess.Popen([
                python, "-c", install_script,
                "--target", modules,
                "mediapipe", "opencv-python",
            ], stdout=self._log_file, stderr=subprocess.STDOUT)
            self._log_file.close()
        except (OSError, subprocess.SubprocessError) as exc:
            if getattr(self, "_log_file", None) is not None:
                self._log_file.close()
            self._cleanup_install()
            self.report({"ERROR"}, f"Could not start pip: {exc}")
            return {"CANCELLED"}

        context.scene.mocap_installing = True
        context.scene.mocap_install_status = "Installing MediaPipe and OpenCV..."
        context.scene.mocap_install_log = ""
        context.scene.mocap_install_started = time.monotonic()
        context.window_manager.progress_begin(0, 100)
        context.window_manager.progress_update(5)
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "ESC":
            self._process.terminate()
            self._finish(context, "Installation cancelled.", error=True)
            return {"CANCELLED"}

        if event.type != "TIMER":
            return {"RUNNING_MODAL"}

        elapsed = int(time.monotonic() - context.scene.mocap_install_started)
        pip_status = self._read_install_status()
        context.scene.mocap_install_status = (
            f"{pip_status} ({elapsed}s elapsed)")
        context.window_manager.progress_update(min(95, 5 + elapsed // 2))

        returncode = self._process.poll()
        if returncode is None:
            return {"RUNNING_MODAL"}

        if returncode != 0:
            detail = self._read_install_log()
            self._finish(
                context,
                f"pip failed with code {returncode}.",
                error=True,
            )
            context.scene.mocap_install_log = detail
            return {"CANCELLED"}

        reset_mediapipe_cache()
        if mediapipe_available():
            self._finish(context, "MediaPipe installed successfully.")
            return {"FINISHED"}

        detail = self._read_install_log()
        self._finish(
            context,
            "Install finished, but import still fails.",
            error=True,
        )
        context.scene.mocap_install_log = detail
        return {"CANCELLED"}

    def _finish(self, context, message, error=False):
        context.window_manager.progress_end()
        if getattr(self, "_timer", None) is not None:
            context.window_manager.event_timer_remove(self._timer)
        context.scene.mocap_installing = False
        context.scene.mocap_install_status = message[:240]
        context.scene.mocap_install_log = ""
        self._cleanup_install()
        self.report({"ERROR" if error else "INFO"}, message)

    def _read_install_log(self):
        try:
            with open(self._log_path, "r", encoding="utf-8", errors="replace") as log:
                lines = [line.strip() for line in log if line.strip()]
            return "\n".join(lines[-8:])[:1200] or "No pip output was captured."
        except OSError as exc:
            return f"Could not read pip log: {exc}"

    def _read_install_status(self):
        try:
            with open(self._log_path, "r", encoding="utf-8", errors="replace") as log:
                log.seek(0, os.SEEK_END)
                log.seek(max(0, log.tell() - 512))
                lines = [line.strip() for line in log.read().splitlines() if line.strip()]
            if lines:
                return lines[-1][:100]
        except OSError:
            pass
        return "Installing..."

    def _cleanup_install(self):
        if getattr(self, "_log_file", None) is not None:
            self._log_file.close()
        if getattr(self, "_log_path", None):
            try:
                os.unlink(self._log_path)
            except OSError:
                pass


class MOCAP_OT_bake_video(bpy.types.Operator):
    bl_idname = "mocap.bake_video"
    bl_label = "Extract & Bake from Video"
    bl_description = ("Decode the video, extract pose with MediaPipe, and "
                      "retarget onto the selected armature")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scn = context.scene

        if not mediapipe_available():
            self.report({"ERROR"},
                        "MediaPipe is not installed. Click 'Install MediaPipe'.")
            return {"CANCELLED"}

        video = bpy.path.abspath(scn.mocap_video_path)
        if not os.path.isfile(video):
            self.report({"ERROR"}, "Video file not found")
            return {"CANCELLED"}

        arm = scn.mocap_target
        if arm is None or arm.type != "ARMATURE":
            self.report({"ERROR"}, "Select an armature")
            return {"CANCELLED"}

        wm = context.window_manager
        wm.progress_begin(0, 100)

        def progress(idx, total):
            if total > 0:
                wm.progress_update(int(100.0 * idx / total))

        try:
            fps, frames = extract_landmarks(
                video,
                model_complexity=scn.mocap_model_complexity,
                min_visibility=scn.mocap_min_visibility,
                progress_cb=progress,
            )
        except Exception as exc:
            wm.progress_end()
            self.report({"ERROR"}, f"Extraction failed: {exc}")
            return {"CANCELLED"}

        wm.progress_end()

        if not any(f is not None for f in frames):
            self.report({"ERROR"}, "No pose detected in the video.")
            return {"CANCELLED"}

        try:
            n = bake_frames(arm, frames, scn)
        except Exception as exc:
            self.report({"ERROR"}, f"Bake failed: {exc}")
            return {"CANCELLED"}

        self.report({"INFO"},
                    f"Baked {n} frames from {os.path.basename(video)} "
                    f"({fps:.2f} fps source)")
        return {"FINISHED"}


class MOCAP_OT_print_bones(bpy.types.Operator):
    bl_idname = "mocap.print_bones"
    bl_label = "Print Bone Map"
    bl_description = "Print which bone each logical slot resolved to (for debugging)"

    def execute(self, context):
        arm = context.scene.mocap_target
        if arm is None:
            self.report({"ERROR"}, "No armature selected")
            return {"CANCELLED"}
        bones = resolve_bones(arm)
        print("\n--- Mocap bone map for", arm.name, "---")
        missing = []
        for key in sorted(bones):
            pb = bones[key]
            if pb is None:
                missing.append(key)
                print(f"  {key:14s} -> (not found)")
            else:
                print(f"  {key:14s} -> {pb.name}")
        if missing:
            print("  Missing:", ", ".join(missing))
        print("---\n")
        self.report({"INFO"}, f"Printed map ({len(missing)} unmapped)")
        return {"FINISHED"}


class MOCAP_OT_clear(bpy.types.Operator):
    bl_idname = "mocap.clear"
    bl_label = "Clear Baked Animation"
    bl_description = "Remove all action data from the target armature"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        arm = context.scene.mocap_target
        if arm is None or arm.type != "ARMATURE":
            self.report({"ERROR"}, "Select an armature")
            return {"CANCELLED"}
        if arm.animation_data is not None:
            arm.animation_data_clear()
        for pb in arm.pose.bones:
            pb.matrix_basis = Matrix.Identity(4)
        self.report({"INFO"}, "Cleared")
        return {"FINISHED"}


# ===========================================================================
#  UI
# ===========================================================================
class MOCAP_PT_panel(bpy.types.Panel):
    bl_label = "Video Mocap → Rigify"
    bl_idname = "MOCAP_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Mocap"

    def draw(self, context):
        layout = self.layout
        scn = context.scene

        # MediaPipe status -------------------------------------------------
        if scn.mocap_installing:
            box = layout.box()
            box.label(text="Installing MediaPipe + OpenCV", icon="TIME")
            box.label(text=scn.mocap_install_status, icon="INFO")
            box.label(text="Blender remains responsive; ESC cancels.", icon="INFO")
            return

        if not mediapipe_available():
            box = layout.box()
            box.label(text="MediaPipe not installed", icon="ERROR")
            box.operator("mocap.install_mediapipe", icon="IMPORT")
            box.label(text="Requires internet; progress is shown here.", icon="INFO")
            if scn.mocap_install_status:
                box.label(text=scn.mocap_install_status, icon="INFO")
            if scn.mocap_install_log:
                box.label(text="Last pip output:", icon="CONSOLE")
                for line in scn.mocap_install_log.splitlines():
                    box.label(text=line[:160])
            return

        # Source -----------------------------------------------------------
        col = layout.column(align=True)
        col.label(text="Source Video")
        col.prop(scn, "mocap_video_path", text="")

        col = layout.column(align=True)
        col.prop(scn, "mocap_model_complexity")
        col.prop(scn, "mocap_min_visibility")

        layout.separator()

        # Target -----------------------------------------------------------
        col = layout.column(align=True)
        col.label(text="Target Armature")
        col.prop(scn, "mocap_target", text="")
        col.operator("mocap.print_bones", icon="BONE_DATA")

        layout.separator()

        # Retarget ---------------------------------------------------------
        col = layout.column(align=True)
        col.label(text="Retarget")
        col.prop(scn, "mocap_scale")
        col.prop(scn, "mocap_smooth")
        col.prop(scn, "mocap_root_motion")
        col.prop(scn, "mocap_start_frame")

        col = layout.column(align=True)
        col.prop(scn, "mocap_mirror")
        col.prop(scn, "mocap_flip_depth")
        col.prop(scn, "mocap_force_fk")

        layout.separator()
        col = layout.column(align=True)
        col.scale_y = 1.4
        col.operator("mocap.bake_video", icon="PLAY")
        col.operator("mocap.clear", icon="TRASH")


# ===========================================================================
#  Registration
# ===========================================================================
classes = (
    MOCAP_OT_install_mediapipe,
    MOCAP_OT_bake_video,
    MOCAP_OT_print_bones,
    MOCAP_OT_clear,
    MOCAP_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    scn = bpy.types.Scene

    scn.mocap_video_path = bpy.props.StringProperty(
        name="Video", subtype="FILE_PATH", default="")

    scn.mocap_target = bpy.props.PointerProperty(
        name="Armature", type=bpy.types.Object,
        poll=lambda self, obj: obj.type == "ARMATURE")

    scn.mocap_scale = bpy.props.FloatProperty(
        name="Scale", default=1.0, min=0.01, max=10.0,
        description="Multiply landmark positions by this factor")

    scn.mocap_smooth = bpy.props.IntProperty(
        name="Smooth", default=0, min=0, max=20,
        description="Extra box-filter radius over frames (0 = off)")

    scn.mocap_root_motion = bpy.props.BoolProperty(
        name="Root Motion", default=True,
        description="Also animate the hips' position (world travel)")

    scn.mocap_start_frame = bpy.props.IntProperty(
        name="Start Frame", default=1, min=0)

    scn.mocap_mirror = bpy.props.BoolProperty(
        name="Mirror", default=False,
        description="Swap left/right (subject facing away from camera)")

    scn.mocap_flip_depth = bpy.props.BoolProperty(
        name="Flip Depth", default=False,
        description="Invert the MediaPipe Z axis if depth looks reversed")

    scn.mocap_force_fk = bpy.props.BoolProperty(
        name="Force FK Mode", default=True,
        description="Set every IK_FK slider on the rig to 0 before baking")

    scn.mocap_model_complexity = bpy.props.IntProperty(
        name="Model Complexity", default=2, min=0, max=2,
        description="0 = fastest, 2 = most accurate")

    scn.mocap_min_visibility = bpy.props.FloatProperty(
        name="Min Visibility", default=0.3, min=0.0, max=1.0,
        description="Drop frames where average landmark visibility is below this")

    scn.mocap_installing = bpy.props.BoolProperty(
        name="Installing MediaPipe", default=False)

    scn.mocap_install_status = bpy.props.StringProperty(
        name="Install Status", default="")

    scn.mocap_install_started = bpy.props.FloatProperty(
        name="Install Start Time", default=0.0)

    scn.mocap_install_log = bpy.props.StringProperty(
        name="Install Log", default="")


def unregister():
    for prop in ("mocap_video_path", "mocap_target", "mocap_scale",
                 "mocap_smooth", "mocap_root_motion", "mocap_start_frame",
                 "mocap_mirror", "mocap_flip_depth", "mocap_force_fk",
                 "mocap_model_complexity", "mocap_min_visibility",
                 "mocap_installing", "mocap_install_status",
                 "mocap_install_started", "mocap_install_log"):
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()