# SPDX-License-Identifier: Apache-2.0
"""B1: one latest-frame worker, shared RGBD, TensorRT models and Norfair IDs.

This deliberately starts with a serial, bounded baseline. Every result in a
published scene belongs to the same frame; expensive work is gated by range
and target limits. Scheduling/precision optimizations belong after B1 capture.
"""

from copy import deepcopy
import time
import threading

import numpy as np

from app.perception import Perception
from app.vision_models import VisionModels
from app.visual_tracks import VisualTracks
from app.identity_gallery import IdentityGallery
from project_paths import ROOT


def range_from_mask(frame, mask, min_fraction=0.15, min_pixels=25):
    if frame.depth_m is None or frame.depth_valid is None:
        return {"available": False, "reason": "depth_unavailable"}
    if mask.shape != frame.depth_m.shape or mask.dtype != np.bool_:
        raise ValueError("Mask and aligned depth geometry differ")
    usable = (
        mask
        & frame.depth_valid
        & np.isfinite(frame.depth_m)
        & (frame.depth_m > 0.1)
        & (frame.depth_m < 10)
    )
    count = int(usable.sum())
    fraction = count / max(1, int(mask.sum()))
    if count < min_pixels or fraction < min_fraction:
        return {
            "available": False,
            "reason": "insufficient_depth",
            "pixels": count,
            "valid_fraction": fraction,
        }
    values = frame.depth_m[usable]
    median = float(np.median(values))
    return {
        "available": True,
        "meters": median,
        "mad_m": float(np.median(np.abs(values - median))),
        "pixels": count,
        "valid_fraction": fraction,
        "quantity": "aligned_native_depth_z_m",
        "method": "median of native aligned depth in target mask/ROI",
    }


class B1Perception(Perception):
    def __init__(self, camera, config, tracker=None, on_result=None):
        super().__init__(camera, config, tracker, on_result)
        self.networks = VisionModels(
            config.model_manifest,
            config.body_threshold,
            config.body_iou,
            config.max_people,
        )
        self.entities = VisualTracks(
            "entity", config.track_capacity, config.track_ttl_seconds
        )
        self.faces = VisualTracks(
            "face", max(4, config.max_face_targets * 3), config.track_ttl_seconds
        )
        self.gallery = None
        self._face_vectors = {}
        self._selected = None
        self._selected_last = None

    def load(self):
        if self.networks.models or self.gallery is not None:
            raise RuntimeError(
                "B1 resources are already loaded; close before reloading"
            )
        try:
            self.networks.load()
            policy = self.networks.entries["face_identity"].get("gallery_policy", {})
            self.gallery = IdentityGallery(
                ROOT / self.config.identity_gallery,
                **{
                    k: policy[k]
                    for k in ("dimensions", "threshold", "margin")
                    if k in policy
                },
            )
        except BaseException:
            self.networks.close()
            raise

    def start(self):
        if not self.networks.models:
            raise RuntimeError("B1 models have not loaded")
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="perception-b1", daemon=True
        )
        self._thread.start()

    def _fresh(self, frame):
        return (
            not self._stop.is_set()
            and frame.source_epoch == self.camera.source_epoch
            and not (frame.invalidated and frame.invalidated.is_set())
            and time.monotonic() - frame.received_monotonic <= self.camera.stale_after
        )

    def _eligible(self, row):
        distance = row["range"]
        return (
            distance["available"]
            and self.config.distance_min_m
            <= distance["meters"]
            <= self.config.distance_max_m
        )

    def process_frame(self, frame):
        if not self._fresh(frame):
            raise RuntimeError("B1 input frame expired")
        started = time.monotonic()
        stages = {}
        skips = {}
        errors = {}

        def run(role, *args, **kwargs):
            if not self._fresh(frame):
                raise RuntimeError("B1 frame expired before " + role)
            tick = time.monotonic()
            try:
                return self.networks.run(role, *args, **kwargs)
            finally:
                stages[role + "_ms"] = (
                    stages.get(role + "_ms", 0) + (time.monotonic() - tick) * 1000
                )

        detections = run("detection_segmentation", frame.bgr)
        for row in detections:
            row["range"] = range_from_mask(
                frame,
                row["mask"],
                self.config.min_depth_fraction,
                self.config.min_depth_pixels,
            )
        people = sorted(
            (r for r in detections if r["class_id"] == 0 and self._eligible(r)),
            key=lambda r: r["range"]["meters"],
        )[: self.config.max_pose_people]
        objects = sorted(
            (r for r in detections if r["class_id"] != 0 and self._eligible(r)),
            key=lambda r: -r["score"],
        )[: self.config.max_object_targets]
        for role, items in (("person_reid", people), ("object_embedding", objects)):
            for row in items:
                try:
                    result = run(role, frame.bgr, row["box_xyxy"])
                    row["_embedding"] = result["embedding"]
                    row["_embedding_space"] = result["embedding_space"]
                except ValueError as exc:
                    errors.setdefault(role, []).append(str(exc))
        tracked = self.entities.update(detections, frame)
        # Only arrays needed by this one frame remain in local variables.
        public = []
        for row in tracked:
            value = {
                k: v
                for k, v in row.items()
                if k not in ("mask", "_embedding", "_embedding_space")
            }
            value.update(
                type="person" if row["class_id"] == 0 else "object",
                mask_pixels=int(row["mask"].sum()),
                appearance_available=row.get("_embedding") is not None,
                frame={
                    "source_epoch": frame.source_epoch,
                    "frame_id": frame.frame_id,
                    "received_monotonic": frame.received_monotonic,
                },
            )
            public.append(value)
        persons = [r for r in public if r["class_id"] == 0]
        object_rows = [r for r in public if r["class_id"] != 0]
        pose_targets = sorted(
            (r for r in persons if r["track_id"] and self._eligible(r)),
            key=lambda r: r["range"]["meters"],
        )[: self.config.max_pose_people]
        poses = []
        hands = []
        for person in pose_targets:
            pose = run(
                "body_hand_pose",
                frame.bgr,
                person["box_xyxy"],
                joint_threshold=self.config.pose_joint_threshold,
            )
            for hand in pose.pop("hands"):
                hands.append(dict(hand, person_track_id=person["track_id"]))
            poses.append(
                dict(pose, target_id=person["track_id"], frame=frame.metadata())
            )
        faces = run("face", frame.bgr)[: self.config.max_face_targets]
        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face["box_xyxy"]]
            h, w = frame.bgr.shape[:2]
            mask = np.zeros((h, w), bool)
            mask[max(y1, 0) : min(y2, h), max(x1, 0) : min(x2, w)] = True
            face["range"] = range_from_mask(
                frame,
                mask,
                self.config.min_depth_fraction,
                self.config.min_depth_pixels,
            )
            if self._eligible(face):
                try:
                    identity = run("face_identity", frame.bgr, face["landmarks"])
                    space = (
                        "face_identity:"
                        + self.networks.manifest["models"]["face_identity"][
                            "onnx_sha256"
                        ]
                        + ":"
                        + self.networks.entries["face_identity"].get(
                            "embedding_space_suffix", "sface-aligned112-v1"
                        )
                    )
                    face["_embedding"] = identity["embedding"]
                    face["_embedding_space"] = space
                    face["identity"] = self.gallery.match(space, identity["embedding"])
                except ValueError as exc:
                    face["identity"] = {"status": "unavailable", "error": str(exc)}
            else:
                face["identity"] = {
                    "status": "not_triggered",
                    "reason": "outside_range_or_invalid_depth",
                }
        faces = self.faces.update(faces, frame)
        attributes = []
        public_faces = []
        vectors = {}
        for face in faces:
            target = face["track_id"]
            entry = {k: v for k, v in face.items() if not k.startswith("_")}
            center = (np.asarray(face["box_xyxy"][:2]) + face["box_xyxy"][2:]) / 2
            parents = [
                p["track_id"]
                for p in persons
                if p["track_id"]
                and p["box_xyxy"][0] <= center[0] <= p["box_xyxy"][2]
                and p["box_xyxy"][1] <= center[1] <= p["box_xyxy"][3]
            ]
            entry["person_track_id"] = parents[0] if len(parents) == 1 else None
            public_faces.append(entry)
            if target and face.get("_embedding") is not None:
                vectors[target] = (
                    face["_embedding"],
                    face["_embedding_space"],
                    frame.metadata(),
                )
            if not target or not self._eligible(face):
                continue
            item = {
                "target_id": target,
                "person_track_id": entry["person_track_id"],
                "frame": frame.metadata(),
            }
            try:
                head = run("head_pose", frame.bgr, face["box_xyxy"])
                points = run("gaze_landmarks", frame.bgr, face["box_xyxy"])
                item["head_pose"] = head
                item["gaze"] = run(
                    "gaze", frame.bgr, points["landmarks"], head["angles_degrees"]
                )
            except ValueError as exc:
                item["gaze"] = {"available": False, "error": str(exc)}
            try:
                item["expression"] = run("expression", frame.bgr, face["landmarks"])
            except ValueError as exc:
                item["expression"] = {"available": False, "error": str(exc)}
            attributes.append(item)
        if not self._fresh(frame):
            raise RuntimeError("B1 result expired before publication")
        prominent = public_faces[0]["box_xyxy"] if public_faces else None
        if self.tracker:
            self.tracker.process_box(frame, prominent)
        face_state = (
            self.tracker.snapshot()
            if self.tracker
            else {
                "face_detected": bool(prominent),
                "face_box": prominent,
                "frame": frame.metadata(),
            }
        )
        skips.update(
            pose_people_detected=len(persons),
            pose_people_processed=len(pose_targets),
            object_features_processed=sum(
                r["appearance_available"] for r in object_rows
            ),
            face_attributes_processed=len(attributes),
        )
        with self._lock:
            self._face_vectors = vectors
            selected = next(
                (r for r in public if r["track_id"] == self._selected), None
            )
            if selected:
                self._selected_last = deepcopy(selected)
            selection = {
                "track_id": self._selected,
                "state": "visible"
                if selected
                else "lost"
                if self._selected in self.entities.known_ids()
                else "expired"
                if self._selected
                else "none",
                "last_observation": deepcopy(self._selected_last),
            }
        stages.update(
            body_ms=stages.get("detection_segmentation_ms", 0),
            face_ms=stages.get("face_ms", 0),
            hands_ms=stages.get("body_hand_pose_ms", 0),
        )
        return {
            "ready": True,
            "error": None,
            "frame": frame.metadata(),
            "persons": persons,
            "objects": object_rows,
            "hands": hands[: self.config.max_hands],
            "poses": poses,
            "faces": public_faces,
            "attributes": attributes,
            "face": face_state,
            "selected_target": selection,
            "completed_monotonic": time.monotonic(),
            "elapsed_ms": (time.monotonic() - started) * 1000,
            "stages": stages,
            "task_counts": skips,
            "task_errors": errors,
            "providers": {
                "body": self.networks.models["detection_segmentation"].provider,
                "face": self.networks.models["face"].provider,
                "hands": self.networks.models["body_hand_pose"].provider,
                "depth": "RealSense SDK native",
            },
            "model_metrics": self.networks.snapshot(),
            "body_stages_ms": {},
        }

    def snapshot(self):
        with self._lock:
            result = deepcopy(self._state)
        frame = result.get("frame")
        if frame and (
            frame["source_epoch"] != self.camera.source_epoch
            or time.monotonic() - frame["received_monotonic"] > self.camera.stale_after
        ):
            result = self._empty("B1 result expired")
        return {
            **result,
            "processed": self.processed,
            "skipped_capture_frames": self.skipped,
            "failures": self.failures,
            "callback_error": self.callback_error,
            "backend": "tensorrt_b1",
            "model_set": self.networks.manifest.get("model_set", "B1"),
            "model_metrics": self.networks.snapshot(),
        }

    def enroll(self, target, label):
        with self._lock:
            value = (
                self._face_vectors.get(target)
                if self._state.get("ready")
                and any(f["track_id"] == target for f in self._state.get("faces", []))
                else None
            )
        if value is None:
            raise ValueError("No fresh face embedding for this target")
        vector, space, frame = value
        if (
            frame["source_epoch"] != self.camera.source_epoch
            or time.monotonic() - frame["received_monotonic"] > self.camera.stale_after
        ):
            raise ValueError("Face observation expired before enrollment")
        return self.gallery.enroll(label, space, vector)

    def select_target(self, target):
        with self._lock:
            observed = self._state.get("frame")
            if target is not None and (
                not observed
                or observed["source_epoch"] != self.camera.source_epoch
                or time.monotonic() - observed["received_monotonic"]
                > self.camera.stale_after
            ):
                raise ValueError("Target observation expired")
            visible = self._state.get("persons", []) + self._state.get("objects", [])
            if target is not None and not any(r["track_id"] == target for r in visible):
                raise ValueError("Target is not currently observed")
            self._selected = target
            self._selected_last = None
        return {"track_id": target}

    def close(self):
        # If the worker cannot join, its native contexts must remain owned.
        super().close()
        errors = []
        try:
            self.networks.close()
        except Exception as exc:
            errors.append(exc)
        if self.gallery:
            try:
                self.gallery.close()
            except Exception as exc:
                errors.append(exc)
            else:
                self.gallery = None
        with self._lock:
            self._face_vectors.clear()
            self._selected = self._selected_last = None
        self.entities.reset()
        self.faces.reset()
        if errors:
            raise ExceptionGroup("B1 resource release failed", errors)
