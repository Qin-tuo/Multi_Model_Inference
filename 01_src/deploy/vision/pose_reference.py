# SPDX-License-Identifier: Apache-2.0
"""Original MMPose CPU reference, isolated from the deployed application.

Import the pinned inference modules directly, avoiding package aggregators that
also import unrelated training datasets/compiled operators. All network layers,
affine transforms, normalization and SimCC decoding remain upstream code.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[3]


def load_reference():
    import numpy as np
    import torch

    entry = json.loads((ROOT / "03_models/candidates/vision-b1.json").read_text())[
        "models"
    ]["body_hand_pose"]
    source = ROOT / "05_runtime/frameworks/b1/mmpose"
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if revision != entry["upstream_revision"] or dirty:
        raise ValueError("MMPose reference differs from the clean pinned revision")
    weights = entry["source_checkpoint"]
    path = ROOT / weights["path"]
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != weights["sha256"] or path.stat().st_size != weights["bytes"]:
        raise ValueError("Original pose checkpoint checksum mismatch")

    sys.path[:0] = [str(source), str(source / "projects/rtmpose3d")]
    namespaces = [
        "mmpose.models",
        "mmpose.models.backbones",
        "mmpose.models.necks",
        "mmpose.models.heads",
        "mmpose.evaluation",
        "mmpose.codecs",
        "mmpose.datasets",
        "mmpose.datasets.transforms",
        "rtmpose3d",
    ]
    for name in namespaces:
        if name in sys.modules:
            raise RuntimeError("Pose reference requires a fresh worker process")
        package = types.ModuleType(name)
        folder = (
            source / name.replace(".", "/")
            if name != "rtmpose3d"
            else source / "projects/rtmpose3d/rtmpose3d"
        )
        package.__path__ = [str(folder)]
        sys.modules[name] = package
    from mmpose.models.heads.base_head import BaseHead

    sys.modules["mmpose.models.heads"].BaseHead = BaseHead
    from mmpose.models.backbones.cspnext import CSPNeXt
    from mmpose.models.necks.cspnext_pafpn import CSPNeXtPAFPN
    from mmpose.models.data_preprocessors import PoseDataPreprocessor
    from mmpose.datasets.transforms.common_transforms import GetBBoxCenterScale
    from mmpose.datasets.transforms.topdown_transforms import TopdownAffine
    from mmpose.structures import PoseDataSample
    from rtmpose3d.rtmw3d_head import RTMW3DHead
    from rtmpose3d.simcc_3d_label import SimCC3DLabel

    config_file = (
        source / "projects/rtmpose3d/configs/rtmw3d-x_8xb32_cocktail14-384x288.py"
    )
    config = deepcopy(runpy.run_path(str(config_file))["model"])

    def options(name):
        return {k: v for k, v in config[name].items() if k != "type"}

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = CSPNeXt(**options("backbone"))
            self.neck = CSPNeXtPAFPN(**options("neck"))
            head = options("head")
            # Training losses and the codec contain no inference parameters.
            # Strict checkpoint loading below rejects any overlooked state.
            head.update(loss=[], decoder=None)
            self.head = RTMW3DHead(**head)

        def forward(self, inputs):
            # BasePoseEstimator._forward / extract_feat use this exact order.
            return self.head(self.neck(self.backbone(inputs)))

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    with torch.serialization.safe_globals(
        [
            np.core.multiarray._reconstruct,
            np.core.multiarray.scalar,
            np.ndarray,
            np.dtype,
            type(np.dtype(np.float64)),
            type(np.dtype(np.float32)),
            type(np.dtype(np.int64)),
            type(np.dtype(np.uint8)),
        ]
    ):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    model = Network().float().eval()
    loaded = model.load_state_dict(state, strict=True)
    processor = PoseDataPreprocessor(**options("data_preprocessor"))
    codec = SimCC3DLabel(
        **{k: v for k, v in config["head"]["decoder"].items() if k != "type"}
    )
    bbox, affine = GetBBoxCenterScale(), TopdownAffine(input_size=(288, 384))

    def prepare(image, box):
        values = affine(bbox({"img": image, "bbox": np.array(box, np.float32)[None]}))
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(values["img"].transpose(2, 0, 1))
        )
        tensor = processor(
            {"inputs": [image_tensor], "data_samples": [PoseDataSample()]},
            training=False,
        )["inputs"]
        return tensor, values["bbox_center"][0], values["bbox_scale"][0]

    provenance = {
        "revision": revision,
        "checkpoint_sha256": digest,
        "strict_loaded_state_keys": len(state),
        "missing_keys": loaded.missing_keys,
        "unexpected_keys": loaded.unexpected_keys,
        "namespaces_without_aggregators": namespaces,
        "forward": "unmodified CSPNeXt -> CSPNeXtPAFPN -> RTMW3DHead; raw SimCC",
        "source_files": {},
    }
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if (
            filename
            and Path(filename).resolve().is_relative_to(source)
            and Path(filename).suffix == ".py"
        ):
            provenance["source_files"][str(Path(filename).relative_to(source))] = (
                hashlib.sha256(Path(filename).read_bytes()).hexdigest()
            )
    for filename in (config_file, source / "mmpose/models/pose_estimators/base.py"):
        provenance["source_files"][str(filename.relative_to(source))] = hashlib.sha256(
            filename.read_bytes()
        ).hexdigest()
    return model, prepare, codec, provenance
