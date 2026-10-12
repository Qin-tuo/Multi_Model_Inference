"""Project-local overlay for NVIDIA CCCL CUDA13.2 issue8833 (no system edits)."""
import hashlib
import json
from pathlib import Path

SOURCE = Path("/usr/local/cuda-13.2/targets/sbsa-linux/include/cccl/cub/device/device_transform.cuh")
OLD = "template <typename T>\nstruct ::cuda::proclaims_copyable_arguments<CUB_NS_QUALIFIER::detail::__return_constant<T>> : ::cuda::std::true_type\n{};"
NEW = "namespace cuda {\ntemplate <typename T>\nstruct proclaims_copyable_arguments<CUB_NS_QUALIFIER::detail::__return_constant<T>> : ::cuda::std::true_type\n{};\n} // namespace cuda"


def prepare_overlay(cache):
    content = SOURCE.read_text()
    if OLD not in content:
        if "struct ::cuda::proclaims_copyable_arguments" in content:
            raise RuntimeError("CUDA CCCL bug pattern changed; review before applying compatibility overlay")
        return None
    if content.count(OLD) != 1: raise RuntimeError("Unexpected number of CCCL specializations")
    expected = json.loads((Path(__file__).with_name("source-lock.json")).read_text())["cccl_compat"]["system_header_sha256"]
    if hashlib.sha256(content.encode()).hexdigest() != expected:
        raise RuntimeError("CCCL source header changed; review and update its pin before applying the overlay")
    root = cache / "sdk/cuda-cccl-overlay"
    output = root / "cub/device/device_transform.cuh"
    output.parent.mkdir(parents=True, exist_ok=True)
    patched = content.replace(OLD, NEW)
    output.write_text(patched)
    report = {"system_header": str(SOURCE), "system_header_unchanged": True,
        "source_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "patched_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "upstream_issue": "https://github.com/NVIDIA/cccl/issues/8833",
        "scope": "template specialization namespace spelling only; no algorithm change"}
    (root / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return root
