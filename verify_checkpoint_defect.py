#!/usr/bin/env python3
"""
verify_checkpoint_defect.py — CPU-only defect audit for
`unsloth/Qwen3.8-27B-unsloth-bnb-4bit`.

Context: unslothai/unsloth issue #10276 — the pre-quantized 4-bit checkpoint
loads GatedDeltaNet projection layers with quant_state=None, so the first
forward crashes with
    RuntimeError: mat1 and mat2 shapes cannot be multiplied
while quantizing the base model at load works fine ("the artifact, not a
platform" — three independent hardware setups confirm).

This script never downloads the 22 GB checkpoint. It fetches only the
safetensors header via an HTTP Range request and audits every packed-4-bit
(uint8) weight tensor for the presence of its bitsandbytes quant-state
sibling keys in the file:

    <name>.weight.absmax
    <name>.weight.quant_map
    <name>.weight.nested_absmax
    <name>.weight.nested_quant_map
    <name>.weight.quant_state.bitsandbytes__nf4

Honest scope: the audit proves whether the quant metadata is present in the
artifact. It cannot by itself prove the loader re-attachment bug (that needs
a GPU forward pass); combined with the three independent crash reports on
issue #10276, a present-but-never-re-attached signature pinpoints the load
path as the suspect.

Usage: python3 verify_checkpoint_defect.py [--repo REPO_ID]
"""

import argparse
import json
import struct
import sys
import urllib.request

DEFAULT_REPO = "unsloth/Qwen3.8-27B-unsloth-bnb-4bit"

# Sibling keys bitsandbytes expects alongside a packed NF4 weight tensor.
SIBLING_SUFFIXES = [
    ".absmax",
    ".quant_map",
    ".nested_absmax",
    ".nested_quant_map",
    ".quant_state.bitsandbytes__nf4",
]


def fetch_header(repo_id: str) -> dict:
    url = (
        f"https://huggingface.co/{repo_id}/resolve/main/model.safetensors"
    )
    # The safetensors header is small (~400 KB here); grab 1 MiB and slice.
    req = urllib.request.Request(url, headers={"Range": "bytes=0-1048575"})
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except Exception as e:
        sys.exit(f"download failed: {e}")
    chunk = resp.read()
    if len(chunk) < 8:
        sys.exit("response too short to contain a safetensors header")
    (header_len,) = struct.unpack("<Q", chunk[:8])
    if len(chunk) < 8 + header_len:
        sys.exit(
            f"header is {header_len} bytes but only fetched {len(chunk) - 8}; "
            "increase the range window"
        )
    return json.loads(chunk[8 : 8 + header_len].decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    args = ap.parse_args()

    header = fetch_header(args.repo)
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    print(f"repo:            {args.repo}")
    print(f"total tensors:   {len(tensors)}")

    # Packed 4-bit weights are stored as uint8 with a matching quant-state
    # sibling set. Collect candidate bases by stripping the ".weight" suffix.
    packed = sorted(
        k[:-len(".weight")]
        for k, v in tensors.items()
        if k.endswith(".weight") and v.get("dtype") == "U8"
    )
    print(f"packed-4bit (U8) weight tensors: {len(packed)}")

    gated = [b for b in packed if "linear_attn" in b]
    print(f"GatedDeltaNet (linear_attn) packed layers: {len(gated)}\n")

    print("GatedDeltaNet layer audit (quant-state sibling keys present in file):")
    print(f"{'layer':70s} {' '.join(s.replace('.', '')[:9] for s in SIBLING_SUFFIXES)}  verdict")
    missing_any = 0
    for base in gated:
        w = base + ".weight"
        flags = []
        ok = True
        for sfx in SIBLING_SUFFIXES:
            present = (w + sfx) in tensors
            flags.append("Y" if present else "n")
            ok = ok and present
        if not ok:
            missing_any += 1
        short = base if len(base) <= 68 else "…" + base[-67:]
        print(f"  {short:68s} {' '.join(flags)}  {'PASS' if ok else 'FAIL'}")

    # Sanity: are ANY packed layers missing quant state in the file?
    missing_global = [
        b for b in packed
        if not all((b + ".weight" + s) in tensors for s in SIBLING_SUFFIXES)
    ]

    print(f"\nlinear_attn layers missing ≥1 quant key: {missing_any}/{len(gated)}")
    print(f"all packed layers missing ≥1 quant key:  {missing_global and len(missing_global) or 0}/{len(packed)}")
    if missing_global:
        for b in missing_global[:10]:
            print(f"  MISSING: {b}")

    print("\nVERDICT:")
    if not missing_global:
        print(
            "  Quant state IS present in the artifact for every packed-4bit "
            "layer, including all GatedDeltaNet projections.\n"
            "  The crash on issue #10276 (quant_state=None at first forward, "
            "3 independent hardware reports) therefore points at the "
            "pre-quantized LOAD path not re-attaching the state — not at a "
            "corrupt artifact. Suggested guard: fail checkpoint release CI "
            "unless every packed weight key re-attaches quant_state on load."
        )
        return 0
    print("  Quant state is missing from the artifact itself — the checkpoint "
          "needs regeneration.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
