# Unsloth checkpoint defect audit

CPU-only audit for [unslothai/unsloth#10276](https://github.com/unslothai/unsloth/issues/10276):
Unsloth's pre-quantized `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` checkpoint loads
GatedDeltaNet projection layers with `quant_state=None`, so the first forward
crashes with `RuntimeError: mat1 and mat2 shapes cannot be multiplied`.
Three independent hardware setups (CUDA, Windows/RTX 3090, ROCm) confirm it,
while quantizing the base model at load works fine — the artifact, not the
platform. The issue is still open (last activity 2026-09-16).

## What this does

`verify_checkpoint_defect.py` downloads only the ~400 KB safetensors **header**
(via HTTP Range request — never the 22 GB checkpoint) and checks every packed
4-bit (uint8) weight tensor for the presence of its bitsandbytes quant-state
sibling keys (`absmax`, `quant_map`, `nested_absmax`, `nested_quant_map`,
`quant_state.bitsandbytes__nf4`).

## Verified results (run 2026-09-16, live checkpoint)

```
repo:            unsloth/Qwen3.8-27B-unsloth-bnb-4bit
total tensors:   2944
packed-4bit (U8) weight tensors: 352
GatedDeltaNet (linear_attn) packed layers: 96   (48 transformer layers x in_proj_z + out_proj)

linear_attn layers missing ≥1 quant key: 0/96
all packed layers missing ≥1 quant key:  0/352

VERDICT: Quant state IS present in the artifact for every packed-4bit layer,
including all GatedDeltaNet projections. The crash on issue #10276
(quant_state=None at first forward, 3 independent hardware reports) therefore
points at the pre-quantized LOAD path not re-attaching the state — not at a
corrupt artifact.
```

## Honest scope

This audit proves the quant metadata exists in the artifact; it cannot by
itself prove the loader bug — that needs a GPU forward pass (not available
on this machine). Combined with the three independent crash reports, a
present-but-never-re-attached signature pinpoints the load path as the
suspect. The suggested guard for Unsloth's checkpoint-release CI: fail the
release unless every packed weight key re-attaches `quant_state` on load.

## Run it

```bash
python3 verify_checkpoint_defect.py
```
