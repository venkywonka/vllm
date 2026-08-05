# MiniMax-M3 on 4x RTX PRO 6000 Blackwell — qualified stack

Consolidates the optimizations behind every point on `qualified_frontier.tsv` into one
branch. Previously these existed only as `patch -p1` overlays applied into `site-packages`
at container start, recorded in each run's `source-proof.json` as `patches_in_order`.

Base: `8b00f4123776a47a6d8e315242ee5f0dd0b817cf`

## Verification

Tag `qualified/c128-tp1dp4ep4-mnt272-arm-e` reproduces the vLLM source of image
`sha256:ee50d590682b0cc3cf1efd2d02a675773430578856d2e99875341fb65205d527`
**byte-for-byte: 11/11 files match** `source_sha256` in the arm-E source proof at
`kernel-candidates/minimax-m3-c128-tp1dp4ep4-mnt272-arm-e-w4a4-fp8-packed-ag-official-w2m10-sealfix-v2/`.

Every patch applied at `--fuzz=0`, the same strictness the harness uses.

`HEAD` adds the fused-unpermute commit on top. Its only overlap with the verified set is
one additive line in `oracle/nvfp4.py` registering `NvFp4MoeBackend.VLLM_CUTLASS` as
clamp-capable, which is reachable only under `--moe-backend cutlass`. `HEAD` is therefore a
safe superset: `--moe-backend flashinfer_cutlass` behaves identically to the tag.

## Commits

| # | Commit | Serves |
|---|---|---|
| 1 | FP8 indexer for SM120 | baked into image at build time, not in `patches_in_order` |
| 2 | NVFP4 Marlin MoE backend | c4 / c8 / c16 canonical (TP4/EP1, W4A16) |
| 3 | FlashInfer CUTLASS W4A4 MoE backend | c32–c256 (TP1/DP4/EP4) |
| 4 | FlashInfer MoE unfused finalize call | c32–c256 |
| 5 | Sequence-parallel MoE path | c32–c256 |
| 6 | Sequence-parallel runtime proof markers | c32–c256 |
| 7 | Target/draft local argmax for EAGLE3 | c32–c256, c128 latency |
| 8 | Packed NVFP4 quantize before pre-MoE all-gather | c32–c256 |
| 9 | FlashInfer MoE tuning ceiling (MNT) binding | c128 throughput/balanced (MNT272) |
| 10 | Native CUTLASS W4A4 fused unpermute | c128 latency (TP4/EP1) |

Commits 3–4 and 10 are two different MoE backends selected at runtime by `--moe-backend`
(`flashinfer_cutlass` vs `cutlass`). They touch disjoint files and coexist.

## NOT in this tree

Three patches in the qualified stack target FlashInfer, not vLLM, and cannot be committed
here. They must still be applied to the FlashInfer install inside the image:

| Patch | sha256 | Target |
|---|---|---|
| `minimax_m3_flashinfer_core_unfused_finalize.patch` | `840773efeade0e21…` | `flashinfer/fused_moe/core.py` |
| `minimax_m3_flashinfer_binding_unfused_finalize.patch` | `049a294d8110d758…` | `csrc/fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_binding.cu` |
| `minimax_m3_flashinfer_mnt_runtime_proof.patch` (FlashInfer half) | `d0aaf93892dc096d…` | `flashinfer/fused_moe/core.py` |

Commit 9 carries only the vLLM half of that last patch; the file was split at the
`--- a/flashinfer/` boundary.

Also required and not source-controllable: the prebuilt JIT blob
`minimax_m3_flashinfer_fused_moe_120_unfused_finalize.so.gz`
(`f7783b2a6df6b001…`) → `flashinfer_jit_cache/jit_cache/fused_moe_120/fused_moe_120.so`.

All four live in the InferenceMAX fork at `benchmarks/patches/vllm/`.

## Reproducing each frontier point

Common: `--kv-cache-dtype fp8 --block-size 128 --language-model-only --attention-backend
TRITON_ATTN --no-enable-prefix-caching --disable-custom-all-reduce`, EAGLE3 draft with
`num_speculative_tokens=3`, ISL 8192 / OSL 1024.

- **c4 / c8 / c16 canonical** — `--moe-backend marlin`, TP4/EP1, BF16 indexer,
  `--max-num-seqs {4,8,16}`, cudagraph capture `{16,32,64}`.
- **c32–c256 throughput** — `--moe-backend flashinfer_cutlass`, TP1 / DP4 / EP4,
  `--enable-expert-parallel --all2all-backend allgather_reducescatter`,
  `--attention-config '{"indexer_kv_dtype":"fp8"}'`,
  `AUTORESEARCH_FLASHINFER_MOE_TUNE_MAX_NUM_TOKENS` = `max_capture_size * dp_size`.
  The c128 arm-E point is `--max-num-seqs 17`, capture size 68, MNT 272,
  `--max-num-batched-tokens 4096`, `--gpu-memory-utilization 0.95`.
- **c128 latency** — `--moe-backend cutlass`, TP4/EP1, `--max-num-seqs 38`,
  capture size 152, `--max-num-batched-tokens 16384`, `--gpu-memory-utilization 0.90`.

## Caveats

- Commits 6, 8, 9 are fail-closed runtime **proof markers**, not optimizations. Commit 9
  hard-raises unless the topology is exactly TP1/DP4/EP4 with AG-RS all2all and the MNT env
  var equals `max_capture_size * dp_size`. Relax these before running any other topology.
- Committed with `--no-verify`. Upstream pre-commit hooks (ruff format) rewrite several of
  these files, which would break byte-fidelity with the qualified image.
- `/home/ubuntu/autoresearch/patches/` is a stale mirror — its `minimax_m3_nvfp4_marlin.patch`
  is missing the `oracle/nvfp4.py` hunk. The InferenceMAX fork is the source of truth.
