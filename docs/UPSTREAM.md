# Upstream

## Base

- Upstream: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), fork base
  [unslothai/llama.cpp](https://github.com/unslothai/llama.cpp) (`origin`, a blob-less partial clone; the
  dsv41 branch carries the DeepSeek-V4.1 model code).
- Versions in this tree: llama.cpp `0.7.0-dev`, ggml `0.27.0`.
- Branch `slim` starts at `ista-metal` `a885ee4af`, which already removed every non-Metal ggml backend.
  `slim` removes the rest of what this fork does not build and rewrites the docs.

## Removed

### ggml backends (`ggml/src/`)

| Backend | Directory | Why |
|---|---|---|
| CUDA | `ggml-cuda/` | NVIDIA GPUs |
| Vulkan | `ggml-vulkan/` | other GPUs |
| SYCL | `ggml-sycl/` | Intel GPUs |
| HIP | `ggml-hip/` | AMD GPUs |
| MUSA | `ggml-musa/` | Moore Threads GPUs |
| CANN | `ggml-cann/` | Ascend NPUs |
| OpenCL | `ggml-opencl/` | Adreno GPUs |
| OpenVINO | `ggml-openvino/` | Intel CPUs/GPUs |
| RPC | `ggml-rpc/` | remote compute |
| VirtGPU | `ggml-virtgpu/` | virtual GPU |
| WebGPU | `ggml-webgpu/` | browser GPUs |
| ET | `ggml-et/` | embedded |
| Hexagon | `ggml-hexagon/` | Snapdragon DSP |
| ZenDNN | `ggml-zdnn/` | IBM Z |

Kept: CPU (required - experts sit in CPU buffers), Metal, BLAS/Accelerate (macOS default).

### Tools (`tools/`)

`gguf-split`, `imatrix`, `quantize`, `tokenize`, `tts`, `cvector-generator`, `export-lora`, `fit-params`,
`tuning`, `results`. The driver scripts in `work/bench` and `work/tools` only call `llama-server`,
`llama-bench`, `llama-perplexity`, `llama-completion`, `llama-cli` and `test-backend-ops`, all of which stay.

### Top level

`app/`, `benches/`, `ci/`, `conversion/`, `convert_*.py`, `docs/` (replaced by ours), `examples/`, `pocs/`,
`grammars/`, `media/`, `models/`, `skills/`, `gguf-py/`, `flak*`, `Makefile`, `CMakePresets.json`,
`CODEOWNERS`, upstream `CONTRIBUTING.md` / `SECURITY.md` / `AGENTS.md` / `CLAUDE.md`, Python tooling
(`requirements*`, `pyproject.toml`, `mypy.ini`, `pyrightconfig.json`, `ty.toml`), dot-folders
(`.devops/`, `.github/`, `.gemini/`, `.pi/`), dotfiles (`.flake8`, `.pre-commit-config.yaml`,
`.dockerignore`, `.gitmodules` - it was empty and no submodules exist).

`scripts/` keeps only `ui-assets.cmake` (invoked by `tools/ui/CMakeLists.txt`); everything else is gone.

### Tests

`tests/` keeps `test-backend-ops.cpp` and `CMakeLists.txt` only. The CMake file is a two-target shim that
builds `test-backend-ops` and nothing else.

## Stale upstream reference

`tools/ui/sources.cmake` lists `scripts/vite-plugin-llama-cpp-build.ts` as a UI input. Upstream deleted that
file in `f7ca93d12` ("ui: PWA support") and kept the reference. The file never existed in `a885ee4af`; the
last version is from `efbacf8d2`. It is restored at `tools/ui/scripts/vite-plugin-llama-cpp-build.ts` so the
input list is valid.

## Kept

- `ggml/`: base, CPU, Metal (plus BLAS on macOS)
- `src/`, `include/`, `common/`, `vendor/`, `cmake/`, root `CMakeLists.txt`
- `LICENSE`, `licenses/`, `AUTHORS`
- `tools/`: `server`, `cli`, `completion`, `llama-bench`, `perplexity`, `batched-bench`, `mtmd`, `ui`
- `tests/`: `test-backend-ops`

`build-slim/` is not tracked: it was committed by mistake and is now removed from the index and covered by
`build*/` in `.gitignore`.

## Merging newer llama.cpp

```bash
git remote add upstream https://github.com/ggml-org/llama.cpp.git   # once
git fetch upstream
git rebase upstream/master
```

Conflicts land only in the files we edited, not in the deleted ones:

- `CMakeLists.txt` - the `examples`/`pocs`/`app` subdirectory blocks and the `LLAMA_BUILD_EXAMPLES` /
  `LLAMA_BUILD_APP` options are gone.
- `tools/CMakeLists.txt` - only the kept tools are listed.
- `ggml/CMakeLists.txt` - `GGML_BUILD_EXAMPLES` and the MSVC block are gone.
- `tests/CMakeLists.txt` - replaced by the shim; take ours.

Then rebuild and test:

```bash
cmake -S . -B build-slim -G Ninja -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-slim -j
build-slim/bin/test-backend-ops -o MUL_MAT_ID -b MTL0
```

Possible future work: dropping the unused model architectures from `src/llama-model.cpp` and the arch tables.
That is deep surgery in the arch registry and was deliberately not done.
