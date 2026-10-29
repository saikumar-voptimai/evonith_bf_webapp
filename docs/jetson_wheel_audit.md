# Jetson wheel audit (SV-6.2, #316)

Does `uv sync --frozen` install the locked packages on the BF-02 Jetson AGX Orin from wheels? JetPack 6 is Ubuntu
22.04 on aarch64 with glibc 2.35; the project runs on CPython 3.12 installed by uv (python-build-standalone,
`deploy/jetson/install_python.sh`), never the system Python. The target of the lock check is therefore
`aarch64-manylinux_2_35` and CPython 3.12 (`JETSON_PLATFORM` in the Makefile).

**Method.** `scripts/jetson_wheel_audit.py` runs the recipe of `make lock-check` by hand with uv 0.7.13 (the lock
rule's version): `uv lock --check`, `uv export --frozen --no-hashes --no-emit-project --all-groups`, then
`uv pip compile` of the export with `--no-deps --python-version 3.12 --python-platform aarch64-manylinux_2_35
--only-binary :all: --no-binary pypika --no-binary quickjs` (the source builds of `JETSON_SOURCE_BUILDS`). For every
resolved package it then picks the file uv installs: the locked wheel whose best tag ranks first for CPython 3.12 on
manylinux_2_35_aarch64 (CPython ABI before abi3 before pure Python, the newest manylinux first), else the sdist.
Re-run it after a lock change:

```bash
python scripts/jetson_wheel_audit.py --write     # or: make wheel-audit
```

The block below is generated; edit the notes in `AUDIT` of the script, not here.

<!-- wheel-audit:start -->
Recipe of `make lock-check` run by hand on 29 Oct 2026 with uv 0.7.13: `uv lock --check` passed; `uv pip compile` resolved 244 packages for `aarch64-manylinux_2_35` and CPython 3.12 with `--only-binary :all:` except pypika, quickjs: no problem.

| File installed on the Jetson | Packages |
|---|---|
| CPython 3.12 aarch64 wheel | 48 |
| abi3 aarch64 wheel | 9 |
| pure-Python wheel | 185 |
| source build (sdist) | 2 |

Highest glibc floor of the selected wheels: 2.34 (cryptography); the Jetson has 2.35. Locked but not installed on linux-aarch64 (markers for other platforms): colorama, nvidia-cublas-cu12, nvidia-cuda-cupti-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-runtime-cu12, nvidia-cudnn-cu12, nvidia-cufft-cu12, nvidia-curand-cu12, nvidia-cusolver-cu12, nvidia-cusparse-cu12, nvidia-nccl-cu12, nvidia-nvjitlink-cu12, nvidia-nvtx-cu12, pyreadline3, pywin32. Local packages (built from this repository): furnace-data.

| Package | Version | Required by | File kind | File installed on the Jetson | glibc floor | Notes |
|---|---|---|---|---|---|---|
| torch | 2.2.2 | project, sentence-transformers | CPython 3.12 aarch64 wheel | `torch-2.2.2-cp312-cp312-manylinux2014_aarch64.whl` | 2.17 | CPU build: its CUDA libraries (`nvidia-*-cu12`) are locked for x86_64 Linux only, so none installs here. The Orin GPU would need NVIDIA's JetPack wheel (not used) |
| highspy | not locked | — | — | — | — | Not a dependency. The LPs (burden engine, BMO) call HiGHS through `scipy.optimize.linprog(method="highs")`; HiGHS is compiled into the scipy wheel |
| scipy | 1.15.0 | project, scikit-learn, sentence-transformers | CPython 3.12 aarch64 wheel | `scipy-1.15.0-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` | 2.17 | Carries HiGHS (see highspy) |
| numpy | 1.26.4 | project and 23 packages | CPython 3.12 aarch64 wheel | `numpy-1.26.4-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` | 2.17 | — |
| psycopg2-binary | 2.9.11 | project, furnace-data | CPython 3.12 aarch64 wheel | `psycopg2_binary-2.9.11-cp312-cp312-manylinux2014_aarch64.manylinux_2_17_aarch64.whl` | 2.17 | libpq and OpenSSL bundled in the wheel |
| duckdb | 1.1.3 | project, pygwalker | CPython 3.12 aarch64 wheel | `duckdb-1.1.3-cp312-cp312-manylinux_2_24_aarch64.manylinux_2_28_aarch64.whl` | 2.28 | — |
| pyarrow | 18.1.0 | influxdb3-python, project, streamlit, pygwalker | CPython 3.12 aarch64 wheel | `pyarrow-18.1.0-cp312-cp312-manylinux_2_28_aarch64.whl` | 2.28 | Arrow Flight client of InfluxDB 3 |
| influxdb3-python | 0.10.0 | project, furnace-data | pure-Python wheel | `influxdb3_python-0.10.0-py3-none-any.whl` | — | Pure Python on pyarrow |
| chromadb | 0.6.1 | project | pure-Python wheel | `chromadb-0.6.1-py3-none-any.whl` | — | Pure Python; its compiled dependencies are the next rows |
| chroma-hnswlib | 0.7.6 | chromadb | CPython 3.12 aarch64 wheel | `chroma_hnswlib-0.7.6-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` | 2.17 | — |
| onnxruntime | 1.20.1 | chromadb | CPython 3.12 aarch64 wheel | `onnxruntime-1.20.1-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl` | 2.28 | CPU execution provider (default embedding function) |
| tokenizers | 0.22.2 | chromadb, transformers, voyageai | abi3 aarch64 wheel | `tokenizers-0.22.2-cp39-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` | 2.17 | — |
| pypika | 0.48.9 | chromadb | source build (sdist) | `PyPika-0.48.9.tar.gz` | — | No wheel at all: sdist, pure Python (Makefile `JETSON_SOURCE_BUILDS`) |
| sentence-transformers | 2.6.1 | project | pure-Python wheel | `sentence_transformers-2.6.1-py3-none-any.whl` | — | Pure Python on torch and transformers |
| safetensors | 0.7.0 | transformers (sentence-transformers) | abi3 aarch64 wheel | `safetensors-0.7.0-cp38-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` | 2.17 | — |
| scikit-learn | 1.6.1 | project, sentence-transformers | CPython 3.12 aarch64 wheel | `scikit_learn-1.6.1-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` | 2.17 | — |
| quickjs | 1.19.4 | project, pygwalker | source build (sdist) | `quickjs-1.19.4.tar.gz` | — | No aarch64 wheel: C extension compiled from the sdist during `uv sync` (needs build-essential; Makefile `JETSON_SOURCE_BUILDS`) |
<!-- wheel-audit:end -->

**What the audit does not cover.** It shows that a binary file exists for the Jetson and which one uv takes; the
on-device proof is the import check of `install_python.sh --release DIR` (every package above imported in the synced
`.venv`, one HiGHS LP through scipy), and the full test suite and `deploy/jetson/benchmark.sh` on the device (pending:
device). Two consequences of the files above:

- **torch runs on the CPU.** PyPI's aarch64 wheel of torch 2.2.2 has no CUDA; the CUDA libraries in the lock carry
  `platform_machine == 'x86_64'` markers. The models that use torch (sentence-transformers embeddings of the shift
  summaries) are small enough for the Orin's CPU. NVIDIA's JetPack wheel of torch (CUDA on the Orin GPU) is not on
  PyPI, is built for its own torch version and would leave the lock; it is not used.
- **Two source builds.** `quickjs` (pygwalker's dependency) compiles a C extension during `uv sync`, so the Jetson
  needs `build-essential` (`install_python.sh` installs it); `pypika` (chromadb's) is pure Python packed as an sdist
  only. Both build once per uv cache.
