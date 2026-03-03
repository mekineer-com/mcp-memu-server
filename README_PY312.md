# Python 3.12 (Alpine) install notes

memu-py 1.4.x on PyPI declares **Requires-Python >= 3.13**, so pip on Python 3.12 will refuse to install it.
If you want to stay on Python 3.12, install memu-py from source after lowering the `requires-python` gate.

## Steps (high level)

1) Unpack memu_py-1.4.0.tar.gz
2) Edit `memu_py-1.4.0/pyproject.toml`:
   - change `requires-python = ">=3.13"` to `>=3.12`
3) Build/install memu-py (maturin builds the Rust extension):
   - `pip install -U pip maturin`
   - `maturin develop --release`
   If you already have a broken `.so`, delete `src/memu/_core*.so` first.
4) Install this server without pulling deps:
   - `pip install -e . --no-deps`
   Then install FastAPI etc manually (or `pip install -r` if you add one).

## Reality check
Upstream officially targets Python 3.13+, so this is “best-effort”.
