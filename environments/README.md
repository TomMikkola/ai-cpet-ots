# Preprocessing environment

Use **64-bit Windows / Python 3.7.4**, separately from the Python 3.11 model environment. The pinned dependencies reconstruct the legacy environment; the original dependency export is unavailable.

From the repository root, replace the executable placeholder below:

```powershell
& '<python-3.7.4-executable>' -m venv .venv-preprocess
.venv-preprocess/Scripts/python.exe -m pip install pip==24.0
.venv-preprocess/Scripts/python.exe -m pip install --require-hashes --only-binary=:all: -r environments/preprocess-legacy.txt
.venv-preprocess/Scripts/python.exe -m pip check
.venv-preprocess/Scripts/python.exe -m ipykernel install --prefix .venv --name ai-cpet-preprocess-legacy --display-name 'Python 3.7.4 (legacy preprocessing)'
```

Keep both environments. Start JupyterLab from the main `.venv`; select the legacy kernel only for [preprocess.ipynb](../notebooks/preprocess.ipynb). Do not install the main package into the legacy environment.

Follow the notebook's input and execution instructions. Review `run-report.json` before modeling; do not use incomplete runs. Set `AI_CPET_DATA_DIR` in the Python 3.11 kernel to the generated directory and verify participant alignment.
