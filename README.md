# Workerville

210 tasks × 16 configs. Python 3.10+, OpenClaw gateway and model endpoint required.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp api.config.example.yaml api.config.yaml
# Edit api.config.yaml.
python -m unittest discover -s tests
python scripts/run_experiment.py --config configs/tasks/ops_status_digest_maildrop_v1/C0__L3.yaml --api-config api.config.yaml --output-dir runs/example
```

Judge: `judge/`. MIT: `LICENSE`.
