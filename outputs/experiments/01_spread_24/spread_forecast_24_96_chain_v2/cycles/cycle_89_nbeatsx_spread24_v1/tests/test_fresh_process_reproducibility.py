import os
import subprocess
import sys
from pathlib import Path


def test_fresh_process_model_initial_state_reproducibility():
    root = Path(__file__).parents[1]
    code = (
        "import json; from pathlib import Path; "
        "from nbeatsx_spread.model.factory import build_model; "
        "from nbeatsx_spread.training.reproducibility import seed_everything; "
        "from nbeatsx_spread.training.provenance import state_dict_hash; "
        "c=json.loads(Path('configs/business_strict34_core.json').read_text(encoding='utf-8')); "
        "seed_everything(42); print(state_dict_hash(build_model(c)))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    values = [subprocess.check_output([sys.executable, "-c", code], cwd=root, env=env, text=True).strip() for _ in range(2)]
    assert values[0] == values[1]

