import torch

from nbeatsx_spread.training.provenance import environment_identity, sha256_json, state_dict_hash


def test_provenance_hashes_and_environment_identity():
    model = torch.nn.Linear(2, 1)
    assert len(state_dict_hash(model)) == 64
    assert sha256_json({"b": 1, "a": 2}) == sha256_json({"a": 2, "b": 1})
    env = environment_identity(device=torch.device("cpu"), deterministic=True, seed=42)
    assert env["precision"] == "float32" and env["amp"] is False

