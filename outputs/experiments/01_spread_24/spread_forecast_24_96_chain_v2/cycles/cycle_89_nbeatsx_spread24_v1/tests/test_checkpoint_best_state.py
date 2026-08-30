from nbeatsx_spread.training.early_stopping import EarlyStopping
from nbeatsx_spread.training.checkpoint import load_checkpoint, save_checkpoint
import torch


def test_early_stopping_keeps_best_and_stops_after_patience():
    stop = EarlyStopping(patience=2)
    assert stop.update(3.0)
    assert stop.best == 3.0
    assert stop.update(2.0)
    assert stop.best == 2.0
    assert not stop.update(2.5)
    assert not stop.should_stop
    assert not stop.update(2.6)
    assert stop.should_stop


def test_checkpoint_round_trip_restores_selected_state():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    path = __import__("pathlib").Path(__file__).parents[1] / "runs" / "test_checkpoint_roundtrip.pt"
    try:
        save_checkpoint(path, model, optimizer, 25, {"best_validation_mae": 1.5})
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(10.0)
        payload = load_checkpoint(path, model)
        assert payload["step"] == 25
        assert all(torch.equal(model.state_dict()[key], value) for key, value in expected.items())
    finally:
        path.unlink(missing_ok=True)
