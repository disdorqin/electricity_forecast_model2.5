from nbeatsx_spread.training.scheduler import scheduled_learning_rate


def test_nominal_milestones_are_not_compressed():
    f = lambda step: scheduled_learning_rate(5e-4, step, 1200, 0.5, 3, (300, 600, 900))
    assert f(1) == 5e-4
    assert f(299) == 5e-4
    assert f(300) == 2.5e-4
    assert f(599) == 2.5e-4
    assert f(600) == 1.25e-4
    assert f(900) == 6.25e-5
