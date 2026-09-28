from ctrisk.ml.registry import latest, load, next_version, save


def test_versions_count_up_from_one(tmp_path):
    assert next_version(tmp_path) == 1
    save(tmp_path, 1, {"model": "stub"}, metrics={"roc_auc": 0.7})
    save(tmp_path, 2, {"model": "stub2"}, metrics={"roc_auc": 0.71})
    assert (next_version(tmp_path), latest(tmp_path)) == (3, 2)


def test_round_trip(tmp_path):
    save(tmp_path, 1, {"model": "stub"}, metrics={"roc_auc": 0.7}, manifest={"clone": "TRIAL_FEATURES_V1"})
    model, docs = load(tmp_path, 1)
    assert model == {"model": "stub"}
    assert docs == {"metrics": {"roc_auc": 0.7}, "manifest": {"clone": "TRIAL_FEATURES_V1"}}
