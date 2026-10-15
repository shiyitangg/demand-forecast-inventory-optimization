import numpy as np
import pytest

from src.folds import make_folds, split_masks


def test_folds_are_ordered_and_end_at_last_week():
    folds = make_folds(277, horizon=4, n_folds=4, step=4)
    assert [f.fold for f in folds] == [1, 2, 3, 4]
    assert folds[-1].test_end == 277
    for a, b in zip(folds, folds[1:]):
        assert a.test_end <= b.test_start


def test_train_ends_where_test_starts():
    for f in make_folds(277):
        assert f.train_end == f.test_start
        assert f.test_end - f.test_start == 4


def test_only_last_fold_is_held_out_from_tuning():
    flags = [f.use_for_tuning for f in make_folds(277)]
    assert flags == [True, True, True, False]


def test_split_masks_do_not_overlap():
    weeks = np.arange(277)
    for f in make_folds(277):
        train, test = split_masks(weeks, f)
        assert not (train & test).any()
        assert weeks[train].max() < weeks[test].min()
        assert test.sum() == 4


def test_overlapping_windows_rejected():
    with pytest.raises(ValueError):
        make_folds(277, horizon=4, step=2)
