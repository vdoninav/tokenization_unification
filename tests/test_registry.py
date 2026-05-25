from tokenstudy.data.registry import build_dataset


def test_registry_known_names():
    import pytest
    with pytest.raises((FileNotFoundError, OSError)):
        build_dataset(
            "ETTh1", split="train", lookback=336, horizon=96,
            data_path="/nonexistent/path.csv",
        )


def test_registry_unknown_name_raises():
    import pytest
    with pytest.raises(ValueError, match="Unknown dataset"):
        build_dataset("Imaginary", split="train", lookback=336, horizon=96, data_path="/x")


def test_registry_variate_count():
    from tokenstudy.data.registry import num_variates
    assert num_variates("ETTh1") == 7
    assert num_variates("Weather") == 21
    assert num_variates("Electricity") == 321
