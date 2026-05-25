def test_package_imports():
    import tokenstudy
    from tokenstudy import data, tokenizers, models, training, eval
    assert tokenstudy.__name__ == "tokenstudy"
