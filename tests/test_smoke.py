import actionlockin


def test_package_importable():
    assert isinstance(actionlockin.__version__, str)
