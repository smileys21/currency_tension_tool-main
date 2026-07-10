import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "network: hits live keyless endpoints (OECD SDMX, BoE)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("-m"):
        return
    skip = pytest.mark.skip(reason="network test; run with -m network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
