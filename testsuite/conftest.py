def pytest_addoption(parser):
    parser.addoption("--api-key", action="store", default=None, help="Optional API key")
    parser.addoption("--timestamp", action="store", default=None, help="Timestamp of the test run")

def pytest_configure(config):
    import builtins
    builtins.VECTORX_API_KEY = config.getoption("--api-key")
    builtins.TEST_RUN_TIMESTAMP = config.getoption("--timestamp")
