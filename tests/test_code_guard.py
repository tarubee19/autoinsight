import pytest

from autoinsight import templates
from autoinsight.code_guard import check_code


def test_vetted_templates_pass_the_guard():
    for name in (templates.ENGINEER, templates.ANALYST):
        result = check_code(templates.source(name))
        assert result.ok, result.violations


def test_plain_pandas_code_is_allowed():
    code = "import pandas as pd\nimport numpy as np\nfrom scipy import stats\ndf = pd.read_csv(INPUT_PATH)\nsave_json({'n': len(df)}, 'o.json')\n"
    assert check_code(code).ok


@pytest.mark.parametrize(
    "code, needle",
    [
        ("import os\nos.system('ls')", "import of 'os'"),
        ("import subprocess", "subprocess"),
        ("from socket import socket", "socket"),
        ("import requests", "requests"),
        ("import pickle", "pickle"),
        ("exec('print(1)')", "exec"),
        ("eval('1+1')", "eval"),
        ("open('x.txt', 'w')", "open"),
        ("x = ().__class__.__bases__", "dunder"),
        ("__import__('os')", "__import__"),
        ("import pandas as pd\npd.read_csv('/etc/passwd')", "absolute path"),
        ("import pandas as pd\npd.read_csv('http://evil.example/x.csv')", "URL"),
        ("import pandas as pd\npd.read_pickle('x')", "read_pickle"),
        ("getattr(object, 'x')", "getattr"),
        ("import importlib", "importlib"),
        ("from . import x", "not allowed"),
        ("def f(:\n  pass", "SyntaxError"),
    ],
)
def test_dangerous_code_is_rejected(code, needle):
    result = check_code(code)
    assert not result.ok
    assert needle.lower() in result.message().lower()


def test_relative_filenames_are_fine():
    assert check_code("import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nsave_fig(fig, 'a.png')").ok
