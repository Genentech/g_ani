"""
(C) 2021 Genentech. All rights reserved.

Test file for ml_qm module

"""

import os
import sys

import pytest_check as check
import scripttest


def run_script(*args, **kwargs):
    """"Runs Python script in test environment"""
    env = scripttest.TestFileEnvironment("./test-outputs")
    return env.run(sys.executable, *args, **kwargs)


def run_help(pyscript):
    """Runs Python script with '--help' and generates output to be
       included into documentation
    """
    script = run_script(pyscript, '--help', expect_stderr=True)
    check.equal(0, script.returncode)
    help_desc = script.stdout.split('\n')

    pyscript_name = os.path.basename(pyscript)[:-3]
    help_file_name = os.path.join("docs", "text", pyscript_name + "-help.txt")
    with open(help_file_name, 'w') as help_file:
        for line in help_desc:
            help_file.write(line+'\n')
