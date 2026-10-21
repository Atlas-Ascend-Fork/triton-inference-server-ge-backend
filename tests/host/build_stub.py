# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared harness for the build.sh host regressions.

Both regressions drive the real script inside a temporary tree with stub
``python3``/``cmake``/``make`` executables, so the setup lives here instead of
being copied into every test module.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

BUILD = Path(__file__).resolve().parents[2] / "build.sh"


def run_build(tmp_path, *, spaced=False, abi="True", python_exit=0, extra_args=()):
    """Run build.sh up to the CMake invocation and capture what CMake received.

    Returns ``(result, arguments_file, marker_file, repository, binaries)``.
    """
    repository = tmp_path / ("source with spaces" if spaced else "source")
    binaries = tmp_path / ("python with spaces" if spaced else "bin")
    installation = tmp_path / "triton"
    for path in (repository, binaries, installation):
        path.mkdir()
    script = repository / "build.sh"
    shutil.copyfile(BUILD, script)
    shutil.copyfile(BUILD.with_name("CMakeLists.txt"), repository / "CMakeLists.txt")
    script.chmod(script.stat().st_mode | 0o100)
    arguments = tmp_path / "cmake_args.json"
    marker = tmp_path / "cmake_called"
    real_python = shutil.which("python3")
    if real_python is None:
        pytest.skip("python3 must be on PATH to run the build.sh regressions")
    scripts = {
        "python3": "#!/bin/bash\necho " + abi + "\nexit " + str(python_exit) + "\n",
        "cmake": "#!" + real_python + "\nimport json,sys\nfrom pathlib import Path\n"
                 + "Path(" + repr(str(arguments)) + ").write_text(json.dumps(sys.argv[1:]))\n"
                 + "Path(" + repr(str(marker)) + ").write_text('called')\n",
        "make": "#!/bin/bash\nexit 0\n",
    }
    for name, text in scripts.items():
        path = binaries / name
        path.write_text(text)
        path.chmod(0o755)
    env = {"PATH": str(binaries) + os.pathsep + os.environ["PATH"],
           "TRITON_HOME_PATH": str(installation)}
    result = subprocess.run([str(script), *extra_args], cwd=repository,
                            env=env, capture_output=True, text=True, timeout=10)
    return result, arguments, marker, repository, binaries
