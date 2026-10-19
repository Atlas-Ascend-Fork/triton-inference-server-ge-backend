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

import os
import subprocess
from pathlib import Path

import pytest

BUILD = Path(__file__).resolve().parents[2] / "build.sh"


@pytest.mark.parametrize("initial", [None, "/custom/triton"])
def test_default_path_is_inherited_by_child_process(initial):
    source = BUILD.read_text()
    start = source.index('if [ -z "$TRITON_HOME_PATH" ]; then')
    end = source.index('echo "Triton install path:', start)
    default_block = source[start:end]
    env = {"PATH": os.environ["PATH"]}
    if initial is not None:
        env["TRITON_HOME_PATH"] = initial
    result = subprocess.run(["bash", "-c", default_block + '\nprintenv TRITON_HOME_PATH'],
                            env=env, capture_output=True, text=True, timeout=5)
    assert result.returncode == 0
    assert result.stdout.splitlines()[-1] == (initial or "/opt/tritonserver")
