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

import json

import pytest

from build_stub import run_build


@pytest.mark.parametrize("spaced", [False, True])
def test_build_preserves_python_and_install_path_arguments(tmp_path, spaced):
    result, arguments, marker, repository, binaries = run_build(tmp_path, spaced=spaced)
    assert result.returncode == 0
    assert marker.exists()
    args = json.loads(arguments.read_text())
    assert "-DPython_EXECUTABLE=" + str(binaries / "python3") in args
    assert "-DCMAKE_INSTALL_PREFIX:PATH=" + str(repository / "build/install") in args
    assert "-DUSE_CXX11_ABI=ON" in args
    assert args[-1] == ".."
