"""utils/system_info.py 的单元测试。"""

import json
import subprocess
from unittest.mock import patch, MagicMock

from utils.system_info import (
    get_cpu_count,
    get_memory_info,
    _is_macos,
    _get_apple_silicon_info,
    get_gpu_info,
    get_hardware_description,
    get_conda_python_path,
    get_conda_packages,
)


class TestGetCpuCount:
    """测试 get_cpu_count 函数。"""

    def test_returns_cpu_count(self):
        """测试正常返回 CPU 核心数。"""
        count = get_cpu_count()
        assert isinstance(count, int)
        assert count >= 1

    @patch("os.cpu_count", return_value=None)
    def test_returns_1_when_none(self, mock_cpu):
        """测试 os.cpu_count() 返回 None 时默认 1。"""
        assert get_cpu_count() == 1

    @patch("os.cpu_count", side_effect=Exception("fail"))
    def test_returns_1_on_exception(self, mock_cpu):
        """测试异常时默认 1。"""
        assert get_cpu_count() == 1


class TestGetMemoryInfo:
    """测试 get_memory_info 函数。"""

    def test_returns_dict_with_keys(self):
        """测试返回包含 total 和 available 的字典。"""
        info = get_memory_info()
        assert "total" in info
        assert "available" in info
        assert info["total"] > 0

    @patch("platform.system", return_value="Linux")
    def test_linux_reads_proc_meminfo(self, mock_sys, tmp_path):
        """测试 Linux 平台从 /proc/meminfo 读取。"""
        info = get_memory_info()
        assert info["total"] > 0


class TestIsMacos:
    """测试 _is_macos 函数。"""

    @patch("platform.system", return_value="Darwin")
    def test_returns_true_on_darwin(self, mock_sys):
        """macOS 返回 True。"""
        assert _is_macos() is True

    @patch("platform.system", return_value="Linux")
    def test_returns_false_on_linux(self, mock_sys):
        """Linux 返回 False。"""
        assert _is_macos() is False


class TestGetAppleSiliconInfo:
    """测试 _get_apple_silicon_info 函数。"""

    @patch("platform.system", return_value="Linux")
    def test_returns_none_on_non_mac(self, mock_sys):
        """非 macOS 返回 None。"""
        assert _get_apple_silicon_info() is None


class TestGetGpuInfo:
    """测试 get_gpu_info 函数。"""

    @patch("subprocess.run")
    @patch("utils.system_info._is_macos", return_value=False)
    def test_nvidia_single_gpu(self, mock_mac, mock_run):
        """检测到单个 NVIDIA GPU。"""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NVIDIA GeForce RTX 3090, 24576 MiB\n",
        )

        result = get_gpu_info()
        assert "RTX 3090" in result
        assert "24576" in result

    @patch("subprocess.run")
    @patch("utils.system_info._is_macos", return_value=False)
    def test_nvidia_multi_gpu(self, mock_mac, mock_run):
        """检测到多个 NVIDIA GPU。"""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NVIDIA A100, 40960 MiB\nNVIDIA A100, 40960 MiB\n",
        )

        result = get_gpu_info()
        assert "2x" in result

    @patch("subprocess.run", side_effect=FileNotFoundError)
    @patch("utils.system_info._is_macos", return_value=False)
    def test_no_nvidia_no_mac(self, mock_mac, mock_run):
        """无 NVIDIA 且非 macOS 返回 None。"""
        assert get_gpu_info() is None

    @patch("subprocess.run", side_effect=FileNotFoundError)
    @patch("utils.system_info._is_macos", return_value=True)
    @patch("utils.system_info._get_apple_silicon_info", return_value="Apple M1")
    def test_apple_silicon_gpu(self, mock_apple, mock_mac, mock_run):
        """macOS Apple Silicon 返回集成 GPU 信息。"""
        result = get_gpu_info()
        assert "Apple M1" in result
        assert "integrated" in result


class TestGetHardwareDescription:
    """测试 get_hardware_description 函数。"""

    @patch("utils.system_info.get_gpu_info", return_value="NVIDIA RTX 3090 24GB")
    @patch("utils.system_info.get_memory_info", return_value={"total": 32.0, "available": 16.0})
    @patch("utils.system_info.get_cpu_count", return_value=8)
    def test_with_gpu(self, mock_cpu, mock_mem, mock_gpu):
        """测试有 GPU 的描述。"""
        desc = get_hardware_description()
        assert "CPU: 8 cores" in desc
        assert "RAM: 32GB" in desc
        assert "GPU: NVIDIA RTX 3090" in desc

    @patch("utils.system_info.get_gpu_info", return_value=None)
    @patch("utils.system_info.get_memory_info", return_value={"total": 8.0, "available": 4.0})
    @patch("utils.system_info.get_cpu_count", return_value=4)
    def test_without_gpu(self, mock_cpu, mock_mem, mock_gpu):
        """测试无 GPU 的描述。"""
        desc = get_hardware_description()
        assert "CPU: 4 cores" in desc
        assert "RAM: 8GB" in desc
        assert "GPU" not in desc


class TestGetCondaPythonPath:
    """测试 get_conda_python_path 函数。"""

    @patch("subprocess.run")
    def test_returns_path_from_which(self, mock_run, tmp_path):
        """测试从 conda run which python 获取路径。"""
        python_path = str(tmp_path / "bin" / "python")
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "python").write_text("#!/usr/bin/env python3")

        mock_run.return_value = MagicMock(
            returncode=0, stdout=python_path + "\n"
        )

        result = get_conda_python_path("test_env")
        assert result == python_path

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_returns_none_when_conda_missing(self, mock_run):
        """conda 不可用时返回 None。"""
        assert get_conda_python_path("test_env") is None

    @patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="conda", timeout=15))
    def test_returns_none_on_timeout(self, mock_run):
        """超时时返回 None。"""
        assert get_conda_python_path("test_env") is None


class TestGetCondaPackages:
    """测试 get_conda_packages 函数。"""

    @patch("subprocess.run")
    def test_returns_description(self, mock_run):
        """测试正常返回包描述。"""
        packages = [
            {"name": "python", "version": "3.10.19", "channel": "defaults"},
            {"name": "pandas", "version": "2.0.3", "channel": "defaults"},
            {"name": "numpy", "version": "1.24.3", "channel": "defaults"},
            {"name": "scikit-learn", "version": "1.3.0", "channel": "defaults"},
            {"name": "torch", "version": "2.0.1", "channel": "pytorch"},
        ]
        mock_run.return_value = MagicMock(
            returncode=0, stdout=json.dumps(packages)
        )

        result = get_conda_packages("Swarm-Evo")

        assert "Swarm-Evo" in result
        assert "python 3.10.19" in result
        assert "pandas" in result
        assert "PyTorch available" in result

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_returns_default_on_missing_conda(self, mock_run):
        """conda 不可用时返回默认描述。"""
        result = get_conda_packages("test_env")
        assert "unavailable" in result

    @patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="conda", timeout=15))
    def test_returns_default_on_timeout(self, mock_run):
        """超时时返回默认描述。"""
        result = get_conda_packages("test_env")
        assert "unavailable" in result

    @patch("subprocess.run")
    def test_not_installed_packages_warning(self, mock_run):
        """测试未安装包的警告。"""
        packages = [
            {"name": "python", "version": "3.10.0", "channel": "defaults"},
            {"name": "pandas", "version": "2.0.0", "channel": "defaults"},
        ]
        mock_run.return_value = MagicMock(
            returncode=0, stdout=json.dumps(packages)
        )

        result = get_conda_packages("test_env")
        assert "NOT INSTALLED" in result

    @patch("subprocess.run")
    def test_invalid_json_output(self, mock_run):
        """测试无效 JSON 输出。"""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="not valid json"
        )

        result = get_conda_packages("test_env")
        assert "unavailable" in result

    @patch("subprocess.run")
    def test_returncode_nonzero(self, mock_run):
        """conda list 返回非零退出码。"""
        mock_run.return_value = MagicMock(
            returncode=1, stderr="EnvironmentNotFound"
        )

        result = get_conda_packages("bad_env")
        assert "unavailable" in result

    @patch("subprocess.run")
    def test_non_list_json_output(self, mock_run):
        """JSON 输出不是列表。"""
        mock_run.return_value = MagicMock(
            returncode=0, stdout='{"error": "unexpected"}'
        )

        result = get_conda_packages("test_env")
        assert "unavailable" in result

    @patch("subprocess.run")
    def test_packages_with_torch_and_timm(self, mock_run):
        """测试 PyTorch + timm 生态描述。"""
        packages = [
            {"name": "python", "version": "3.10.0", "channel": "defaults"},
            {"name": "pandas", "version": "2.0.0", "channel": "defaults"},
            {"name": "torch", "version": "2.0.1", "channel": "pytorch"},
            {"name": "timm", "version": "0.9.0", "channel": "conda-forge"},
            {"name": "torchvision", "version": "0.15.0", "channel": "pytorch"},
        ]
        mock_run.return_value = MagicMock(
            returncode=0, stdout=json.dumps(packages)
        )

        result = get_conda_packages("test_env")
        assert "PyTorch available" in result
        assert "timm" in result
        assert "Image classification" in result

    @patch("subprocess.run")
    def test_packages_with_tensorflow_no_torch(self, mock_run):
        """测试仅 TensorFlow（无 PyTorch）描述。"""
        packages = [
            {"name": "python", "version": "3.10.0", "channel": "defaults"},
            {"name": "pandas", "version": "2.0.0", "channel": "defaults"},
            {"name": "tensorflow", "version": "2.13.0", "channel": "defaults"},
        ]
        mock_run.return_value = MagicMock(
            returncode=0, stdout=json.dumps(packages)
        )

        result = get_conda_packages("test_env")
        assert "TensorFlow" in result
        assert "no PyTorch detected" in result

    @patch("subprocess.run")
    def test_packages_with_transformers(self, mock_run):
        """测试含 transformers 包的 NLP 推荐。"""
        packages = [
            {"name": "python", "version": "3.10.0", "channel": "defaults"},
            {"name": "transformers", "version": "4.30.0", "channel": "defaults"},
            {"name": "torch", "version": "2.0.1", "channel": "pytorch"},
        ]
        mock_run.return_value = MagicMock(
            returncode=0, stdout=json.dumps(packages)
        )

        result = get_conda_packages("test_env")
        assert "Text/NLP" in result
        assert "transformers" in result

    @patch("subprocess.run")
    def test_packages_with_boosting(self, mock_run):
        """测试含 Boosting 包的表格推荐。"""
        packages = [
            {"name": "python", "version": "3.10.0", "channel": "defaults"},
            {"name": "xgboost", "version": "1.7.0", "channel": "defaults"},
            {"name": "lightgbm", "version": "3.3.0", "channel": "defaults"},
        ]
        mock_run.return_value = MagicMock(
            returncode=0, stdout=json.dumps(packages)
        )

        result = get_conda_packages("test_env")
        assert "Tabular" in result
        assert "gradient boosting" in result

    @patch("subprocess.run", side_effect=Exception("Unexpected error"))
    def test_generic_exception(self, mock_run):
        """通用异常返回默认描述。"""
        result = get_conda_packages("test_env")
        assert "unavailable" in result


class TestGetGpuInfoExtended:
    """get_gpu_info 额外路径。"""

    @patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5))
    @patch("utils.system_info._is_macos", return_value=False)
    def test_nvidia_timeout(self, mock_mac, mock_run):
        """nvidia-smi 超时返回 None。"""
        assert get_gpu_info() is None

    @patch("subprocess.run", side_effect=FileNotFoundError)
    @patch("utils.system_info._is_macos", return_value=True)
    @patch("utils.system_info._get_apple_silicon_info", return_value=None)
    def test_macos_no_apple_silicon(self, mock_apple, mock_mac, mock_run):
        """macOS 非 Apple Silicon 返回 None。"""
        result = get_gpu_info()
        assert result is None


class TestGetCondaPythonPathExtended:
    """get_conda_python_path 额外路径。"""

    @patch("subprocess.run")
    def test_fallback_to_conda_info(self, mock_run, tmp_path):
        """which python 失败后 fallback 到 conda info --envs。"""
        python_path = str(tmp_path / "envs" / "test_env" / "bin" / "python")
        (tmp_path / "envs" / "test_env" / "bin").mkdir(parents=True)
        (tmp_path / "envs" / "test_env" / "bin" / "python").write_text("#!/usr/bin/env python3")

        def run_side_effect(cmd, **kwargs):
            if "which" in cmd:
                return MagicMock(returncode=1, stdout="")
            elif "--envs" in cmd:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({"envs": [str(tmp_path / "envs" / "test_env")]})
                )
            return MagicMock(returncode=1, stdout="")

        mock_run.side_effect = run_side_effect

        result = get_conda_python_path("test_env")
        assert result == python_path

    @patch("subprocess.run", side_effect=Exception("generic error"))
    def test_generic_exception(self, mock_run):
        """通用异常返回 None。"""
        assert get_conda_python_path("test_env") is None


class TestGetMemoryInfoExtended:
    """get_memory_info 额外未覆盖路径测试。"""

    @patch("platform.system", return_value="Linux")
    @patch("builtins.open", side_effect=Exception("read fail"))
    def test_linux_proc_meminfo_exception_falls_through_to_psutil(self, mock_open, mock_sys):
        """Linux 下读取 /proc/meminfo 抛出异常时，回退到 psutil 继续获取内存信息。"""
        # psutil 在当前环境可用，异常后仍能通过 psutil 返回正确结果
        info = get_memory_info()
        assert "total" in info
        assert info["total"] > 0

    @patch("platform.system", return_value="Windows")
    def test_non_linux_skips_proc_meminfo_uses_psutil(self, mock_sys):
        """非 Linux 平台跳过 /proc/meminfo，直接使用 psutil。"""
        info = get_memory_info()
        assert "total" in info
        assert "available" in info
        assert info["total"] > 0

    @patch("platform.system", return_value="Windows")
    def test_psutil_import_error_returns_default(self, mock_sys):
        """psutil 不可用（ImportError）时返回默认内存值 {"total": 8.0, "available": 8.0}。"""
        import sys

        # 用 None 替换 psutil 模块，模拟未安装
        with patch.dict(sys.modules, {"psutil": None}):
            info = get_memory_info()
            assert info["total"] == 8.0
            assert info["available"] == 8.0

    @patch("platform.system", return_value="Windows")
    def test_psutil_exception_returns_default(self, mock_sys):
        """psutil 已安装但调用 virtual_memory() 抛出异常时返回默认值。"""
        import psutil

        with patch.object(psutil, "virtual_memory", side_effect=Exception("psutil fail")):
            info = get_memory_info()
            assert info["total"] == 8.0
            assert info["available"] == 8.0


class TestGetAppleSiliconInfoExtended:
    """_get_apple_silicon_info 额外未覆盖路径测试。"""

    @patch("platform.system", return_value="Darwin")
    @patch("subprocess.run")
    def test_apple_silicon_detected_from_sysctl(self, mock_run, mock_sys):
        """macOS sysctl 成功返回含 'Apple' 品牌名时，应返回该字符串。"""
        mock_run.return_value = MagicMock(returncode=0, stdout="Apple M1 Pro\n")
        result = _get_apple_silicon_info()
        assert result == "Apple M1 Pro"

    @patch("platform.system", return_value="Darwin")
    @patch("subprocess.run")
    @patch("platform.machine", return_value="x86_64")
    def test_intel_mac_sysctl_non_apple_brand_returns_none(self, mock_machine, mock_run, mock_sys):
        """macOS sysctl 成功但品牌名不含 'Apple'（Intel Mac），且架构为 x86_64，返回 None。"""
        mock_run.return_value = MagicMock(returncode=0, stdout="Intel(R) Core(TM) i9-9900K\n")
        result = _get_apple_silicon_info()
        assert result is None

    @patch("platform.system", return_value="Darwin")
    @patch("subprocess.run", side_effect=Exception("sysctl command not found"))
    @patch("platform.machine", return_value="arm64")
    def test_sysctl_exception_fallback_arm64(self, mock_machine, mock_run, mock_sys):
        """sysctl 抛出异常后，通过 platform.machine() == 'arm64' 识别 Apple Silicon。"""
        result = _get_apple_silicon_info()
        assert result == "Apple Silicon (arm64)"

    @patch("platform.system", return_value="Darwin")
    @patch("subprocess.run", side_effect=Exception("sysctl fail"))
    @patch("platform.machine", return_value="x86_64")
    def test_sysctl_exception_non_arm64_returns_none(self, mock_machine, mock_run, mock_sys):
        """sysctl 异常且架构非 arm64 时，最终返回 None。"""
        result = _get_apple_silicon_info()
        assert result is None

    @patch("platform.system", return_value="Darwin")
    @patch("subprocess.run")
    @patch("platform.machine", return_value="x86_64")
    def test_sysctl_nonzero_returncode_falls_through(self, mock_machine, mock_run, mock_sys):
        """sysctl 返回非零退出码时，跳过品牌检测，x86_64 架构最终返回 None。"""
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        result = _get_apple_silicon_info()
        assert result is None

    @patch("platform.system", return_value="Darwin")
    @patch("subprocess.run")
    @patch("platform.machine", return_value="arm64")
    def test_sysctl_nonzero_returncode_fallback_arm64(self, mock_machine, mock_run, mock_sys):
        """sysctl 返回非零退出码，但架构为 arm64 时，回退识别返回 'Apple Silicon (arm64)'。"""
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        result = _get_apple_silicon_info()
        assert result == "Apple Silicon (arm64)"


class TestGetGpuInfoGenericException:
    """get_gpu_info 通用异常路径（lines 172-173）测试。"""

    @patch("subprocess.run", side_effect=RuntimeError("unexpected nvidia error"))
    @patch("utils.system_info._is_macos", return_value=False)
    def test_generic_nvidia_exception_returns_none(self, mock_mac, mock_run):
        """nvidia-smi 抛出非 FileNotFoundError、非 TimeoutExpired 的通用异常时，返回 None。"""
        assert get_gpu_info() is None

    @patch("subprocess.run", side_effect=OSError("device busy"))
    @patch("utils.system_info._is_macos", return_value=False)
    def test_oserror_nvidia_exception_returns_none(self, mock_mac, mock_run):
        """nvidia-smi 抛出 OSError 时，返回 None（走通用 except 分支）。"""
        assert get_gpu_info() is None


class TestGetCondaPackagesExtended2:
    """get_conda_packages 额外未覆盖路径测试（line 287 和 363-364）。"""

    @patch("subprocess.run")
    def test_no_env_name_omits_name_flag(self, mock_run):
        """env_name=None 时构建的命令不应包含 '--name' 参数。"""
        packages = [
            {"name": "python", "version": "3.10.0", "channel": "defaults"},
            {"name": "pandas", "version": "2.0.0", "channel": "defaults"},
        ]
        mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(packages))

        result = get_conda_packages(None)

        # 验证命令中没有 --name 标志
        call_args = mock_run.call_args[0][0]
        assert "--name" not in call_args
        # 结果应该包含 'current' 作为环境名
        assert "current" in result
        assert "pandas" in result

    @patch("subprocess.run")
    def test_no_env_name_default_call(self, mock_run):
        """env_name=None 时使用默认参数调用，命令为 ['conda', 'list', '--json']。"""
        mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps([]))

        get_conda_packages(None)

        call_args = mock_run.call_args[0][0]
        assert call_args == ["conda", "list", "--json"]

    @patch("subprocess.run")
    def test_priority_key_unknown_package_sorted_last(self, mock_run):
        """不在优先级列表中的包排序时触发 ValueError，作为 fallback 排在已知包后面。"""
        packages = [
            {"name": "python", "version": "3.10.0", "channel": "defaults"},
            {"name": "pandas", "version": "2.0.0", "channel": "defaults"},
            {"name": "numpy", "version": "1.24.0", "channel": "defaults"},
            # obscure-pkg 不在 core_packages_priority，触发 ValueError → 返回 len(priority)
            {"name": "obscure-pkg", "version": "1.0", "channel": "defaults"},
            {"name": "xgboost", "version": "1.7.0", "channel": "defaults"},
        ]
        mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(packages))

        result = get_conda_packages("test_env")

        # 正常包应出现在描述中
        assert "pandas" in result
        assert "xgboost" in result
        # 未知包 obscure-pkg 不在 core_packages 集合内，不会出现在高亮列表
        # 但整体结果不应因 ValueError 崩溃，应正常返回描述
        assert "test_env" in result

    @patch("subprocess.run")
    def test_priority_sort_stable_with_mixed_packages(self, mock_run):
        """混合已知/未知包排序时，已知包按优先级排列，整体不崩溃。"""
        packages = [
            {"name": "python", "version": "3.10.0", "channel": "defaults"},
            # 以下均在 core_packages_priority 中，触发正常 index 查找
            {"name": "torch", "version": "2.0.1", "channel": "pytorch"},
            {"name": "pandas", "version": "2.0.0", "channel": "defaults"},
            {"name": "numpy", "version": "1.24.0", "channel": "defaults"},
        ]
        mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(packages))

        result = get_conda_packages("test_env")

        assert "pandas" in result
        assert "numpy" in result
        assert "torch" in result
