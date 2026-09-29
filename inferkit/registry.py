"""模型注册表：管理多个模型与版本，默认版本切换原子可见。

版本切换的在途安全由执行器保证：批次形成时会固定该批次使用的模型对象，
注册表随后如何切换都不会影响已经形成的批次。
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple

from .errors import UnknownModelError
from .model import LocalModel


class _ModelEntry:
    __slots__ = ("versions", "default_version")

    def __init__(self) -> None:
        self.versions: Dict[str, LocalModel] = {}
        self.default_version: Optional[str] = None


class ModelRegistry:
    def __init__(self) -> None:
        self._entries: Dict[str, _ModelEntry] = {}
        self._lock = threading.RLock()

    def register(self, model: LocalModel, make_default: bool = True) -> None:
        """注册一个模型实例；首个注册版本自动成为默认版本。"""
        with self._lock:
            entry = self._entries.setdefault(model.name, _ModelEntry())
            entry.versions[model.version] = model
            if entry.default_version is None or make_default:
                entry.default_version = model.version

    def resolve(self, name: str, version: Optional[str] = None) -> LocalModel:
        """解析模型实例；``version=None`` 时返回当前默认版本。"""
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                raise UnknownModelError(f"模型 {name!r} 未注册")
            target_version = version or entry.default_version
            model = entry.versions.get(target_version)
            if model is None:
                raise UnknownModelError(
                    f"模型 {name!r} 不存在版本 {target_version!r}，"
                    f"可用版本: {sorted(entry.versions)}"
                )
            return model

    def swap_default(self, name: str, model: LocalModel) -> Tuple[Optional[LocalModel], str]:
        """原子地把 ``model`` 注册为新版本并设为默认。

        返回 ``(旧默认模型, 旧默认版本)``。整个替换在同一把锁内完成，
        因此并发解析调用看到的要么全是旧版本、要么全是新版本。
        """
        if model.name != name:
            raise ValueError(
                f"模型名不一致: 注册名 {name!r} vs 实例名 {model.name!r}"
            )
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                raise UnknownModelError(f"模型 {name!r} 未注册，无法重载")
            old_model = entry.versions.get(entry.default_version or "")
            old_version = entry.default_version or ""
            entry.versions[model.version] = model
            entry.default_version = model.version
            return old_model, old_version

    def current_version(self, name: str) -> str:
        with self._lock:
            entry = self._entries.get(name)
            if entry is None or entry.default_version is None:
                raise UnknownModelError(f"模型 {name!r} 未注册")
            return entry.default_version

    def versions(self, name: str) -> List[str]:
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                raise UnknownModelError(f"模型 {name!r} 未注册")
            return sorted(entry.versions)

    def model_names(self) -> List[str]:
        with self._lock:
            return sorted(self._entries)

    def shutdown(self) -> None:
        """关闭所有已注册模型实例。"""
        with self._lock:
            for entry in self._entries.values():
                for model in entry.versions.values():
                    model.close()
