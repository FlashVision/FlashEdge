"""Registry pattern for pluggable components.

Usage:
    from flashedge.registry import MODELS, EXPORTERS, RUNTIMES, OPTIMIZERS

    @EXPORTERS.register("MyExporter")
    class MyExporter:
        ...

    exporter = EXPORTERS.build("MyExporter", **kwargs)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


class Registry:
    """A registry that maps names to classes/functions for dynamic instantiation."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: Dict[str, Callable] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def registry(self) -> Dict[str, Callable]:
        return self._registry

    def register(self, name: Optional[str] = None) -> Callable:
        """Register a class or function under a given name.

        Can be used as a decorator:
            @REGISTRY.register("MyClass")
            class MyClass:
                ...
        """

        def decorator(cls_or_fn: Callable) -> Callable:
            key = name or cls_or_fn.__name__
            if key in self._registry:
                raise KeyError(f"'{key}' is already registered in {self._name}")
            self._registry[key] = cls_or_fn
            return cls_or_fn

        return decorator

    def build(self, name: str, **kwargs: Any) -> Any:
        """Instantiate a registered component by name."""
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(f"'{name}' not found in {self._name} registry. Available: [{available}]")
        return self._registry[name](**kwargs)

    def get(self, name: str) -> Optional[Callable]:
        """Get a registered component without instantiation."""
        return self._registry.get(name)

    def list(self) -> list:
        """List all registered component names."""
        return sorted(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        return f"Registry(name={self._name}, items={len(self._registry)})"


MODELS = Registry("models")
EXPORTERS = Registry("exporters")
RUNTIMES = Registry("runtimes")
OPTIMIZERS = Registry("optimizers")

__all__ = [
    "Registry",
    "MODELS", "EXPORTERS", "RUNTIMES", "OPTIMIZERS",
]
