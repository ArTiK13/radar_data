from abc import ABC, abstractmethod

from .types import Scene


class BaseFilter(ABC):
    def __init__(self):
        # TODO: decice how to configure filters
        pass

    @abstractmethod
    def apply(self, scene: Scene) -> Scene:
        pass


class IdentityFilter(BaseFilter):
    apply = lambda self, scene: scene
