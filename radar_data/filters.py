from abc import ABC, abstractmethod
from typing import Sequence

from .types import Scene


class BaseFilter(ABC):
    def __init__(self):
        # TODO: decice how to configure filters
        pass

    @abstractmethod
    def apply(self, scene: Scene) -> Scene:
        pass


class Compose(BaseFilter):
    """
    Compose several fitlers

    Given (filter_1, filter_2, ..., fitler_n) applies them sequentially
    """
    def __init__(self, filters: Sequence[BaseFilter]):
        self.filters = filters

    def apply(self, scene: Scene) -> Scene:
        for filter_ in self.filters:
            scene = filter_.apply(scene)
        return scene


class IdentityFilter(BaseFilter):
    apply = lambda self, scene: scene
