from typing import (
    Text, List, Dict, Optional, Any, Tuple, Type, Hashable
)

from simatcher.log import logger
from simatcher.common.stdlib import override_defaults, class_from_module_path
from simatcher.meta.model import Metadata
from simatcher.meta.message import Message
from simatcher.exceptions import MissingArgumentError, InvalidRecipeException


class Component(object):
    name = ""
    provides = []
    requires = []
    defaults = {}

    language_list = None

    def __init__(self, component_config: Dict[Text, Any] = None):
        self.component_config = override_defaults(
                self.defaults, component_config)

        self.partial_processing_pipeline = None
        self.partial_processing_context = None

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __getstate__(self):
        d = self.__dict__.copy()
        if "partial_processing_context" in d:
            del d["partial_processing_context"]
        if "partial_processing_pipeline" in d:
            del d["partial_processing_pipeline"]
        return d

    @classmethod
    def load(cls,
             model_dir: Optional[Text] = None,
             model_metadata: Optional[Metadata] = None,
             cached_component: Optional['Component'] = None,
             **kwargs
             ) -> 'Component':
        if cached_component:
            return cached_component
        else:
            component_config = model_metadata.for_component(cls.name)
            return cls(component_config)

    @classmethod
    def required_packages(cls) -> List[Text]:
        return []

    def provide_context(self) -> Optional[Dict[Text, Any]]:
        pass

    def prepare_partial_processing(self, pipeline: List[object], context: Dict):
        self.partial_processing_pipeline = pipeline
        self.partial_processing_context = context

    @classmethod
    def create(cls, cfg: Dict, *args, **kwargs):
        return cls(cfg)

    def train(self, *args, **kwargs):
        pass

    def process(self, message: Message, **kwargs):
        pass

    def persist(self, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this component to disk for future loading."""
        pass

    @classmethod
    def cache_key(cls, model_metadata: Metadata) -> Optional[Text]:
        return None

    @classmethod
    def can_handle_language(cls, language: Hashable) -> bool:
        """if language_list is set to `None` it means: support all languages"""
        if language is None or cls.language_list is None:
            return True
        return language in cls.language_list


class ComponentBuilder(object):
    """
    Creates trainers based on configurations.
    1, Try to load from cache
    2, If not cache, register it
    3, Redis will be first consider
    """

    def __init__(self, use_cache=True, components_class: Dict[Text, Optional[Type[Component]]] = {}):
        self.use_cache = use_cache
        self.component_cache = {}
        self._registered_components = components_class

    def _add_to_cache(self, component: Component, cache_key: Text):
        if cache_key is not None and self.use_cache:
            self.component_cache[cache_key] = component
            logger.info(f'Added "{component.name}" to component cache. Key "{cache_key}".')

    def _get_from_cache(self,
                        component_name: Text,
                        model_metadata: Metadata) -> Tuple[Optional[Component], Optional[Text]]:

        component_class = self.get_component_class(component_name)
        cache_key = component_class.cache_key(model_metadata)
        if cache_key is not None and self.use_cache and cache_key in self.component_cache:
            return self.component_cache[cache_key], cache_key
        return None, cache_key

    def load_component(self,
                       component_name: Text,
                       model_dir: Text,
                       model_metadata: Metadata,
                       **context) -> Component:
        """Tries to retrieve a component from the cache, calls
               `load` to create a new component."""

        try:
            cached_component, cache_key = self._get_from_cache(
                component_name, model_metadata)
            component = self.load_component_by_name(
                component_name, model_dir, model_metadata,
                cached_component, **context)
            if not cached_component:
                self._add_to_cache(component, cache_key)
            return component
        except MissingArgumentError as e:
            logger.error(f'Failed to load component "{component_name}". {e}')
            raise

    def create_component_by_name(self,
                                 component_name: Text,
                                 config: Dict) -> Optional[Component]:
        component_clz = self.get_component_class(component_name)
        return component_clz.create(config)

    def load_component_by_name(self,
                               component_name: Text,
                               model_dir: Text,
                               metadata: Metadata,
                               cached_component: Optional[Component],
                               **kwargs) -> Optional[Component]:
        component_clz = self.get_component_class(component_name)
        return component_clz.load(model_dir, metadata, cached_component, **kwargs)

    def get_component_class(self, component_name: Text) -> Optional[Type[Component]]:
        if component_name not in self._registered_components:
            try:
                return class_from_module_path(component_name)
            except InvalidRecipeException:
                raise InvalidRecipeException(
                    f"Failed to find component class for '{component_name}'. Unknown ")
        return self._registered_components[component_name]

    def create_component(self,
                         component_name: Text,
                         cfg: Dict,
                         menu: Optional) -> Component:
        try:
            component, cache_key = self._get_from_cache(
                component_name, Metadata(cfg, None), menu)
            if component is None:
                component = menu.create_component_by_name(component_name, cfg)
                self._add_to_cache(component, cache_key)
            return component
        except MissingArgumentError as e:  # pragma: no cover
            logger.error(f'Failed to load component "{component_name}". {e}')
            raise


def validate_requirements(component_names: List[Text]):
    """Ensures that all required python packages are installed to
        instantiate and used the passed components."""
    pass
