from typing import List, Dict, Text, Any, Optional, Union

from simatcher.nlp.base import Component, ComponentBuilder, validate_requirements
from simatcher.meta.model import Metadata
from simatcher.meta.message import Message
from simatcher.exceptions import UnsupportedModelError, MissingArgumentError
from simatcher.version import __version__
from .default_components import COMPONENT_CLASSES


class Runner(object):
    """
    parse -> load -> create -> nlp_class
    """
    def __init__(self,
                 pipeline: List[Component],
                 context: Dict[Text, Any],
                 model_metadata: Optional[Metadata] = None):
        self.pipeline = pipeline
        self.context = context if context is not None else {}
        self.model_metadata = model_metadata

    @staticmethod
    def ensure_model_compatibility(metadata: Metadata):
        from packaging import version

        model_version = metadata.get("version", "0.0.0")
        if version.parse(model_version) != version.parse(__version__):
            raise UnsupportedModelError(f'Only support version {model_version}')

    @staticmethod
    def create(model_metadata: Metadata,
               component_builder: Optional[ComponentBuilder] = None,
               skip_validation: bool = False):
        """
        A Factory based on metadata
        Read the class path and Init object
        Insert to the sorted pipeline
        """
        if component_builder is None:
            _components_class = {c.name: c for c in COMPONENT_CLASSES}
            component_builder = ComponentBuilder(components_class=_components_class)

        if not skip_validation:
            validate_requirements(model_metadata.component_classes)

        pipeline, context = [], {}
        for component_name in model_metadata.component_classes:
            component = component_builder.load_component(
                component_name, model_metadata.model_dir,
                model_metadata, **context)
            try:
                updates = component.provide_context()
                if updates:
                    context.update(updates)
                pipeline.append(component)
            except MissingArgumentError as e:
                raise Exception("Failed to initialize component '{}'. "
                                "{}".format(component.name, e))

        return Runner(pipeline, context, model_metadata)

    @staticmethod
    def load(model_source: Union[Text, Dict],
             component_builder: Optional[ComponentBuilder] = None,
             skip_validation: bool = False):
        """Creates an interpreter based on a persisted model."""

        if isinstance(model_source, Text):
            model_metadata = Metadata.load(model_source)
        else:
            model_metadata = Metadata(model_source)

        Runner.ensure_model_compatibility(model_metadata)
        return Runner.create(model_metadata,
                             component_builder,
                             skip_validation)

    def parse(self,
              text: Text,
              time=None,
              output_properties: Dict = None,
              only_output_properties=True) -> Dict[Text, Any]:
        """
        Parse the input text, classify it and return pipeline result.
        The pipeline result usually contains intent and entities.
        """

        message = Message(text, output_properties, time=time)
        for component in self.pipeline:
            component.process(message, **self.context)
        return message.as_dict(only_output_properties=only_output_properties)
