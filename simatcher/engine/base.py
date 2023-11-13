import json
import os
import copy
from typing import List, Dict, Text, Any, Optional, Union
from collections import defaultdict

from simatcher.common.stdlib import get_random_str, module_path_from_object
from simatcher.common.io import make_path_absolute, create_dir, write_to_file
from simatcher.nlp.base import Component, ComponentBuilder, validate_requirements
from simatcher.meta.model import Metadata
from simatcher.meta.message import Message
from simatcher.meta.training import TrainerModelConfig, TrainingData
from simatcher.exceptions import UnsupportedModelError, MissingArgumentError
from simatcher.version import __version__
from simatcher.log import logger
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
              output_properties: Dict = None,
              time=None,
              **kwargs) -> Message:
        """
        Parse the input text, classify it and return pipeline result.
        The pipeline result usually contains intent and entities.
        """

        message = Message(text, kwargs, output_properties=output_properties, time=time)
        for component in self.pipeline:
            component.process(message, **self.context)
        return message


class Trainer(object):
    """
    Trainer will load the data and train all components.
    Requires a pipeline specification(config.yml) and configuration(domain.yml)
    to use for the training.
    """

    SUPPORTED_LANGUAGES = ["zh", "en"]
    DEFAULT_PROJECT_NAME = 'default'

    def __init__(self,
                 cfg: Union[Dict, TrainerModelConfig],
                 component_builder: Optional[ComponentBuilder] = None,
                 skip_validation: bool = True):

        self.config = cfg
        self.skip_validation = skip_validation
        self.training_data = None
        self.component_names = [c.get("name") for c in cfg.get('pipeline')]

        if component_builder is None:
            _components_class = {c.name: c for c in COMPONENT_CLASSES}
            component_builder = ComponentBuilder(components_class=_components_class)

        if not self.skip_validation:
            validate_requirements(self.component_names)
        self.pipeline = self._build_pipeline(cfg, component_builder)

    @staticmethod
    def _build_pipeline(cfg: Union[Dict, TrainerModelConfig],
                        component_builder: ComponentBuilder) -> List:
        """Transform the passed names of the pipeline components into classes"""
        pipeline = [
            component_builder.create_component(component['name'], cfg, component)
            for component in cfg.get('pipeline')
        ]
        return pipeline

    def train(self, data: Union[Dict, TrainingData], **kwargs) -> Dict[Text, Any]:
        """
        1, Trains the pipeline using the provided training data.
        2, checking all the input parameter (empty pipeline, pre layer component)
        3, every component run training model
        """
        # domain.yml
        self.training_data = data
        context = kwargs

        for component in self.pipeline:
            updates = component.provide_context()
            updates and context.update(updates)

        working_data = copy.deepcopy(data)
        for i, component in enumerate(self.pipeline):
            logger.info(f"Starting to train component {component.name}")
            component.prepare_partial_processing(self.pipeline[:i], context)
            updates = component.train(working_data, self.config, **context)
            logger.info("Finished training component.")
            updates and context.update(updates)

        return context

    def persist(self,
                path: Text,
                persistor: Optional = None,
                project_name: Text = None,
                fixed_model_name: Text = None) -> Text:
        """
        Persist all components of the pipeline to the passed path.
        1, generate storage path
        2, create dir
        3, write metadata to the json file
        todo mv save flow to storage module
        """
        metadata = defaultdict()
        metadata["language"] = self.config["language"]

        model_name = fixed_model_name if fixed_model_name else f'model_{get_random_str()}'
        path = make_path_absolute(path)
        project_name = project_name or self.DEFAULT_PROJECT_NAME
        dir_name = os.path.join(path, project_name, model_name)
        create_dir(dir_name)

        if self.training_data:
            data_file = os.path.join(dir_name, "training_data.json")
            write_to_file(data_file, json.dumps(self.training_data, indent=2))

        metadata["pipeline"] = []
        for component in self.pipeline:
            update = component.persist(dir_name)
            component_meta = component.component_config
            update and component_meta.update(update)
            component_meta["class"] = module_path_from_object(component)
            metadata["pipeline"].append(component_meta)

        Metadata(metadata, dir_name).persist(dir_name)

        if persistor is not None:
            persistor.persist(dir_name, model_name, project_name)
        logger.info(f'Successfully saved model into {os.path.abspath(dir_name)}')
        return dir_name
