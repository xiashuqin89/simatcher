import io
import os
import json
import errno
from functools import wraps
from typing import Text, List, Any

import six
import yaml
import simplejson


def fix_yaml_loader(func):
    @wraps(func)
    def _wrapper(*args, **kwargs):
        from yaml import Loader, SafeLoader

        def construct_yaml_str(self, node):
            # Override the default string handling function
            # to always return unicode objects
            return self.construct_scalar(node)

        Loader.add_constructor(u'tag:yaml.org,2002:str', construct_yaml_str)
        SafeLoader.add_constructor(u'tag:yaml.org,2002:str', construct_yaml_str)
        return func(*args, **kwargs)
    return _wrapper


@fix_yaml_loader
def read_yaml_file(filename):
    return yaml.load(read_file(filename, "utf-8"))


def read_file(filename, encoding="utf-8"):
    """Read text from a file."""
    with io.open(filename, encoding=encoding) as f:
        return f.read()


def list_directory(path: Text) -> List[Text]:
    """
    Returns all files and folders excluding hidden files.
    If the path points to a file, returns the file. This is a recursive
    implementation returning files in any depth of the path.
    """

    if not isinstance(path, six.string_types):
        raise ValueError("Resource name must be a string type")

    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        results = []
        for base, _, files in os.walk(path):
            results.extend([os.path.join(base, f) for f in files if not f.startswith('.')])
        return results
    else:
        raise ValueError(f"Could not locate the resource '{os.path.abspath(path)}'.")


def list_files(path: Text) -> List[Text]:
    return [fn for fn in list_directory(path) if os.path.isfile(fn)]


def json_to_string(obj: Any, **kwargs):
    indent = kwargs.pop("indent", 2)
    ensure_ascii = kwargs.pop("ensure_ascii", False)
    return json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii, **kwargs)


def read_json_file(filename):
    """Read json from a file."""
    content = read_file(filename)
    try:
        return simplejson.loads(content)
    except ValueError as e:
        raise ValueError(f'Failed to read json from "{os.path.abspath(filename)}". Error: {e}')


def write_json_to_file(filename: Text, obj: Any, **kwargs):
    write_to_file(filename, json_to_string(obj, **kwargs))


def write_to_file(filename: Text, text: Text):
    with io.open(filename, 'w', encoding="utf-8") as f:
        f.write(str(text))


def create_dir(dir_path: Text):
    """Creates a directory and its super paths.
    Succeeds even if the path already exists."""
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def make_path_absolute(path: Text) -> Text:
    if path and not os.path.isabs(path):
        return os.path.join(os.getcwd(), path)
    else:
        return path


def py_cloud_unpickle(file_name: Text) -> Any:
    from future.utils import PY2
    import cloudpickle

    with io.open(file_name, 'rb') as f:
        if PY2:
            return cloudpickle.load(f)
        else:
            return cloudpickle.load(f, encoding="latin-1")


def py_cloud_pickle(file_name: Text, obj: Any):
    import cloudpickle
    with io.open(file_name, 'wb') as f:
        cloudpickle.dump(obj, f)
