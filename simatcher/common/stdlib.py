import os
import string
import random
from typing import Dict, Generator, List, Callable, Text
from concurrent.futures import ThreadPoolExecutor, as_completed


def override_defaults(defaults: Dict, custom: Dict) -> Dict:
    cfg = defaults or {}
    if custom:
        cfg.update(custom)
    return cfg


def class_from_module_path(module_path):
    """Catch AttributeError and ImportError"""
    import importlib
    if "." in module_path:
        module_name, _, class_name = module_path.rpartition('.')
        m = importlib.import_module(module_name)
        return getattr(m, class_name)
    else:
        return globals()[module_path]


def run_in_thread_pool(func: Callable,
                       params: List[Dict] = [],
                       pool: ThreadPoolExecutor = None,
                       ) -> Generator:
    """
    Run the task in thread pool, use generator to return result
    Confirm thread security
    """
    pool = pool or ThreadPoolExecutor(os.cpu_count())
    tasks = [pool.submit(func, **kwargs) for kwargs in params]
    for obj in as_completed(tasks):
        yield obj.result()


def get_random_str(count: int = 8) -> Text:
    return ''.join(random.sample(string.ascii_letters + string.digits, count))


def module_path_from_object(o):
    """Returns the fully qualified class path of the instantiated object."""
    return o.__class__.__module__ + "." + o.__class__.__name__
