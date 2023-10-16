from typing import Dict


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
