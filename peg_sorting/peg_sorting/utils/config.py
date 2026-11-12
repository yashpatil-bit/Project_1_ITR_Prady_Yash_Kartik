"""Loading / saving the YAML configuration files.

Config files live in ``peg_sorting/config``.  At run time they are looked up in
this order:

1. the directory given in the ``PEG_SORTING_CONFIG_DIR`` environment variable
2. the installed package share directory (``ros2 pkg prefix peg_sorting``)
3. the source tree (``<repo>/peg_sorting/config``) - useful without ROS

Build with ``colcon build --symlink-install`` so that the installed files are
symlinks to the source files: calibration tools then write straight back into
the repository and you can commit the calibrated values.
"""
import os
import copy
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE_CONFIG = os.path.normpath(os.path.join(_HERE, '..', '..', 'config'))


def config_dir():
    env = os.environ.get('PEG_SORTING_CONFIG_DIR')
    if env and os.path.isdir(env):
        return env
    try:
        from ament_index_python.packages import get_package_share_directory
        d = os.path.join(get_package_share_directory('peg_sorting'), 'config')
        if os.path.isdir(d):
            return d
    except Exception:  # noqa: BLE001 - ament not available / package not installed
        pass
    return _SOURCE_CONFIG


def config_path(name):
    if os.path.isabs(name):
        return name
    return os.path.join(config_dir(), name)


def load_yaml(name):
    path = config_path(name)
    with open(path, 'r') as f:
        data = yaml.safe_load(f) or {}
    return data


def _plain(v):
    """numpy scalars/arrays -> plain Python (yaml.safe_dump cannot write numpy types)."""
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if hasattr(v, 'tolist'):
        return v.tolist()
    return v


def save_yaml(name, data, header=None):
    """Write ``data`` to the config file, following symlinks so the source
    file is updated when the workspace was built with --symlink-install."""
    path = os.path.realpath(config_path(name))
    with open(path, 'w') as f:
        if header:
            for line in header.strip().splitlines():
                f.write('# ' + line + '\n')
        yaml.safe_dump(_plain(data), f, sort_keys=False, default_flow_style=None)
    return path


def update_yaml(name, updates, header=None):
    data = load_yaml(name)
    merged = deep_merge(data, updates)
    return save_yaml(name, merged, header=header)


def deep_merge(base, updates):
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out
