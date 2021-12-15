from pathlib import Path
import numpy as np
from dm_control.suite import common
from dm_control.utils import io as resources
import xmltodict


_SUITE_DIR = Path("/opt/conda/envs/pad/lib/python3.9/site-packages/dm_control/suite")
_FILENAMES = [
    Path("common/materials.xml"),
    Path("common/skybox.xml"),
    Path("common/visual.xml"),
]


def get_model_and_assets_from_setting_kwargs(model_fname, setting_kwargs=None):
    """Returns a tuple containing the model XML string and a dict of assets."""
    assets = {str(filename): resources.GetResource(_SUITE_DIR/filename)
          for filename in _FILENAMES}

    if setting_kwargs is None:
        return common.read_model(_SUITE_DIR/model_fname), assets

    # Convert XML to dicts
    model = xmltodict.parse(common.read_model(_SUITE_DIR/model_fname))
    materials = xmltodict.parse(assets['common/materials.xml'])
    skybox = xmltodict.parse(assets['common/skybox.xml'])

    # Edit cartpole
    if 'cartpole_length' in setting_kwargs:
        assert isinstance(setting_kwargs['cartpole_length'], (int, float))
        model['mujoco']['default']['default']['geom']['@fromto'] = \
            f"0 0 0 0 0 {setting_kwargs['cartpole_length']}"
    if 'cartpole_mass' in setting_kwargs:
        assert isinstance(setting_kwargs['cartpole_mass'], (int, float))
        model['mujoco']['default']['default']['geom']['@mass'] = \
            f"{setting_kwargs['cartpole_mass']}"

    # Edit grid floor
    if 'grid_rgb1' in setting_kwargs:
        assert isinstance(setting_kwargs['grid_rgb1'], (list, tuple, np.ndarray))
        materials['mujoco']['asset']['texture']['@rgb1'] = \
            f'{setting_kwargs["grid_rgb1"][0]} {setting_kwargs["grid_rgb1"][1]} {setting_kwargs["grid_rgb1"][2]}'
    if 'grid_rgb2' in setting_kwargs:
        assert isinstance(setting_kwargs['grid_rgb2'], (list, tuple, np.ndarray))
        materials['mujoco']['asset']['texture']['@rgb2'] = \
            f'{setting_kwargs["grid_rgb2"][0]} {setting_kwargs["grid_rgb2"][1]} {setting_kwargs["grid_rgb2"][2]}'

    # Edit self
    if 'self_rgb' in setting_kwargs:
        assert isinstance(setting_kwargs['self_rgb'], (list, tuple, np.ndarray))
        materials['mujoco']['asset']['material'][1]['@rgba'] = \
            f'{setting_kwargs["self_rgb"][0]} {setting_kwargs["self_rgb"][1]} {setting_kwargs["self_rgb"][2]} 1'

    # Edit skybox
    if 'skybox_rgb' in setting_kwargs:
        assert isinstance(setting_kwargs['skybox_rgb'], (list, tuple, np.ndarray))
        skybox['mujoco']['asset']['texture']['@rgb1'] = \
            f'{setting_kwargs["skybox_rgb"][0]} {setting_kwargs["skybox_rgb"][1]} {setting_kwargs["skybox_rgb"][2]}'
    if 'skybox_rgb2' in setting_kwargs:
        assert isinstance(setting_kwargs['skybox_rgb2'], (list, tuple, np.ndarray))
        skybox['mujoco']['asset']['texture']['@rgb2'] = \
            f'{setting_kwargs["skybox_rgb2"][0]} {setting_kwargs["skybox_rgb2"][1]} {setting_kwargs["skybox_rgb2"][2]}'
    if 'skybox_markrgb' in setting_kwargs:
        assert isinstance(setting_kwargs['skybox_markrgb'], (list, tuple, np.ndarray))
        skybox['mujoco']['asset']['texture']['@markrgb'] = \
            f'{setting_kwargs["skybox_markrgb"][0]} {setting_kwargs["skybox_markrgb"][1]} {setting_kwargs["skybox_markrgb"][2]}'

    # Convert back to XML
    model_xml = xmltodict.unparse(model)
    assets['common/materials.xml'] = xmltodict.unparse(materials)
    assets['common/skybox.xml'] = xmltodict.unparse(skybox)

    return model_xml, assets
    