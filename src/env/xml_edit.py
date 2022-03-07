from pathlib import Path
import numpy as np
from dm_control.suite import common
from dm_control.utils import io as resources
import xmltodict


_SUITE_DIR = Path("../../../../anaconda3/envs/pad/lib/python3.9/site-packages/dm_control/suite")
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
    model = xmltodict.parse(common.read_model(model_fname))
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
    if 'cartpole_size' in setting_kwargs:
        assert isinstance(setting_kwargs['cartpole_size'], (int, float))
        model['mujoco']['default']['default']['geom']['@size'] = \
            f"{setting_kwargs['cartpole_size']}"
    if 'cartpole_damping' in setting_kwargs:
        assert isinstance(setting_kwargs['cartpole_damping'], (int, float))
        model['mujoco']['default']['default']['joint']['@damping'] = \
            f"{setting_kwargs['cartpole_damping']}"

    
    # Edit cheetah
    if 'cheetah_leg_length' in setting_kwargs:
        assert isinstance(setting_kwargs['cheetah_leg_length'], (int, float))
        bthigh_size = model['mujoco']['worldbody']['body']['body'][0]['geom']['@size']
        bthigh_radius = float(bthigh_size.split(' ')[0])
        bthigh_length = float(bthigh_size.split(' ')[1])
        bshin_size = model['mujoco']['worldbody']['body']['body'][0]['body']['geom']['@size']
        bshin_radius = float(bshin_size.split(' ')[0])
        bshin_length = float(bshin_size.split(' ')[1])
        bfoot_size = model['mujoco']['worldbody']['body']['body'][0]['body']['body']['geom']['@size']
        bfoot_radius = float(bfoot_size.split(' ')[0])
        bfoot_length = float(bfoot_size.split(' ')[1])
        fthigh_size = model['mujoco']['worldbody']['body']['body'][1]['geom']['@size']
        fthigh_radius = float(fthigh_size.split(' ')[0])
        fthigh_length = float(fthigh_size.split(' ')[1])
        fshin_size = model['mujoco']['worldbody']['body']['body'][1]['body']['geom']['@size']
        fshin_radius = float(fshin_size.split(' ')[0])
        fshin_length = float(fshin_size.split(' ')[1])
        ffoot_size = model['mujoco']['worldbody']['body']['body'][1]['body']['body']['geom']['@size']
        ffoot_radius = float(ffoot_size.split(' ')[0])
        ffoot_length = float(ffoot_size.split(' ')[1])
        model['mujoco']['worldbody']['body']['body'][0]['geom']['@size'] = \
            f"{bthigh_radius} {bthigh_length * setting_kwargs['cheetah_leg_length']}"
        model['mujoco']['worldbody']['body']['body'][0]['body']['geom']['@size'] = \
            f"{bshin_radius} {bshin_length * setting_kwargs['cheetah_leg_length']}"
        model['mujoco']['worldbody']['body']['body'][0]['body']['body']['geom']['@size'] = \
            f"{bfoot_radius} {bfoot_length * setting_kwargs['cheetah_leg_length']}"
        model['mujoco']['worldbody']['body']['body'][1]['geom']['@size'] = \
            f"{fthigh_radius} {fthigh_length * setting_kwargs['cheetah_leg_length']}"
        model['mujoco']['worldbody']['body']['body'][1]['body']['geom']['@size'] = \
            f"{fshin_radius} {fshin_length * setting_kwargs['cheetah_leg_length']}"
        model['mujoco']['worldbody']['body']['body'][1]['body']['body']['geom']['@size'] = \
            f"{ffoot_radius} {ffoot_length * setting_kwargs['cheetah_leg_length']}"
    if 'cheetah_mass' in setting_kwargs:
        assert isinstance(setting_kwargs['cheetah_mass'], (int, float))
        model['mujoco']['compiler']['@settotalmass'] = \
            f"{setting_kwargs['cheetah_mass']}"
    if 'cheetah_ground_friction' in setting_kwargs:
        assert isinstance(setting_kwargs['cheetah_ground_friction'], (int, float))
        model['mujoco']['worldbody']['geom']['@friction'] = \
            f"{setting_kwargs['cheetah_ground_friction']} 0.005 0.0001"

    # Edit walker
    if 'walker_torso_length' in setting_kwargs:
        assert isinstance(setting_kwargs['walker_torso_length'], (int, float))
        model['mujoco']['worldbody']['body']['geom']['@size'] = \
            f"0.07 {setting_kwargs['walker_torso_length']}"
    if 'walker_ground_friction' in setting_kwargs:
        assert isinstance(setting_kwargs['walker_ground_friction'], (int, float))
        model['mujoco']['worldbody']['geom']['@friction'] = \
            f"{setting_kwargs['walker_ground_friction']} 0.005 0.0001"

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
    