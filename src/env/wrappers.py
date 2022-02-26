import numpy as np
from numpy.random import randint
import os
import gym
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import dmc2gym
from env.xml_edit import get_model_and_assets_from_setting_kwargs
import cv2
from collections import deque


def make_pad_env(
	domain_name,
	task_name,
	seed=0,
	episode_length=1000,
	frame_stack=3,
	action_repeat=4,
	mode='train',
	use_state_vector=False
):
	"""Make environment for PAD experiments"""
	from_pixels = not use_state_vector
	env = dmc2gym.make(
		domain_name=domain_name,
		task_name=task_name,
		seed=seed,
		visualize_reward=False,
		from_pixels=from_pixels,
		height=100,
		width=100,
		episode_length=episode_length,
		frame_skip=action_repeat
	)
	env.seed(seed)
	env = GreenScreen(env, mode)
	env = FrameStack(env, frame_stack, domain_name)
	if from_pixels and 'color' in mode:
		env = ColorWrapper(env, mode)
	if 'cartpole' in mode:
		env = CartpoleWrapper(env, mode)
	if 'cheetah' in mode:
		env = CheetahWrapper(env, mode)

	assert env.action_space.low.min() >= -1
	assert env.action_space.high.max() <= 1

	return env


class CartpoleWrapper(gym.Wrapper):
	"""Wrapper for the cartpole length experiments"""
	def __init__(self, env, mode):
		assert isinstance(env, FrameStack), 'wrapped env must be a framestack'
		gym.Wrapper.__init__(self, env)
		self._max_episode_steps = env._max_episode_steps
		self._mode = mode
		self.time_step = 0
		self._iteration = 0
		self._lengths = [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5]
		self._masses = [0.0001, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
		self._sizes = [0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
		self._damping = [6e-3, 7e-3, 8e-3, 9e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 1e-1]
	
	def reset(self):
		self.time_step = 0
		if 'cartpole' in self._mode:
			self.next()
		return self.env.reset()

	def step(self, action):
		self.time_step += 1
		return self.env.step(action)

	def next(self):
		assert 'cartpole' in self._mode, f'can only set cartpole parameters, received {self._mode}'		
		if (self._mode == 'cartpole_length'):
			assert self._iteration < len(self._lengths), f'too many eval episodes'
			self.reload_physics({'cartpole_length': self._lengths[self._iteration]})
		elif (self._mode == 'cartpole_mass'):
			assert self._iteration < len(self._masses), f'too many eval episodes'
			self.reload_physics({'cartpole_mass': self._masses[self._iteration]})
		elif (self._mode == 'cartpole_size'):
			assert self._iteration < len(self._sizes), f'too many eval episodes'
			self.reload_physics({'cartpole_size': self._sizes[self._iteration]})
		elif (self._mode == 'cartpole_damping'):
			assert self._iteration < len(self._damping), f'too many eval episodes'
			self.reload_physics({'cartpole_damping': self._damping[self._iteration]})
		self._iteration += 1


class CheetahWrapper(gym.Wrapper):
	"""Wrapper for the cheetah experiments"""
	def __init__(self, env, mode):
		assert isinstance(env, FrameStack), 'wrapped env must be a framestack'
		gym.Wrapper.__init__(self, env)
		self._max_episode_steps = env._max_episode_steps
		self._mode = mode
		self.time_step = 0
		self._iteration = 0
		self._length_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.11, 1.25, 1.43, 1.67, 2.0]
		self._masses = [10, 11, 12, 13, 15, 16, 17, 18]
		self._ground_friction = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0]

	def reset(self):
		self.time_step = 0
		if 'cheetah' in self._mode:
			self.next()
		return self.env.reset()

	def step(self, action):
		self.time_step += 1
		return self.env.step(action)

	def next(self):
		assert 'cheetah' in self._mode, f'can only set cheetah parameters, received {self._mode}'		
		if (self._mode == 'cheetah_leg_length'):
			assert self._iteration < len(self._length_factors), f'too many eval episodes'
			self.reload_physics({'cheetah_leg_length': self._length_factors[self._iteration]})
		elif (self._mode == 'cheetah_mass'):
			assert self._iteration < len(self._masses), f'too many eval episodes'
			self.reload_physics({'cheetah_mass': self._masses[self._iteration]})
		elif (self._mode == 'cheetah_ground_friction'):
			assert self._iteration < len(self._ground_friction), f'too many eval episodes'
			self.reload_physics({'cheetah_ground_friction': self._ground_friction[self._iteration]})
		self._iteration += 1


class WalkerWrapper(gym.Wrapper):
    """Wrapper for the walker experiments"""
    def __init__(self, env, mode):
        assert isinstance(env, FrameStack), 'wrapped env must be a framestack'
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps
        self._mode = mode
        self.time_step = 0
        self._iteration = 0
        self._torso_length = [0.1, 0.15, 0.2, 0.25, 0.35, 0.4, 0.45, 0.5]
        self._ground_friction = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0]
    
    def reset(self):
        self.time_step = 0
        if 'walker' in self._mode:
            self.next()
        return self.env.reset()
    
    def step(self, action):
        self.time_step += 1
        return self.env.step(action)

    def next(self):
        assert 'walker' in self._mode, f'can only set walker parameters, received {self._mode}'
        if (self._mode == 'walker_torso_length'):
            assert self._iteration < len(self._torso_length), f'too many eval episodes'
            self.reload_physics({'walker_torso_length': self._torso_length[self._iteration]})
        elif (self._mode == 'walker_ground_friction'):
            assert self._iteration < len(self._ground_friction), f'too many eval episodes'
            self.reload_physics({'walker_ground_friction': self._ground_friction[self._iteration]})
            self._iteration += 1


class ColorWrapper(gym.Wrapper):
	"""Wrapper for the color experiments"""
	def __init__(self, env, mode):
		assert isinstance(env, FrameStack), 'wrapped env must be a framestack'
		gym.Wrapper.__init__(self, env)
		self._max_episode_steps = env._max_episode_steps
		self._mode = mode
		self.time_step = 0
		if 'color' in self._mode:
			self._load_colors()
	
	def reset(self):
		self.time_step = 0
		if 'color' in self._mode:
			self.randomize()
		if 'video' in self._mode:
			# apply greenscreen
			self.reload_physics(
				{'skybox_rgb': [.2, .8, .2],
				'skybox_rgb2': [.2, .8, .2],
				'skybox_markrgb': [.2, .8, .2]
			})
		return self.env.reset()

	def step(self, action):
		self.time_step += 1
		return self.env.step(action)

	def randomize(self):
		assert 'color' in self._mode, f'can only randomize in color mode, received {self._mode}'		
		self.reload_physics(self.get_random_color())

	def _load_colors(self):
		assert self._mode in {'color_easy', 'color_hard'}
		self._colors = torch.load(f'src/env/data/{self._mode}.pt')

	def get_random_color(self):
		assert len(self._colors) >= 100, 'env must include at least 100 colors'
		return self._colors[randint(len(self._colors))]


class FrameStack(gym.Wrapper):
	"""Stack frames as observation"""
	def __init__(self, env, k, domain_name):
		gym.Wrapper.__init__(self, env)
		self._k = k
		self._frames = deque([], maxlen=k)
		self._state_vectors = deque([], maxlen=k)
		self.domain_name = domain_name
		obs_shape = env.observation_space.shape
		self.observation_space = gym.spaces.Box(
			low=0,
			high=1,
			shape=((obs_shape[0] * k,) + obs_shape[1:]),
			dtype=env.observation_space.dtype
		)
		self._max_episode_steps = env._max_episode_steps

	def reset(self):
		obs = self.env.reset()
		for _ in range(self._k):
			self._frames.append(obs)
		return self._get_obs()

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		self._frames.append(obs)
		return self._get_obs(), reward, done, info
		
	def _get_obs(self):
		assert len(self._frames) == self._k
		return np.concatenate(list(self._frames), axis=0)
	
	def reload_physics(self, setting_kwargs=None, state=None):
		domain_name = self._get_dmc_wrapper()._domain_name
		if setting_kwargs is None:
			setting_kwargs = {}
		if state is None:
			state = self._get_state()
		self._reload_physics(
			*get_model_and_assets_from_setting_kwargs(
				domain_name+'.xml', setting_kwargs
			)
		)
		self._set_state(state)
	
	def get_state(self):
		return self._get_state()
	
	def set_state(self, state):
		self._set_state(state)

	def _get_dmc_wrapper(self):
		_env = self.env
		while not isinstance(_env, dmc2gym.wrappers.DMCWrapper) and hasattr(_env, 'env'):
			_env = _env.env
		assert isinstance(_env, dmc2gym.wrappers.DMCWrapper), 'environment is not dmc2gym-wrapped'
		return _env

	def _reload_physics(self, xml_string, assets=None):
		_env = self.env
		while not hasattr(_env, '_physics') and hasattr(_env, 'env'):
			_env = _env.env
		assert hasattr(_env, '_physics'), 'environment does not have physics attribute'
		_env.physics.reload_from_xml_string(xml_string, assets=assets)

	def _get_physics(self):
		_env = self.env
		while not hasattr(_env, '_physics') and hasattr(_env, 'env'):
			_env = _env.env
		assert hasattr(_env, '_physics'), 'environment does not have physics attribute'

		return _env._physics

	def _get_state(self):
		return self._get_physics().get_state()
		
	def _set_state(self, state):
		self._get_physics().set_state(state)

def rgb_to_hsv(r, g, b):
	"""Convert RGB color to HSV color"""
	maxc = max(r, g, b)
	minc = min(r, g, b)
	v = maxc
	if minc == maxc:
		return 0.0, 0.0, v
	s = (maxc-minc) / maxc
	rc = (maxc-r) / (maxc-minc)
	gc = (maxc-g) / (maxc-minc)
	bc = (maxc-b) / (maxc-minc)
	if r == maxc:
		h = bc-gc
	elif g == maxc:
		h = 2.0+rc-bc
	else:
		h = 4.0+gc-rc
	h = (h/6.0) % 1.0
	return h, s, v


def do_green_screen(x, bg):
	"""Removes green background from observation and replaces with bg; not optimized for speed"""
	assert isinstance(x, np.ndarray) and isinstance(bg, np.ndarray), 'inputs must be numpy arrays'
	assert x.dtype == np.uint8 and bg.dtype == np.uint8, 'inputs must be uint8 arrays'
	
	# Get image sizes
	x_h, x_w = x.shape[1:]

	# Convert to RGBA images
	im = TF.to_pil_image(torch.ByteTensor(x))
	im = im.convert('RGBA')
	pix = im.load()
	bg = TF.to_pil_image(torch.ByteTensor(bg))
	bg = bg.convert('RGBA')
	bg = bg.load()

	# Replace pixels
	for x in range(x_w):
		for y in range(x_h):
			r, g, b, a = pix[x, y]
			h_ratio, s_ratio, v_ratio = rgb_to_hsv(r / 255., g / 255., b / 255.)
			h, s, v = (h_ratio * 360, s_ratio * 255, v_ratio * 255)

			min_h, min_s, min_v = (100, 80, 70)
			max_h, max_s, max_v = (185, 255, 255)
			if min_h <= h <= max_h and min_s <= s <= max_s and min_v <= v <= max_v:
				pix[x, y] = bg[x, y]

	x = np.moveaxis(np.array(im).astype(np.uint8), -1, 0)[:3]

	return x


class GreenScreen(gym.Wrapper):
	"""Green screen for video experiments"""
	def __init__(self, env, mode):
		gym.Wrapper.__init__(self, env)
		self._mode = mode
		if 'video' in mode:
			self._video = mode
			if not self._video.endswith('.mp4'):
				self._video += '.mp4'
			self._video = os.path.join('src/env/data', self._video)
			self._data = self._load_video(self._video)
		else:
			self._video = None
		self._max_episode_steps = env._max_episode_steps

	def _load_video(self, video):
		"""Load video from provided filepath and return as numpy array"""
		cap = cv2.VideoCapture(video)
		assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) >= 100, 'width must be at least 100 pixels'
		assert cap.get(cv2.CAP_PROP_FRAME_HEIGHT) >= 100, 'height must be at least 100 pixels'
		n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		buf = np.empty((n, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), np.dtype('uint8'))
		i, ret = 0, True
		while (i < n  and ret):
			ret, frame = cap.read()
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			buf[i] = frame
			i += 1
		cap.release()
		return np.moveaxis(buf, -1, 1)

	def reset(self):
		self._current_frame = 0
		return self._greenscreen(self.env.reset())

	def step(self, action):
		self._current_frame += 1
		obs, reward, done, info = self.env.step(action)
		return self._greenscreen(obs), reward, done, info
	
	def _interpolate_bg(self, bg, size:tuple):
		"""Interpolate background to size of observation"""
		bg = torch.from_numpy(bg).float().unsqueeze(0) / 255
		bg = F.interpolate(bg, size=size, mode='bilinear', align_corners=False)
		return (bg*255).byte().squeeze(0).numpy()

	def _greenscreen(self, obs):
		"""Applies greenscreen if video is selected, otherwise does nothing"""
		if self._video:
			bg = self._data[self._current_frame % len(self._data)] # select frame
			bg = self._interpolate_bg(bg, obs.shape[1:]) # scale bg to observation size
			return do_green_screen(obs, bg) # apply greenscreen
		return obs

	def apply_to(self, obs):
		"""Applies greenscreen mode of object to observation"""
		obs = obs.copy()
		channels_last = obs.shape[-1] == 3
		if channels_last:
			obs = torch.from_numpy(obs).permute(2,0,1).numpy()
		obs = self._greenscreen(obs)
		if channels_last:
			obs = torch.from_numpy(obs).permute(1,2,0).numpy()
		return obs
