import logging
import os
import random

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf

from aime.data import ArrayDict

log = logging.getLogger("utils")


def setup_seed(seed=42):
    """Fix the common random source in deep learning programs"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    log.info(f"global seed is set to {seed}")


class AverageMeter:
    """Class to collect and average a sequence of metrics"""

    def __init__(self) -> None:
        self.storage = None

    def add(self, metrics):
        if self.storage is None:
            self.storage = {k: [v] for k, v in metrics.items()}
        else:
            for k in metrics.keys():
                self.storage[k].append(metrics[k])

    def get(
        self,
    ):
        if self.storage is None:
            return {}
        return {k: np.mean(v) for k, v in self.storage.items()}


def get_sensor_shapes(example_data):
    shapes = {}
    for k, v in example_data.items():
        shape = v.shape
        if len(shape) == 1 or len(shape) == 2:
            shapes[k] = shape[-1]
        elif len(shape) == 3 or len(shape) == 4:
            shapes[k] = shape[-2:]
    return shapes


def get_inputs_outputs(sensor_layout, environment_setup):
    assert environment_setup in ["lpomdp", "pomdp", "mdp", "exp", "visual", "full", "real"]
    if environment_setup == "mdp":
        input_sensors = [
            k for k, v in sensor_layout.items() if v["modility"] == "tabular"
        ]
        output_sensors = input_sensors
        probe_sensors = []
    elif environment_setup == "lpomdp" or environment_setup == "pomdp":
        input_sensors = [
            k
            for k, v in sensor_layout.items()
            if v["modility"] == "tabular" and v["order"] == "first"
        ]
        output_sensors = input_sensors
        probe_sensors = [
            k
            for k, v in sensor_layout.items()
            if v["modility"] == "tabular" and v["order"] == "second"
        ]
    elif environment_setup == "visual":
        input_sensors = [
            k for k, v in sensor_layout.items() if v["modility"] == "visual"
        ]
        output_sensors = input_sensors
        probe_sensors = [
            k for k, v in sensor_layout.items() if v["modility"] == "tabular"
        ]

    return input_sensors, output_sensors, probe_sensors


def parse_world_model_config(config, sensor_layout, example_data, predict_reward=True):
    world_model_config = dict(config["world_model"])
    input_sensors, output_sensors, probe_sensors = get_inputs_outputs(
        sensor_layout, config["environment_setup"]
    )
    sensor_shapes = get_sensor_shapes(example_data)
    sensor_layout = dict(sensor_layout)
    encoder_configs = world_model_config.pop("encoders")
    decoder_configs = world_model_config.pop("decoders")
    probe_configs = world_model_config.pop("probes")
    world_model_config["input_config"] = [
        (k, sensor_shapes[k], dict(encoder_configs[sensor_layout[k]["modility"]]))
        for k in input_sensors
    ]
    world_model_config["output_config"] = [
        (k, sensor_shapes[k], dict(decoder_configs[sensor_layout[k]["modility"]]))
        for k in output_sensors
    ]
    if predict_reward:
        world_model_config["output_config"] = world_model_config["output_config"] + [
            ("reward", 1, dict(decoder_configs["tabular"]))
        ]
    world_model_config["probe_config"] = [
        (k, sensor_shapes[k], dict(probe_configs[sensor_layout[k]["modility"]]))
        for k in probe_sensors
    ]
    world_model_config["action_dim"] = sensor_shapes["pre_action"]
    return world_model_config


def get_image_sensors(world_model_config, sensor_layout):
    image_sensors = [k for k, v in sensor_layout.items() if v["modility"] == "visual"]
    used_sensors = [config[0] for config in world_model_config["output_config"]]
    used_image_sensors = [
        image_sensor for image_sensor in image_sensors if image_sensor in used_sensors
    ]
    return image_sensors, used_image_sensors

def load_pretrained_model(model_root):
    from aime.env import env_classes
    from aime.models.ssm import ssm_classes

    config = OmegaConf.load(os.path.join(model_root, 'config.yaml'))
    env_config = config["env"]
    env_class_name = env_config["class"]
    env = env_classes[env_class_name](
        env_config["name"],
        action_repeat=env_config["action_repeat"],
        seed=config["seed"],
        render=True
    )

    sensor_layout = env_config["sensors"]
    world_model_config = parse_world_model_config(config, sensor_layout, env.observation_space, False)
    world_model_name = world_model_config.pop("name")
    model = ssm_classes[world_model_name](**world_model_config)
    model.load_state_dict(
        torch.load(os.path.join(model_root, "model.pt"), map_location="cpu")
    )

    return model

def need_render(environment_setup: str):
    """determine whether the render is a must during training"""
    return environment_setup in ["visual", "full", "real"]


def interact_with_environment(env, actor, image_sensors) -> float:
    """interact a environment with an actor for one trajectory"""
    obs = env.reset()
    actor.reset()
    reward = 0
    while not obs.get("is_last", False) and not obs.get("is_terminal", False):
        for image_key in image_sensors:
            if image_key in obs.keys():
                obs[image_key] = rearrange(obs[image_key], "h w c -> c h w") / 255.0
        action = actor(obs)
        obs = env.step(action)
        reward += obs["reward"]
    return reward


@torch.no_grad()
def generate_prediction_videos(
    model,
    data,
    env,
    all_image_sensors,
    used_image_sensors,
    filter_step: int = 10,
    samples: int = 6,
    custom_action_seq=None,
):
    videos = {}
    data = data[:, :samples]
    data.vmap_(lambda x: x.contiguous())
    pre_action_seq = (
        data["pre_action"]
        if custom_action_seq is None
        else custom_action_seq[:, :samples]
    )
    predicted_obs_seq, _, _, _ = model(data, pre_action_seq, filter_step=filter_step)
    if len(used_image_sensors) == 0:
        # one must render the scene from other signals
        some_key = list(predicted_obs_seq.keys())[0]
        some_value = predicted_obs_seq[some_key][..., 0]
        t, b = predicted_obs_seq[some_key].shape[:2]
        predicted_obs_seq.to_numpy()
        image_obs_seq = []
        for i in range(t):
            _image_obs_seq = []
            for j in range(b):
                obs = predicted_obs_seq[i, j]
                env.set_state_from_obs(obs)
                _image_obs_seq.append(ArrayDict(env.render()))
            image_obs_seq.append(ArrayDict.stack(_image_obs_seq, dim=0))
        image_obs_seq = ArrayDict.stack(image_obs_seq, dim=0)
        image_obs_seq.to_torch()
        for image_key in image_obs_seq.keys():
            image_obs_seq[image_key] = (
                rearrange(image_obs_seq[image_key], "t b h w c -> t b c h w") / 255.0
            )
        predicted_obs_seq.to_torch()
        predicted_obs_seq.update(image_obs_seq)
        predicted_obs_seq.to(some_value)

        data.to_numpy()
        image_obs_seq = []
        for i in range(t):
            _image_obs_seq = []
            for j in range(b):
                obs = data[i, j]
                env.set_state_from_obs(obs)
                _image_obs_seq.append(ArrayDict(env.render()))
            image_obs_seq.append(ArrayDict.stack(_image_obs_seq, dim=0))
        image_obs_seq = ArrayDict.stack(image_obs_seq, dim=0)
        image_obs_seq.to_torch()
        for image_key in image_obs_seq.keys():
            image_obs_seq[image_key] = (
                rearrange(image_obs_seq[image_key], "t b h w c -> t b c h w") / 255.0
            )
        data.to_torch()
        data.update(image_obs_seq)
        data.to(some_value)

    for image_key in all_image_sensors:
        if image_key not in predicted_obs_seq.keys():
            continue
        gt_video = data[image_key]
        pred_video = predicted_obs_seq[image_key]
        diff_video = (gt_video - pred_video) / 2 + 0.5
        log_video = torch.cat([gt_video, pred_video, diff_video], dim=1)
        log_video = rearrange(log_video, "t (m b) c h w -> t (m h) (b w) c", m=3) * 255
        videos[f"rollout_video_{image_key}"] = log_video

    return videos


@torch.no_grad()
def eval_prediction(
    model,
    data,
    filter_step: int = 10,
):
    metrics = {}
    pre_action_seq = data["pre_action"]
    predicted_obs_seq, _, _, _ = model(data, pre_action_seq, filter_step=filter_step)

    for name in model.decoders.keys():
        metrics[f"prediction_{name}_mse"] = torch.nn.MSELoss()(
            predicted_obs_seq[name][filter_step:], data[name][filter_step:]
        ).item()

    return metrics


@torch.jit.script
def lambda_return(reward, value, discount, bootstrap, lambda_: float):
    """
    Modify from https://github.com/danijar/dreamer/blob/master/tools.py,
    Setting lambda=1 gives a discounted Monte Carlo return.
    Setting lambda=0 gives a fixed 1-step return.
    """
    next_values = torch.cat([value[1:], bootstrap[None]], dim=0)
    inputs = reward + discount * next_values * (1 - lambda_)
    returns = []
    curr_value = bootstrap
    for t in reversed(torch.arange(len(value))):
        curr_value = inputs[t] + lambda_ * discount[t] * curr_value
        returns.append(curr_value)
    returns = torch.stack(returns)
    returns = torch.flip(returns, dims=[0])
    return returns


CONFIG_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "configs")
OUTPUT_PATH = "logs"
DATA_PATH = "datasets"
MODEL_PATH = "pretrained-models"

EXPERT_PERFORMANCE = {
    'walker-stand' : 957.87109375,
    'walker-walk' : 943.7876586914062,
    'walker-run' : 604.10009765625,
    'cheetah-run' : 888.6480102539062,
    'cheetah-runbackward' : 218.50271606445312,
    'cheetah-flip' : 485.7939758300781,
    'cheetah-flipbackward' : 379.9083251953125,
}
