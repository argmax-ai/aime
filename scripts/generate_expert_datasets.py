import logging as log
import os
from argparse import ArgumentParser

import torch
from omegaconf import OmegaConf

from aime.actor import PolicyActor
from aime.env import DMC, SaveTrajectories, TerminalSummaryWrapper
from aime.models.policy import TanhGaussianPolicy
from aime.models.ssm import ssm_classes
from aime.utils import (
    get_image_sensors,
    interact_with_environment,
    parse_world_model_config,
    setup_seed,
)

log.basicConfig(level=log.INFO)


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_folder", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_trajectories", type=int, default=100)
    args = parser.parse_args()

    setup_seed(args.seed)

    config = OmegaConf.load(os.path.join(args.model_path, "config.yaml"))
    if args.output_folder is None:
        expert_folder = os.path.join(args.model_path, "expert_trajectories")
    else:
        expert_folder = args.output_folder

    env_config = config["env"]
    test_env = DMC(
        env_config["name"], action_repeat=env_config["action_repeat"], seed=args.seed
    )
    test_env = SaveTrajectories(test_env, expert_folder)
    test_env = TerminalSummaryWrapper(test_env)

    data = test_env.observation_space

    sensor_layout = env_config["sensors"]
    world_model_config = parse_world_model_config(config, sensor_layout, data)
    world_model_name = world_model_config.pop("name")
    image_sensors, _ = get_image_sensors(world_model_config, sensor_layout)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"using device {device}")
    model = ssm_classes[world_model_name](**world_model_config)
    model.load_state_dict(
        torch.load(os.path.join(args.model_path, "model.pt"), map_location="cpu"),
        strict=False,
    )
    model = model.to(device)

    # load the pretrained policy
    log.info("Creating and loading policy ...")
    policy_config = config["policy"]
    policy = TanhGaussianPolicy(
        model.state_feature_dim, world_model_config["action_dim"], **policy_config
    )
    policy.load_state_dict(
        torch.load(os.path.join(args.model_path, "policy.pt"), map_location="cpu")
    )
    policy = policy.to(device)

    # directly test this model and policy on the new task
    log.info("Generating expert trajectories ...")
    with torch.no_grad():
        actor = PolicyActor(model, policy)
        for _ in range(args.num_trajectories):
            interact_with_environment(test_env, actor, image_sensors)


if __name__ == "__main__":
    main()
