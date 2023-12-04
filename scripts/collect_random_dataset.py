import logging as log
import os
from argparse import ArgumentParser

from omegaconf import OmegaConf

from aime.actor import RandomActor
from aime.env import DMC, SaveTrajectories, TerminalSummaryWrapper
from aime.utils import CONFIG_PATH, DATA_PATH, interact_with_environment, setup_seed

log.basicConfig(level=log.INFO)


def main():
    parser = ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--num_trajectories", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_seed(args.seed)
    env_config = OmegaConf.load(os.path.join(CONFIG_PATH, "env", args.env + ".yaml"))
    output_folder = os.path.join(
        DATA_PATH,
        f'{env_config["name"]}-random-nt{args.num_trajectories}-ar{env_config["action_repeat"]}-s{args.seed}',
    )

    env = DMC(
        env_config["name"], action_repeat=env_config["action_repeat"], seed=args.seed
    )
    env = SaveTrajectories(env, output_folder)
    env = TerminalSummaryWrapper(env)
    env.action_space.seed(args.seed)

    actor = RandomActor(env.action_space)
    for i in range(args.num_trajectories):
        interact_with_environment(env, actor, [])


if __name__ == "__main__":
    main()
