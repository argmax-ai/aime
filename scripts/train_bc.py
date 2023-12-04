import logging
import os

import hydra
import numpy as np
import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from aime.actor import StackPolicyActor
from aime.data import NPZFolder, get_epoch_loader
from aime.env import SaveTrajectories, TerminalSummaryWrapper, env_classes
from aime.logger import get_default_logger
from aime.models.base import MLP, MultimodalEncoder
from aime.utils import (
    CONFIG_PATH,
    DATA_PATH,
    OUTPUT_PATH,
    AverageMeter,
    get_inputs_outputs,
    get_sensor_shapes,
    interact_with_environment,
    setup_seed,
)

log = logging.getLogger("main")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="bc")
def main(config: DictConfig):
    setup_seed(config["seed"])

    log.info("using the following config:")
    log.info(config)

    stack = 1 if config["environment_setup"] == "mdp" else config["stack"]

    log_name = config["log_name"]
    output_folder = os.path.join(OUTPUT_PATH, log_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    OmegaConf.save(config, os.path.join(output_folder, "config.yaml"))
    demonstration_dataset_folder = os.path.join(
        DATA_PATH, config["demonstration_dataset_name"]
    )
    eval_folder = os.path.join(output_folder, "eval_trajectories")

    log.info("Creating environment ...")
    env_config = config["env"]
    env_class_name = env_config["class"]
    test_env = env_classes[env_class_name](
        env_config["name"],
        action_repeat=env_config["action_repeat"],
        seed=config["seed"] * 2,
        render=True,
    )
    test_env = SaveTrajectories(test_env, eval_folder)
    test_env = TerminalSummaryWrapper(test_env)

    log.info("Loading datasets ...")
    demonstration_dataset = NPZFolder(
        demonstration_dataset_folder, stack + 1, overlap=True
    )
    demonstration_dataset.keep(config["num_expert_trajectories"])
    eval_dataset = NPZFolder(eval_folder, stack + 1, overlap=False)
    data = demonstration_dataset[0]

    log.info("Creating models ...")
    sensor_layout = env_config["sensors"]
    encoder_configs = config["encoders"]
    sensor_shapes = get_sensor_shapes(data)
    input_sensors, _, _ = get_inputs_outputs(sensor_layout, config["environment_setup"])
    multimodal_encoder_config = [
        (k, sensor_shapes[k], dict(encoder_configs[sensor_layout[k]["modility"]]))
        for k in input_sensors
    ]
    image_sensors = [k for k, v in sensor_layout.items() if v["modility"] == "visual"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"using device {device}")

    policy_encoder = MultimodalEncoder(multimodal_encoder_config)
    policy_encoder = policy_encoder.to(device)

    policy_config = config["policy"]
    policy = MLP(
        policy_encoder.output_dim * stack,
        sensor_shapes["pre_action"],
        output_activation="tanh",
        **policy_config,
    )
    policy = policy.to(device)

    loss_fn = torch.nn.MSELoss()

    logger = get_default_logger(output_folder)

    policy_optim = torch.optim.Adam(
        [*policy.parameters(), *policy_encoder.parameters()], lr=config["policy_lr"]
    )

    log.info("Training Policy ...")
    train_size = int(
        len(demonstration_dataset) * config["train_validation_split_ratio"]
    )
    val_size = len(demonstration_dataset) - train_size
    (
        demonstration_dataset_train,
        demonstration_dataset_val,
    ) = torch.utils.data.random_split(demonstration_dataset, [train_size, val_size])
    train_loader = get_epoch_loader(
        demonstration_dataset_train, config["batch_size"], shuffle=True, num_workers=4
    )
    val_loader = get_epoch_loader(
        demonstration_dataset_val, config["batch_size"], shuffle=False, num_workers=4
    )
    e = 0
    best_val_loss = float("inf")
    convergence_count = 0
    while True:
        log.info(f"Starting epcoh {e}")

        metrics = {}
        train_metric_tracker = AverageMeter()
        for data in tqdm(iter(train_loader)):
            data = data.to(device)
            emb_policy = policy_encoder(data[:-1])
            emb_policy = rearrange(emb_policy, "t b f -> b (t f)")
            policy_action = policy(emb_policy)
            loss = loss_fn(data[-1]["pre_action"], policy_action)

            policy_optim.zero_grad()
            loss.backward()
            policy_optim.step()

            metric = {
                "train/policy_loss": loss.item(),
            }

            train_metric_tracker.add(metric)

        metrics.update(train_metric_tracker.get())

        val_metric_tracker = AverageMeter()
        with torch.no_grad():
            for data in tqdm(iter(val_loader)):
                data = data.to(device)
                emb_policy = policy_encoder(data[:-1])
                emb_policy = rearrange(emb_policy, "t b f -> b (t f)")
                policy_action = policy(emb_policy)
                loss = loss_fn(data[-1]["pre_action"], policy_action)

                metric = {
                    "val/policy_loss": loss.item(),
                }

                val_metric_tracker.add(metric)

        metrics.update(val_metric_tracker.get())

        log.info("Evaluating the model ...")
        with torch.no_grad():
            actor = StackPolicyActor(policy_encoder, policy, stack)
            reward = interact_with_environment(test_env, actor, image_sensors)
            metrics["eval_reward"] = reward
        eval_dataset.update()
        for image_key in image_sensors:
            metrics[f"eval_video_{image_key}"] = (
                eval_dataset.trajectories[-1][image_key]
                .permute(0, 2, 3, 1)
                .contiguous()
                * 255
            )

        logger(metrics, e)

        e += 1

        if metrics["val/policy_loss"] < best_val_loss:
            best_val_loss = metrics["val/policy_loss"]
            torch.save(
                policy_encoder.state_dict(),
                os.path.join(output_folder, "policy_encoder.pt"),
            )
            torch.save(policy.state_dict(), os.path.join(output_folder, "policy.pt"))
            convergence_count = 0
        else:
            convergence_count += 1
            if (
                convergence_count >= config["patience"]
                and e >= config["min_policy_epoch"]
            ):
                break

    log.info(f"Policy training finished in {e} epoches!")

    # restore the best policy for a final test
    policy_encoder.load_state_dict(
        torch.load(os.path.join(output_folder, "policy_encoder.pt"))
    )
    policy.load_state_dict(torch.load(os.path.join(output_folder, "policy.pt")))

    metrics = {}
    with torch.no_grad():
        actor = StackPolicyActor(policy_encoder, policy, stack)
        rewards = [
            interact_with_environment(test_env, actor, image_sensors)
            for _ in range(config["num_test_trajectories"])
        ]
        metrics["eval_reward_raw"] = rewards
        metrics["eval_reward"] = np.mean(rewards)
        metrics["eval_reward_std"] = np.std(rewards)
    eval_dataset.update()
    for image_key in image_sensors:
        metrics[f"eval_video_{image_key}"] = (
            eval_dataset.trajectories[-1][image_key].permute(0, 2, 3, 1).contiguous()
            * 255
        )

    logger(metrics, e)


if __name__ == "__main__":
    main()
