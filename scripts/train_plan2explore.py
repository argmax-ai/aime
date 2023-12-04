import logging
import os
import time

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from aime.actor import GuassianNoiseActorWrapper, PolicyActor, RandomActor
from aime.data import NPZFolder, get_sample_loader
from aime.env import SaveTrajectories, TerminalSummaryWrapper, env_classes
from aime.logger import get_default_logger
from aime.models.base import MLP
from aime.models.policy import TanhGaussianPolicy
from aime.models.ssm import ssm_classes
from aime.utils import (
    CONFIG_PATH,
    MODEL_PATH,
    OUTPUT_PATH,
    AverageMeter,
    generate_prediction_videos,
    get_image_sensors,
    interact_with_environment,
    lambda_return,
    need_render,
    parse_world_model_config,
    setup_seed,
)

log = logging.getLogger("main")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="plan2explore")
def main(config: DictConfig):
    setup_seed(config["seed"])

    log.info("using the following config:")
    log.info(config)

    log_name = config["log_name"]
    output_folder = os.path.join(OUTPUT_PATH, log_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    OmegaConf.save(config, os.path.join(output_folder, "config.yaml"))
    dataset_folder = os.path.join(output_folder, "train_trajectories")
    eval_folder = os.path.join(output_folder, "eval_trajectories")

    env_config = config["env"]
    env_class_name = env_config["class"]
    env = env_classes[env_class_name](
        env_config["name"],
        action_repeat=env_config["action_repeat"],
        seed=config["seed"],
        render=env_config["render"] or need_render(config["environment_setup"]),
    )
    env = SaveTrajectories(env, dataset_folder)
    env = TerminalSummaryWrapper(env)
    env.action_space.seed(config["seed"])
    test_env = env_classes[env_class_name](
        env_config["name"],
        action_repeat=env_config["action_repeat"],
        seed=config["seed"] * 2,
        render=True,
    )
    test_env = SaveTrajectories(test_env, eval_folder)
    test_env = TerminalSummaryWrapper(test_env)

    # collect initial dataset
    for _ in range(config["prefill"]):
        actor = RandomActor(env.action_space)
        interact_with_environment(env, actor, [])

    dataset = NPZFolder(dataset_folder, config["horizon"], overlap=True)
    eval_dataset = NPZFolder(eval_folder, config["horizon"], overlap=False)
    data = dataset[0]

    sensor_layout = env_config["sensors"]
    world_model_config = parse_world_model_config(config, sensor_layout, data)
    world_model_name = world_model_config.pop("name")
    image_sensors, used_image_sensors = get_image_sensors(
        world_model_config, sensor_layout
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"using device {device}")
    model = ssm_classes[world_model_name](**world_model_config)
    if config["pretrained_model_name"] is not None:
        model.load_state_dict(
            torch.load(
                os.path.join(MODEL_PATH, config["pretrained_model_name"], "model.pt"),
                map_location="cpu",
            ),
            strict=False,
        )
    model.decoders.pop("reward")
    model = model.to(device)

    policy_config = config["policy"]
    policy = TanhGaussianPolicy(
        model.state_feature_dim, world_model_config["action_dim"], **policy_config
    )
    policy = policy.to(device)

    vnet_config = config["vnet"]
    vnet = MLP(model.state_feature_dim, 1, **vnet_config)
    vnet = vnet.to(device)

    logger = get_default_logger(output_folder)

    model_optim = torch.optim.Adam(model.parameters(), lr=config["model_lr"])
    model_scaler = torch.cuda.amp.GradScaler(enabled=config["use_fp16"])
    policy_optim = torch.optim.Adam(policy.parameters(), lr=config["policy_lr"])
    policy_scaler = torch.cuda.amp.GradScaler(enabled=config["use_fp16"])
    vnet_optim = torch.optim.Adam(vnet.parameters(), lr=config["vnet_lr"])
    vnet_scaler = torch.cuda.amp.GradScaler(enabled=config["use_fp16"])

    if config["pretraining_iterations"] > 0:
        log.info(
            f'pretrain the model for {config["pretraining_iterations"]} iterations.'
        )
        loader = get_sample_loader(
            dataset,
            config["batch_size"],
            config["pretraining_iterations"],
            num_workers=4,
        )
        for data in tqdm(iter(loader)):
            data = data.to(device)
            with torch.autocast(
                device_type=device, dtype=torch.float16, enabled=config["use_fp16"]
            ):
                _, state_seq, loss, metrics = model(data, data["pre_action"])

            model_optim.zero_grad(set_to_none=True)
            model_scaler.scale(loss).backward()
            model_scaler.unscale_(model_optim)
            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
                model.parameters(), config["grad_clip"]
            )
            model_scaler.step(model_optim)
            model_scaler.update()

    for e in range(config["epoch"]):
        log.info(f"Starting epcoh {e}")

        loader = get_sample_loader(
            dataset, config["batch_size"], config["batch_per_epoch"], num_workers=4
        )

        log.info("Training Model ...")
        train_metric_tracker = AverageMeter()
        training_start_time = time.time()
        for data in tqdm(iter(loader)):
            data = data.to(device)
            with torch.autocast(
                device_type=device, dtype=torch.float16, enabled=config["use_fp16"]
            ):
                _, state_seq, loss, metrics = model(data, data["pre_action"])

            model_optim.zero_grad(set_to_none=True)
            model_scaler.scale(loss).backward()
            model_scaler.unscale_(model_optim)
            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
                model.parameters(), config["grad_clip"]
            )
            model_scaler.step(model_optim)
            model_scaler.update()
            metrics["model_grad_norm"] = grad_norm.item()

            # rollout for longer
            states = model.flatten_states(state_seq)
            states.vmap_(lambda v: v.detach())
            with torch.autocast(
                device_type=device, dtype=torch.float16, enabled=config["use_fp16"]
            ):
                state_seq, _, outputs = model.rollout_with_policy(
                    states,
                    policy,
                    config["imagine_horizon"],
                    names=[],
                    state_detach=True,
                    action_sample=True,
                )

                state_features = torch.stack(
                    [model.get_state_feature(state) for state in state_seq]
                )
                reward = outputs["intrinsic_reward"]
                value = vnet(state_features) / (1 - config["gamma"])

                discount = config["gamma"] * torch.ones_like(reward)
                returns = lambda_return(
                    reward[:-1], value[:-1], discount[:-1], value[-1], config["lambda"]
                )
                discount = torch.cumprod(discount, dim=0)

                policy_loss = -torch.mean(discount[:-1] * returns)
                metrics["policy_loss"] = policy_loss.item()
                policy_entropy_loss = config["policy_entropy_scale"] * torch.mean(
                    outputs["action_entropy"].sum(dim=-1)
                )
                metrics["policy_entropy_loss"] = policy_entropy_loss.item()
                policy_loss = policy_loss + policy_entropy_loss

            policy_optim.zero_grad(set_to_none=True)
            policy_scaler.scale(policy_loss).backward()
            policy_scaler.unscale_(policy_optim)
            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
                policy.parameters(), config["grad_clip"]
            )
            policy_scaler.step(policy_optim)
            policy_scaler.update()
            metrics["policy_grad_norm"] = grad_norm.item()

            with torch.autocast(
                device_type=device, dtype=torch.float16, enabled=config["use_fp16"]
            ):
                value = vnet(state_features[:-1].detach())
                value_loss = 0.5 * torch.mean(
                    (returns.detach() * (1 - config["gamma"]) - value) ** 2
                )
                metrics["value_loss"] = value_loss.item()

            vnet_optim.zero_grad(set_to_none=True)
            vnet_scaler.scale(value_loss).backward()
            vnet_scaler.unscale_(vnet_optim)
            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
                vnet.parameters(), config["grad_clip"]
            )
            vnet_scaler.step(vnet_optim)
            vnet_scaler.update()
            metrics["vnet_grad_norm"] = grad_norm.item()

            train_metric_tracker.add(metrics)

        metrics = train_metric_tracker.get()
        log.info(f"Training last for {time.time() - training_start_time:.3f} s")

        log.info("Collecting new data ...")
        with torch.no_grad():
            actor = PolicyActor(model, policy, eval=True)
            actor = GuassianNoiseActorWrapper(
                actor, config["epsilon"], env.action_space
            )
            reward = interact_with_environment(env, actor, image_sensors)
            metrics["train_reward"] = reward

        dataset.update()

        if e % 10 == 0:
            log.info("Evaluating the model ...")
            with torch.no_grad():
                actor = PolicyActor(model, policy)
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

            if len(used_image_sensors) > 0 or test_env.set_state_from_obs_support:
                log.info("Generating prediction videos ...")
                metrics.update(
                    generate_prediction_videos(
                        model, data, test_env, image_sensors, used_image_sensors, 10, 6
                    )
                )

            log.info("Saving the models ...")
            torch.save(model.state_dict(), os.path.join(output_folder, "model.pt"))
            torch.save(policy.state_dict(), os.path.join(output_folder, "policy.pt"))
            torch.save(vnet.state_dict(), os.path.join(output_folder, "vnet.pt"))

        metrics = {"train/" + k: v for k, v in metrics.items()}
        logger(metrics, e)


if __name__ == "__main__":
    main()
