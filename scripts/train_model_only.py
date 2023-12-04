import logging
import os
import time

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from aime.data import NPZFolder, get_sample_loader
from aime.env import env_classes
from aime.logger import get_default_logger
from aime.models.ssm import ssm_classes
from aime.utils import (
    CONFIG_PATH,
    DATA_PATH,
    MODEL_PATH,
    OUTPUT_PATH,
    AverageMeter,
    eval_prediction,
    generate_prediction_videos,
    get_image_sensors,
    parse_world_model_config,
    setup_seed,
)

log = logging.getLogger("main")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="model-only")
def main(config: DictConfig):
    setup_seed(config["seed"])

    log.info("using the following config:")
    log.info(config)

    log_name = config["log_name"]
    output_folder = os.path.join(OUTPUT_PATH, log_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    OmegaConf.save(config, os.path.join(output_folder, "config.yaml"))
    dataset_folder = os.path.join(DATA_PATH, config["embodiment_dataset_name"])

    env_config = config["env"]
    env_class_name = env_config["class"]
    try:
        env = env_classes[env_class_name](env_config["name"])
    except Exception as e:
        log.info(f"The environment is not instanceable due to {e}.")
        env = None
    dataset = NPZFolder(dataset_folder, config["horizon"], overlap=True)
    data = dataset[0]

    assert env_config["name"].split("-")[0] == config["embodiment_dataset_name"].split("-")[0]

    sensor_layout = env_config["sensors"]
    world_model_config = parse_world_model_config(
        config, sensor_layout, data, predict_reward=config["use_reward"]
    )
    world_model_name = world_model_config.pop("name")
    all_image_sensors, used_image_sensors = get_image_sensors(
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
    model = model.to(device)

    logger = get_default_logger(output_folder)

    model_optim = torch.optim.Adam(model.parameters(), lr=config["model_lr"])
    model_scaler = torch.cuda.amp.GradScaler(enabled=config["use_fp16"])

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
                _, _, loss, metrics = model(data, data["pre_action"])

            model_optim.zero_grad(set_to_none=True)
            model_scaler.scale(loss).backward()
            model_scaler.unscale_(model_optim)
            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
                model.parameters(), config["grad_clip"]
            )
            model_scaler.step(model_optim)
            model_scaler.update()
            metrics["model_grad_norm"] = grad_norm.item()

            train_metric_tracker.add(metrics)

        metrics = train_metric_tracker.get()
        log.info(f"Training last for {time.time() - training_start_time:.3f} s")

        with torch.no_grad():
            if len(used_image_sensors) > 0 or (
                env is not None and env.set_state_from_obs_support
            ):
                log.info("Generating prediction videos ...")
                metrics.update(
                    generate_prediction_videos(
                        model, data, env, all_image_sensors, used_image_sensors, 10, 6
                    )
                )
            metrics.update(eval_prediction(model, data, 10))

        log.info("Saving the models ...")
        torch.save(model.state_dict(), os.path.join(output_folder, "model.pt"))

        metrics = {"train/" + k: v for k, v in metrics.items()}
        logger(metrics, e)


if __name__ == "__main__":
    main()
