# Data Card

## Motivation

> **For what purpose was the dataset created?**

The dataset is created for the experiment section of our paper "Action Inference by Maximising Evidence: Zero-Shot Imitation from Observation with World Models" at NeurIPS 2023 to pretrained world models and do imitation learning. We release the datasets for the community as a common test bench for similar problems.

> **Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?**

The dataset is collected by [Xingyuan Zhang](https://icaruswizard.github.io/) during his Ph.D. at Machine Learning Research Lab at Volkswagen AG.

## Uses

> **Has the dataset been used for any tasks already?**

Yes, the datasets has been used in our AIME paper for pretraining world models and imitation learning from observation.

> **Is there a repository that links to any or all papers or systems that use the dataset?**

No.

> **What (other) tasks could the dataset be used for?**

The datasets can also be used for offline reinforcement learning.

> **Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?**

No. Everything from the simulator is recorded in the dataset. 

> **Are there tasks for which the dataset should not be used?**

Not at the moment.

## Data description

> **What data does each instance consist of? What is the format of it?**

Every dataset consists of certain number of trajectories and each trajecory is stored as a separate `.npz` file.
The `.npz` file can be loaded by `np.load` which give you a dictionary of `np.ndarray`. 
The dictionary has both the proprioceptions and the images for each time step. 
Note: the key `pre_action` means the actions taken by the agent one time step before which leads to the current observation, hence all the `pre_action` in the first time step is 0.

> **Are there recommended data splits (e.g., training, development/validation, testing)?**

Each dataset is self-contained, we don't have a recommended data splits inside of it.

> **Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?**

Yes, the datasets are self-contained.

> **Is there any example code for loading the dataset?**

```python
from aime.data import NPZFolder
from aime.utils import DATA_PATH

dataset_name = 'walker-plan2explore-buffer'
dataset = NPZFolder(dataset_name, horizon=50, overlap=True)
```

## Data Creation

The random datasets are collected by running a random policy which uniformly sample the action space at each time step. For example, you can collect the `walker-random` dataset by `python scripts/collect_random_dataset.py --env walker`.

The buffer datasets are collected by running dreamer or plan2explore algorithm on each tasks with the visual setup for 1000 trajectories and taking the replay buffer. For example, you can collect the `walker-plan2explore-buffer` dataset by `python scripts/train_plan2explore.py env=walker environment_setup=visual`.

The `walker-mix` dataset is a mixup of `walker-random`, `walker-plan2explore-buffer`, `walker-stand-buffer` and `walker-walk-buffer`. You can create the dataset with `python scripts/mix_datasets.py -i walker-random walker-plan2explore-buffer walker-stand-buffer walker-walk-buffer -o walker-mix`.

The expert datasets are collected by the trained dreamer agent for 100 trajectories. For example, you can collect the `walker-run-expert` dataset by `python scripts/generate_expert_datasets.py --model_path <log_path_of_the_dreamer_run_agent> --output_folder datasets/walker-run-expert`.

## Distribution

> **How will the dataset will be distributed (e.g., tarball on website, API, GitHub)?**

The datasets will be hosted with [Github Release](https://github.com/argmax-ai/aime/releases/latest).

> **When will the dataset be distributed?**

November 2023.

> **Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?**

CC BY 4.0.

> **Have any third parties imposed IP-based or other restrictions on the data associated with the instances?**

No.

## Maintenance

> **Who will be supporting/hosting/maintaining the dataset?**

Xingyuan Zhang will maintain this dataset. You can contact him with wizardicarus@gmail.com.

> **Will there be an erratum? If yes, how should people get access to that?**

There won't. 

> **Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)?**

Not planned.

> **Will older versions of the dataset continue to be supported/hosted/maintained?**

As long as github release allows to do so.

> **If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?**

The dataset is free to use, people can build their own work on it and release by themselves. 

## Additional Information

### Version

Version 1.0, the initial release.

### Dataset Curators

The dataset is collected by [Xingyuan Zhang](https://icaruswizard.github.io/) during his Ph.D. at Machine Learning Research Lab at Volkswagen AG.

### Licensing Information

_© 2023. This work is licensed under a [_CC BY 4.0 license_](https://creativecommons.org/licenses/by/4.0/)_.

### Citation Information

If you find the datasets useful, please cite our paper.

```BibTeX
@inproceedings{
zhang2023aime,
title={Action Inference by Maximising Evidence: Zero-Shot Imitation from Observation with World Models},
author={Xingyuan Zhang and Philip Becker-Ehmck and Patrick van der Smagt and Maximilian Karl},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=WjlCQxpuxU}
}
```
