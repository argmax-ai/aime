# Model Card

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

The pretrained world model is released here as the results of our paper "Action Inference by Maximising Evidence: Zero-Shot Imitation from Observation with World Models" at NeurIPS 2023. 

We release the models for the community to replicate the results in our paper and encourage people to study the transferablity of the pretrained world models.
The model can be accessed at [here](https://github.com/argmax-ai/aime/releases/latest).

- **Developed by:** [Xingyuan Zhang](https://icaruswizard.github.io/) during his Ph.D. at Machine Learning Research Lab at Volkswagen AG.
- **Model type:** Latent variable world model, RSSM.
- **License:** _© 2023. This work is licensed under a [_CC BY 4.0 license_](https://creativecommons.org/licenses/by/4.0/)_.

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** https://github.com/vw-argmax-ai/opensource-aime
- **Paper:** [Xingyuan Zhang, Philip Becker-Ehmck, Patrick van der Smagt, Maximilian Karl, Action Inference by Maximising Evidence: Zero-Shot Imitation from Observation with World Models, NeurIPS 2023](https://openreview.net/forum?id=WjlCQxpuxU)

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

The model can:

1. estimate the states of the environment from the observations and actions, i.e. $s_{1:t} \sim q(s_{1:t} | o_{1:t}, a_{0:t-1})$.
2. make prediction from certain state to the future states and observations given an action sequence, i.e. $s_{1:t} \sim p(s_{1:t} | s_0, a_{0:t-1})$ and $o_t \sim p(o_t | s_t)$.
3. evaluate the lower bound of the likelihood of the observations sequence given a certain action sequence, i.e. $\log p(o_{1:t} | a_{0:t-1})$.

### Downstream Use

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

There are applications with each of the ability the model provides:

- with 1, the model can be considered to give a pretrained representation for the task you want to apply on.
- with 2, the model can be treated as a virtual environment to replace the real environment for interaction, which is typically useful for model-based reinforcement learning.
- with 3, the model can be used to infer the actions given the observations with the AIME algorithm as proposed in our paper.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

- Although the model can predict the future observations by rollout on the latent states, it should not be considered for high-quality video generation.
- The model shouldn't work well on embodiments other than what it is trained for without finetuning. 

## How to Get Started with the Model

Use the code below to get started with the model:

```python
from aime.utils import load_pretrained_model

model_root = ... 

model = load_pretrained_model(model_root)
```

## Training Details

### Training Data

<!-- This should link to a Data Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

The datasets we use to train the model are also released [here](https://github.com/argmax-ai/aime/releases/latest). For more details about the datasets, please checkout the [data card](../datasets/readme.md).

### Training Procedure 

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

The model is pretrained on each datasets by running the [train_model_only.py](../scripts/train_model_only.py) script. 
The walker models are trained with the RSSM architecture while the cheetah models are trained with the RSSMO architecture, you can find their implementation details at [code](../aime/models/ssm.py).
For example, to get the `walker-mix-visual` model, you can run `python scripts/train_model_only.py env=walker environment_setup=visual embodiment_dataset_name=walker-mix world_model=rssm`

#### Training Hyperparameters

Please checkout the general config at [here](../aime/configs/model-only.yaml) and model configs for [RSSM](../aime/configs/world_model/rssm.yaml) and [RSSMO](../aime/configs/world_model/rssmo.yaml).

## Citation

<!-- If there is a paper or blog post introducing the model, Bibtex information for that should go in this section. -->

If you find the models useful, please cite our paper.

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

## Model Card Authors and Contact

Xingyuan Zhang with wizardicarus@gmail.com.
