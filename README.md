# Debiasing with Mixed Capacity Ensembles
This repo contains the code for our paper 
"Learning to Model and Ignore Dataset Bias with Mixed Capacity Ensembles"
In particular, it contains code to train various models that are debiased, meaning they are trained to 
avoid using particular strategies that are known to work well on the training data, but do not generalize to
out-of-domain of adversarial settings. Unlike our [prior work](https://github.com/chrisc36/debias),
the methods in this repo do not require prior knowledge of the biases in the dataset.

## Setup
### Dependencies
We require python>=3.7 and torch >= 1.3.1. Additional requirements are are in

`requirements.txt`

To install, install torch and then run:

`pip install -r requirements.txt`

Our implementation of fp16 uses [apex](https://github.com/NVIDIA/apex/), in particular
apex installed with git commit `4a8c4ac088b6f84a10569ee89db3a938b48922b4`.

This is, unfortunately, a pretty clumsy/out-dated way of doing fp16. To run 
apex needs to be checked out from that commit and installed, that is:

```
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 4a8c4ac088b6f84a10569ee89db3a938b48922b4
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Other apex commits might work, but are not tested.

### Data
Scripts will automatically download any data they need. See config.py for the download locations, by default
everything will be downloaded to ./data.
The first time models are run be patient, some of the downloads can take a while.

The exception is ImageNet, which will have to be downloaded manually, after which
 `config.IMAGENET_HOME` should be set to point towards the imagenet directory.
 
 ## Running Experiments
Each task has a corresponding directory in `autobias/experiments`. 
Each directory has a train script. By default, the scripts will train our MCE method, 
but our ablations can be trained as well by setting the `mode` flag. 

VQA is currently missing since I am still porting it, but will be added soon.

The experiments use defaults from the paper, but I would recommend double-checking 
that they are what you want. 

The training experiments use large batches, so most of them require a lot
RAM and can't be trained on single GPUs. I trained these models on 4 GPUs with 12gp of RAM in fp16, and used fp16 the 
mnli and vqa models. 

If the batch size requirement is too steep, the noci or adv baselines can be applied with 
lower batch sizes, and are still pretty effective. 

Each directory also has an evaluation script the will evaluate models on the dev or test data 
and save the results in the training run's directory.

`autobias/experiments/run_another_seed.py` can be used to train additional runs using 
the saved config from an previous run. 

## Examples
Train MCE for MNLI:

`python autobias/experiments/mnli/train_model.py --output_dir path/to/output --fp16`

Train the adversary baseline for MNLI

`python autobias/experiments/mnli/train_model.py --output_dir path/to/output --fp16 --mode adv`

Train the NoCI ablations on MNIST Background with a weight of 0.5 on the lower capacity mode:

`python autobias/experiments/mnist/train_model.py --output_dir path/to/output --dataset backgroud --mode noci --lc_weight 0.5`

Evaluate a MNLI model on the dev st:

`python autobias/experiments/mnli/evaluate_ensemble.py path/to/model`

(Output will be saved in path/to/mode/eval)

## Code
The main implementation of our methods is in autobias/modules/classifier_ensembles.py. The argmin
conditional independence method is implemented in autobias/argmin_modules

## Cite
If you use this work, please cite:

"Learning to Model and Ignore Dataset Bias with Mixed Capacity Ensembles". 
Christopher Clark, Mark Yatskar, Luke Zettlemoyer. In Finds of EMNLP 2020.
