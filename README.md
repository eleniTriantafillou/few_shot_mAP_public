# Few-Shot Learning Through an Information Retrieval Lens
This repository contains the code for the paper "Few-Shot Learning Through an Information Retrieval Lens". Eleni Triantafillou, Richard Zemel, Raquel Urtasun [arXiv preprint](https://arxiv.org/abs/1707.02610).

### Prerequisites
- Python 2
- tensorflow version 0.12
- NumPy
- tqdm
- cuda (if you want to run on GPU)


### Setting up
The code assumes the existence of a directory named `data` within the `few_shot_mAP_public` directory. `data` contains 3 subdirectories: `omniglot` and `mini_imagenet`, containing the corresponding datasets, and another directory called `dataset_splits` also containing subdirectories for `omniglot` and `mini_imagenet` containing the data splits for these datasets (.csv files indicating which classes are meant to be used for training / validation / testing).

This structure will be created by running the provided setup script. Please modify the first 4 lines of this script to add the paths to the Omniglot and mini-ImageNet datasets and their corresponding splits (the datasets and splits are not provided in this repository).
```
cd few_shot_mAP_public
./setup.sh
```

If you'd like to monitor the training progress via [Deep Dashboard](https://github.com/renmengye/deep-dashboard), please follow these instructions:
- Setup Deep Dashboard as detailed here https://github.com/renmengye/deep-dashboard
- In `few_shot_mAP_public/src/configs` there are a number of files, one for each example experiment (corresponding to some choice of dataset and model). Please modify the following line that is found on the top of each config file in order to point to the directory where Deep Dashboard should store its results.
```
DASHBOARD_LOC = "/u/eleni/public_html/results/few_shot_mAP/"
```


## Reproducing our results

The experiements in the paper can be reproduced by running 
```
python run_train.py
```

with the appropriate tf.FLAGS set to point to the correct dataset and model. A config file will then be looked up (among the files in `few_shot_mAP_public/src/configs`) based on these two pieces of information and the settings in that file will be used for training.

To evaluate a trained model on the benchmark tasks, you can run
```
python run_eval.py
```
Similarly as before, this requires setting the appropriate dataset and model so that the corresponding config file and model can be looked up.
