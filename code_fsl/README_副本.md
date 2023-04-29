# Fisher Information-based Evidential Deep Learning ($\mathcal{I}$-EDL) with FSL

<!-- [![Project](https://img.shields.io/badge/Project-Website-blue?style=flat-square)](https://correr-zhou.github.io/RepMode/) -->
[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat-square)](https://arxiv.org/pdf/2303.02045.pdf)
[![Project](https://img.shields.io/badge/Code-Github-purple?style=flat-square)](https://github.com/danruod/IEDL)

**Authors**: Danruo Deng, [Guangyong Chen](https://guangyongchen.github.io/), Yu Yang, Furui Liu, [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/1.html)

**Affiliations**: CUHK, Zhejiang Lab

> This repository contains the core experiments on MNIST and CIFAR10 in our paper [Uncertainty Estimation by Fisher Information-based Evidential Deep Learning](https://arxiv.org/pdf/2303.02045.pdf). This is one of the two code repositories of our paper, and is a sub-module of the the [main repository](https://github.com/danruod/IEDL).  


## Evaluations in few-shot experiment
 

  * Classification accuracy (Acc.), AUPR scores for both confidence evaluation (Conf. (Max.$\alpha$)) and OOD detection (OOD ($\alpha_0$)) under 5-way {1, 5, 20}-shot settings of mini-ImageNet. CUB is used for OOD detection. Each experiment is run for over 10,000 few-shot episodes.

  <div align="center">
  <table><thead><tr><th rowspan="2">Method</th><th colspan="3">5-Way 1-Shot</th><th colspan="3">5-Way 5-Shot </th><th colspan="3">5-Way 20-Shot </th></tr><tr><th>Acc.</th><th>Conf.</th><th>OOD</th><th>Acc.</th><th>Conf.</th><th>OOD</th><th>Acc.</th><th>Conf.</th><th>OOD</th></tr></thead>
  <tbody>
  <!-- <tr><td>MC Dropout</td><td>55.83±0.20</td><td>75.30±0.24</td><td>63.45±0.20</td><td>80.61±0.14</td><td>94.47±0.08</td><td>74.95±0.18</td><td>88.17±0.09</td><td>97.38±0.04</td><td>73.04±0.22</tr>
  <tr><td>Radial BNN</td><td>61.71±0.21</td><td>81.03±0.23</td><td>70.16±0.23</td><td>82.28±0.14</td><td>95.21±0.09</td><td>76.18±0.18</td><td>88.12±0.14</td><td>97.22±0.12</td><td>74.65±0.21</tr><
  tr><td>Deep Ensembles</td><td>60.41±0.20</td><td>80.27±0.22</td><td>71.32±0.22</td><td>82.14±0.13</td><td>95.36±0.07</td><td>79.03±0.18</td><td>88.73±0.09</td><td>97.84±0.04</td><td>77.28±0.21</tr>
  <tr><td>Focal Loss</td><td>59.80±0.21</td><td>79.94±0.23</td><td>71.80±0.25</td><td>81.34±0.14</td><td>94.82±0.08</td><td>78.32±0.18</td><td>88.71±0.09</td><td>97.96±0.03</td><td>79.20±0.17</tr> -->
  <td>KL-PN</td><td>63.00±0.21</td><td>80.72±0.22</td><td>74.48±0.24</td><td>80.72±0.14</td><td>93.46±0.09</td><td>77.82±0.21</td><td>87.20±0.10</td><td>96.69±0.04</td><td>76.72±0.20</tr>
  <tr><td>RKL-PN</td><td>62.65±0.22</td><td>80.28±0.24</td><td>74.40±0.24</td><td>80.22±0.24</td><td>92.68±0.25</td><td>80.93±0.20</td><td>85.62±0.28</td><td>94.83±0.30</td><td>85.25±0.20</tr>
  <tr><td>EDL</td><td>61.00±0.22</td><td>80.59±0.23</td><td>65.40±0.26</td><td>80.38±0.15</td><td>93.92±0.09</td><td>76.53±0.27</td><td>85.54±0.12</td><td>97.51±0.04</td><td>79.78±0.23</tr>
  <tr><td>IEDL(Ours)</td><td>63.82±0.20</td><td>82.00±0.21</td><td>74.76±0.25</td><td>82.00±0.14</td><td>94.09±0.09</td><td>82.48±0.20</td><td>88.12±0.09</td><td>97.54±0.04</td><td>85.40±0.19</tr></tbody></table>
  </div>


# Quick Q&A Rounds

1. **Question**: Give me a quick-starter code to start reproducing the paper trainings on a GPU?
    ```
    git clone --recursive https://github.com/danruod/IEDL.git
    conda env create -f environment.yml
    conda activate IEDL
    cd IEDL-main/code_fsl
    bash ./features/download.sh
    bash ./main.sh
    ```
---------
2. **Question**: Give me a simple python command to run?
   ```
   python main.py --configid "1_mini/5w-iedl" --suffix 'test'
   ```
    
    * run the configuration specifed at [`./configs/1_mini/5w-iedl.json`](./configs/1_mini/5w-iedl.json), and
    * store the generated outputs periodically at `./results/1_mini_test/5w-iedl.csv`.


# Step-by-Step Guide to the Code
   
+  <details open>
   <summary><strong>Step 1: Cloning the Repo</strong></summary>

   +  <details open>
      <summary><strong>[Option 1] Cloning All Two Repositories of Our Paper</strong></summary>
 
      ```
      git clone --recursive https://github.com/danruod/IEDL.git
      cd IEDL-main/code_fsl
      ```
      </details>
 
   +  <details>
      <summary><strong>[Option 2] Cloning This Repository Alone</strong></summary>
 
      ```
      git clone https://github.com/ehsansaleh/code_firth.git`
      cd code_fsl
      ```
      </details>

   </details>

+  <details open>
   <summary><strong>Step 2: Create a Environment</strong></summary>
   
   > All two repositories of our paper use the same environment and you only need to install it once. 
   1. Create a Conda environment for the code:
      ```
      conda env create -f environment.yml
      ```
   2. Activate the environment:
      ```
      conda activate IEDL
      ```
   3. Following [this guide](https://docs.wandb.ai/quickstart#set-up-wb) to set up [W&B](https://wandb.ai/) (a tool for experimental logging). 
   
   </details>

+  <details open>
   <summary><strong>Step 3: Download the Features</strong></summary>

   1. To use our pre-computed features, run `bash ./features/download.sh`
   2. **[Optional]** If you like to download the base class features, run `bash ./features/download.sh base`

   </details>
   
+  <details open>
   <summary><strong>Step 4: Training Few-shot Classifiers</strong></summary>
   
      * To fire up some training yourself, run
        ```
        python main.py --configid "1_mini/5w-iedl" --suffix 'test'
        ```
      * This command will read the `./configs/1_mini/5w-iedl.json` config as input.
      * if you set `store_results = True`, the computed statistics (including accuracy, uncertainty related metrics) will be saved at  `./results/1_mini_test/5w-iedl.csv`.
      * To run multiple trainings, run [`./main.sh`](./main.sh).
   
   </details>

+  <details open>
   <summary><strong>Step 5: Summarizing the Results</strong></summary>

      * Set the `csvdir_expname` of `./utils/cfg.py` to specify which folders under `./results` to summarize.
      * To summerize the files under these folders, run 
        ```
        python utils/csv2summ.py
        ```
      * The output will be saved at `./summary/test.csv` for `split_list=novel` or `./summary/val.csv` for `split_list=val`.

   </details>


# Configurations and Arguments

+ <details>
  <summary><strong>Example</strong></summary>

  Here is an example for [`./configs/1_mini/5w-iedl.json`](./configs/1_mini/5w-iedl.json):
  ```json
  {
  "rng_seed": 0,
  "n_tasks": 10000,
  "source_dataset": "miniImagenet",
  "target_dataset": "miniImagenet",
  "ood_dataset": "CUB",
  "backbone_arch": "WideResNet28_10",
  "backbone_method": "S2M2_R",
  "n_shots_list": [20, 5, 1],
  "n_ways_list": [5],
  "split_list": ["novel"],
  "model_type": "evnet",
  "loss_type": "IEDL",
  "act_type": "softplus",
  "fisher_coeff_list": [0.0],
  "lbfgs_iters": 100,
  "store_results": true,
  "dump_period": 10000,
  "use_wandb": true,
  "print_freq": 50,
  "torch_threads": null
  }
  ```
  
  * Note that our code runs the cartesian product of all arguments ending with `_list`. 
    * For instance, there is `2=1*1*1*1*2` different settings to try in the above config file.
    * Each of these settings runs 10,000 tasks, creating a total of 20,000 tasks to perform for this file.
  </details>
  
+ <details>
  <summary><strong>Brief Argument Descriptions</strong></summary>
  
  * `"rng_seed"` determine the random seed to generate the set of 10,000 few-shot tasks.
  * `"n_tasks"` determines the number of few-shot tasks for evaluation of the approach.
  * `"source_dataset"` is the source dataset in few-shot experiments.
    * The features are extracted by a backbone network trained on the base split of the source dataset. 
    * The source dataset should be one of the `"miniImagenet"` or `"tieredImagenet"` options.
  * `"targe_dataset"` is the targe in-distribution (id) dataset in few-shot experiments.
    * This is the dataset from which the id evaluation images and classes (novel or validation) are chosen.
    * The features used are extracted by the backbone trained on the base class of the source dataset. 
    * The target dataset should be one of the `"miniImagenet"` or `"tieredImagenet"` options.
    * We set the source and target datasets to be the same.
  * `"ood_dataset"` is the out-of-distribution (ood) dataset in few-shot experiments.
    * This is the dataset from which the ood evaluation images and classes are chosen.
    * The features used are extracted by the backbone trained on the base class of the source dataset. 
    * The ood dataset only support `"CUB"` options.
  * `"backbone_arch"` specifies the feature backbone architucture to use.
    * We only used the `WideResNet28_10` model in our experiments.
  * `"backbone_method"` specifies the feature backbone training algorithm to evaluate.
    * We only used feature backbones trained with the `S2M2_R` method in our experiments.
  * `"n_shots_list"` specifies a list of number of shots to test.
  * `"n_ways_list"` specifies a list of number of classes to perform few-shot classification tasks over.
  * `"split_list"` is a list of data splits to go through:
    * It should be a subset of `["base", "val", "novel"]`.
  * `"fisher_coeff_list"` specifies a list of coefficients of log determinant of fisher information matrix to iterate over. 
  * `"lbfgs_iters"` specifies the number of L-BFGS iterations to train the few-shot classifier.
  * `"store_results"` should mostly be set to true, so that the python script writes its results in a `./results/{configid}/*.csv` file.
  * `"dump_period"` specifies the number of CSV lines that need to be buffered before flushing them to the disk. This was set to a large value to prevent frequent disk dumps and causing system call over-heads.
  * `"use_wandb"` should mostly be set to true, so that you can observe the experimental effect curve in wandb.
  * `"print_freq"` specifies the number of tasks that need to be buffered before flushing them to the wandb. 
  * `"torch_threads"` sets the number of torch threads.
    * This is just in case you wanted to train the classifiers on a CPU device. 
    * The code was optimized to require minimal CPU usage if a GPU was provided.
    * Therefore, you can safely set this to a small number when using a GPU.
    * You can set this option to `null` to keep the default value PyTorch sets.
  </details>

</details>

# Acknowledgement
Our code is built upon the repository of [Firth Bias Reduction in Few-shot Distribution Calibration](https://github.com/ehsansaleh/code_dcf). We would like to thank its authors for their excellent work. If you want to use and redistribe our code, please follow [this license](https://github.com/danruod/IEDL/blob/main/LICENSE) as well.

# Citation
If you find that our work is helpful in your research, please consider citing our paper:
```latex
@article{deng2023uncertainty,
  title={Uncertainty Estimation by Fisher Information-based Evidential Deep Learning},
  author={Deng, Danruo and Chen, Guangyong and Yu, Yang and Liu, Furui and Heng, Pheng-Ann},
  journal={arXiv preprint arXiv:2303.02045},
  year={2023}
}
```

# Contact
Feel free to contact me (Danruo DENG: [drdeng@link.cuhk.edu.hk](mailto:drdeng@link.cuhk.edu.hk)) if anything is unclear or you are interested in potential collaboration.
