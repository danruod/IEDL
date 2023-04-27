# Fisher Information-based Evidential Deep Learning ($\mathcal{I}$-EDL) on MNIST and CIFAR

<!-- [![Project](https://img.shields.io/badge/Project-Website-blue?style=flat-square)](https://correr-zhou.github.io/RepMode/) -->
[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat-square)](https://arxiv.org/pdf/2303.02045.pdf)
[![Project](https://img.shields.io/badge/Code-Github-purple?style=flat-square)](https://github.com/danruod/IEDL)

**Authors**: Danruo Deng, [Guangyong Chen](https://guangyongchen.github.io/), Yu Yang, Furui Liu, [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/1.html)

**Affiliations**: CUHK, Zhejiang Lab

> This repository contains the core experiments on MNIST and CIFAR10 in our paper [Uncertainty Estimation by Fisher Information-based Evidential Deep Learning](https://arxiv.org/pdf/2303.02045.pdf). This is one of the two code repositories of our paper, and is a sub-module of the the [main repository](https://github.com/danruod/IEDL).  


# Quick Q&A Rounds

1. **Question**: Give me a quick-starter code to start reproducing the paper trainings on a GPU?
    ```
    git clone --recursive https://github.com/danruod/IEDL.git
    conda env create -f environment.yml
    conda activate IEDL
    cd IEDL-main/code_classical
    bash ./main.sh
    ```
---------
2. **Question**: Give me a simple python command to run?
   ```
   python main.py --configid "1_mnist/mnist-iedl" --suffix test
   ```
    * run the configuration specifed at [`./configs/1_mnist/mnist-iedl.json`](./configs/1_mini/5w-iedl.json), and
    * store the generated outputs periodically at `./results/1_mnist_test/mnist-iedl.csv`.


# Step-by-Step Guide to the Code
   
+  <details open>
   <summary><strong>Step 1: Cloning the Repo</strong></summary>

   +  <details open>
      <summary><strong>[Option 1] Cloning All Two Repositories of Our Paper</strong></summary>
 
      ```
      git clone --recursive https://github.com/danruod/IEDL.git
      cd IEDL-main/code_classical
      ```
      </details>
 
   +  <details>
      <summary><strong>[Option 2] Cloning This Repository Alone</strong></summary>
 
      ```
      git clone https://github.com/ehsansaleh/code_classical.git`
      cd code_classical
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
   <summary><strong>Step 3: Training and Evaluation</strong></summary>
   
      * To fire up some training yourself, run
        ```
        python main.py --configid "1_mnist/mnist-iedl" --suffix test
        ```
      * This command will read the `./configs/1_mnist/mnist-iedl.json` config as input.
      * if you set `store_results = True`, the computed statistics (including accuracy, uncertainty related metrics) will be saved at  `./results/1_mnist_test/mnist-iedl.csv`.
      * To run multiple trainings, run [`./main.sh`](./main.sh).
   
   </details>

</details>

# Pre-trained Models
You can find pre-trained models in the folder `./saved_models`. Models in `./saved_models/mnist_iedl` and `./saved_models/mnist_edl` are trained on classic MNIST,  Models in `./saved_models/cifar10_iedl` and `./saved_models/cifar10_edl` are trained on classic MNIST and CIFAR10 splits.

# Acknowledgement
Our code is built upon the repository of [Posterior Network](https://github.com/sharpenb/Posterior-Network). We would like to thank its authors for their excellent work. If you want to use and redistribe our code, please follow [this license](https://github.com/danruod/IEDL/blob/main/LICENSE) as well.

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
