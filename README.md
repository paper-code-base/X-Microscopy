# X-Microscopy

**Protocol for performing STORM-like superresolution image reconstruction from conventional microscopy with deep learning-based X-Microscopy**

## Overview

 This protocol provides comprehensive instructions on model implementation for performing STORM-like superresolution image reconstrcuction from wide-field images (WFs) of biological structures  with X-Microscopy. The execution of X-Microscopy can be divided into UR-Net-8 training, UR-Net-8 testing, X-Net training, X-Net testing, and performance evaluation of X-Microscopy.

## Start here

### Hardware and software requirements

The following protocol is configured for Ubuntu 16.04 with Python 3.6.4 and TensorFlow 1.13.1. The experiments in our study were carried out on a system equipped with 64GB of RAM and a single 11GB NVIDIA GeForce GTX 1080 GPU.

### Environment configuration and dependency installation

a. Set up the required Python dependencies with this link: https://www.anaconda.com/download

b. Installing X-Microscopy pipeline.

```
cd [your_project_folder_name]
git clone https://github.com/paper-code-base/X-Microscopy
cd X-Microscopy
```

c. Setting up and activating a Python 3.6 virtual environment with Conda.

```
conda create --name X-Microscopy python=3.6
conda activate X-Microscopy
```

d. Install all Python libraries required by the project.

```
cd X-Microscopy
pip install -r requirements.txt 
```

### UR-Net-8 training

a. Open the terminal and navigate to the 'X-Microscopy/UR-Net-8' directory.

```
cd X-Microscopy
cd UR-Net-8
```

b. Ensure the training and test datasets are named the same as the example data we provided.

```
cd X-Microscopy # Ensure you are in the "X-Microscopy" directory; if you are already in this directory, skip this step.
cd UR-Net-8/dataset
tree -d -L 3
training-example-data
└── 1
    ├── F-SRM
    ├── U-SRM
    ├── W-SRM
    └── wf
└── 2
    ├── F-SRM
    ├── U-SRM
    ├── W-SRM
    └── wf
...
└── n
    ├── F-SRM
    ├── U-SRM
    ├── W-SRM
    └── wf
test-example-data
└── 1
    ├── F-SRM
    └── wf
└── 2
    ├── F-SRM
    └── wf
...
└── n
    ├── F-SRM
    └── wf
```

c. Configure all arguments and hyperparameters for training the UR-Net-8 model in ‘main.py’.

```
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='dataset/training-example-data', help='name of the dataset')
parser.add_argument('--test_dataset_name', dest='test_dataset_name', default='dataset/test-example-data', help='name of the test dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=1000, help='the epoch')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')#default=512,1024,256
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='images used to train')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--scale', dest='scale', default=1.0, help='1.0 for most models, 3.0 or laminB1')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./UR-Net-8-weights/', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./experiment_model/base_val/', help='sample are saved here')
parser.add_argument('--abf_dir', dest='abf_dir', default='datasets', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./experiment_model/datasets/test_train/save/', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=5000, help='weight on L1 term in objective')
parser.add_argument('--withbn', dest='withbn', type=bool, default=True, help='model with or without bn, if withbn=True, the batch_size must be 1, else the batch_size can be set larger than 1, only True is supported now')#True
parser.add_argument('--same_input_size', dest='same_input_size', type=bool, default=True, help='mode input with same or different size')
parser.add_argument('--Model', dest='Model', default='my', help='which mode to choose, only my is supported now')
parser.add_argument('--log_dir', default='experiment_model/runs/logs_generated/',help='Directory for saving checkpoint models')
args = parser.parse_args()
```

d. Start the training process by running ‘main.py’’.

```
cd X-Microscopy
cd UR-Net-8
python3 main.py
```

### UR-Net-8 finetuning

a.  Adjust specific parameters for fine-tuning the UR-Net-8 model in ‘main.py’.

```
parser.add_argument('--sample_dir', dest='sample_dir', default='./experiment_model/base_val_finetune/', help='sample are saved here')
parser.add_argument('--same_input_size', dest='same_input_size', type=bool, default=False, help='mode input with same or different size')
```

b. Perform the fine-tuning by running ‘main.py’ .

```
python3 main.py
```

### UR-Net-8 testing

a. Modify the following parameters in ‘main.py’ for testing.

```
parser.add_argument('--phase', dest='phase', default='test', help='train, test')
parser.add_argument('--same_input_size', dest='same_input_size', type=bool, default=True, help='mode input with same or different size')
```

b. Initiate the testing process by executing ‘main.py’.

```
python3 main.py
```

### X-Net training

a. Open the terminal and navigate to the "X-Microscopy/X-Net" directory.

```
cd X-Microscopy
cd X-Net
```

b. Ensure that the dataset is stored in the training-example-data and test-example-data folders within 'X-Microscopy/X-Net/dataset.

```
cd X-Microscopy # Ensure you are in the "X-Microscopy" directory; if you are already in this directory, skip this step.
cd X-Net/dataset
tree -d -L 3
training-example-data
└── 1
    ├── F-SRM
    ├── U-SRM
    ├── W-SRM
    └── wf
└── 2
    ├── F-SRM
    ├── U-SRM
    ├── W-SRM
    └── wf
...
└── n
    ├── F-SRM
    ├── U-SRM
    ├── W-SRM
    └── wf
test-example-data
└── 1
    ├── F-SRM
    └── wf
└── 2
    ├── F-SRM
    └── wf
...
└── n
    ├── F-SRM
    └── wf
```

c. Configure all arguments and hyperparameters for training the X-Net model in ‘main.py’.

```
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='dataset/training-example-data', help='name of the dataset')
parser.add_argument('--test_dataset_name', dest='test_dataset_name', default='dataset/test-example-data', help='name of the test dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=1000,help='of epoch')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='images used to train')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='of output image channels')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./X-Net-weights/', help='models are saved here')
parser.add_argument('--fine_checkpoint_dir', dest='checkpoint_dir', default='./experiment_model/fine_checkpoint/', help='models are saved here')
parser.add_argument('--best_checkpoint_dir', dest='best_checkpoint_dir', default='./experiment_model/check_best_new/', help='best models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./experiment_model/base_val/', help='sample are saved here')
parser.add_argument('--abf_dir', dest='abf_dir', default='./base_super/sample_1129/', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./experiment_model/test_0219/', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=1000, help='weight on L1 term in objective')
parser.add_argument('--withbn', dest='withbn', type=bool, default=True, help='model with or without fusion')
parser.add_argument('--same_input_size', dest='same_input_size', type=bool, default=True, help='mode input with same or different size')
parser.add_argument('--Model', dest='Model', default='my', help='which mode to choose, my or other')
parser.add_argument('--log_dir', default='experiment_model/runs/logs_xl_sparse/',help='Directory for saving checkpoint models')
args = parser.parse_args()
```

d. Start the training process by running ‘main.py’.

```
cd X-Microscopy
cd X-Net
python3 main.py
```

### X-Net finetuning

a. Set up all parameters and hyperparameters for fine-tuning the X-Net model in ‘main.py’.

```
parser.add_argument('--sample_dir', dest='sample_dir', default='./experiment_model/base_val_finetune/', help='sample are saved here')
parser.add_argument('--same_input_size', dest='same_input_size', type=bool, default=False, help='mode input with same or different size')
```

b. Start the training process by running ‘main.py’.

```
python3 main.py
```

### X-Net testing

a. Modify the following parameters of X-Net in ‘main.py’.

```
parser.add_argument('--phase', dest='phase', default='test', help='train, test')
parser.add_argument('--same_input_size', dest='same_input_size', type=bool, default=True, help='mode input with same or different size')
```

b. execute the testing code.

```
python3 main.py
```

## Citation

If you use this method or this code in your research, please cite as:

    @inproceedings{Xulei-2024,
    title={Protocol for performing STORM-like superresolution image reconstruction from conventional microscopy with deep learning-based X-Microscopy},
    author={Lei Xu, Yuancheng Xu, Xiying Yu, Wei Jiang},
    booktitle={},
    pages={},
    year={2024}
    }

### Acknowledgments

This code is written based on the tensorflow framework of pix2pix. 

### License

This code is released for academic research / non-commercial use only. This project is covered under the MIT License.
