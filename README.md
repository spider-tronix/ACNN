# Project Structure
```
├── Archives				# Old Code goes here           
│
├── benchmarks				# Benchmark models
│   └── vanilla_CNN
|   	└── base_model.py   # code
│   └── vanilla_ResNet
|
├── experimental
|
├── models                  # ACNN variants of benchmarks models
│   ├── Vanilla_ACNN.py
│   └── Vanilla_ResNet.py
|
├── results
│   ├── main.py
│   └── MNIST
│   	├── Benchmark Results 	# Results of each Benchmark model	
|		|	
│       └── Vanilla_ACNN
│            └──Training_1
│               ├── Output.txt 		# Detailed training data
│               ├── README.md 		# Hyperparams used
│               ├── Tensorboard_Graphs
│               └── training_loss.png
|		|
│       └── ACNN_ResNet
|	|
|	├──	SVHN
|	└── CIFAR 10
|
├── tests					# modular tests to check implemented methods
│   ├── connect_net.py
│   ├── group_conv_test1.py
│   └── group_conv_time_test.py	
|
└── utilities				# Helper classes and functions
    ├── connect_net.py
    ├── data.py
    ├── train_helpers.py
    └── visuals.py
```