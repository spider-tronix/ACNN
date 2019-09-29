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
├── results					# Results on standard datasets
│   └── MNIST
│   	├── Benchmark Results    # Benchmarks Results for each benchmark model				
│   	└── Vanilla_ACNN    
|			└── Training 1		# Each Training in a folder
|				├── Tensorboard Graphs     
|		 		├──	Output.txt 			# Detailed training data
|		 		├──	Readme.md 			# Hyperparams used
|				└── Visualizations	   	# Images of both nets
│   	└── Resnet_ACNN
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