# Project Structure
```
.
├── Archives                    # Old Code goes here 
├── README.md                   # You are here
├── benchmarks                  # Benchmark models
│   ├── vanilla_CNN
│   │   └── base_model.py
│   └── vanilla_ResNet
│       ├── base_resnet.py
│       └── resnet10.py
├── experimental                # Playground for testing new features
├── models                      # ACNN variants of benchmarks models
│   ├── Acnn_ResNet.py
│   ├── Vanilla_Acnn.py
├── results
│   ├── MNIST                   # Ring generated Stats and logs
│   │   └── VanillaACNN
│   │       ├── Training_1
│   │           ├── README.md
│   │           ├── Tensorboard_Summary
│   │           │   └── events.out.tfevents...
│   │           ├── test.csv
│   │           ├── test.png
│   │           ├── train.csv
│   │           └── train.png
│   │       ├── Training_2
│   │       ├── ...
│   │       └── ...
│   └── main.py                 # ONE RING TO RULE THEM ALL
├── tests                       # Modular tests to check implemented methods
│   ├── connect_net.py
│   ├── group_conv_test1.py
│   └── group_conv_time_test.py
└── utilities                   # Helper classes and functions
    ├── connect_net.py
    ├── data.py
    ├── train_helpers.py
    └── visuals.py

```