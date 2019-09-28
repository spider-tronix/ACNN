# Project Structure
```
.
├── Archives                    # Old Code goes here           
├── models                      # Training agents
│   ├── ACNN                        # Main models
│   │   └── agents.py               
│   ├── connect_net.py              # Custom Conv Class
│   ├── vanilla_CNN                 # Base benchmark of LeNet Type Conv Model
│   │   └── base_model.py           
│   └── vanilla_ResNet              # Base benchmark of ResNet Type Conv Model
│       ├── resnet10.py             
│       └── resnet10_bare.py
├── results                     # Perfect Codes, Images, Graphs
│   ├── Visualizations              # Learnt features from Basic Conv layer
├── tests                       # Training Loops 
│   ├── connect_net.py              # Custom Conv 
│   ├── vanilla_acnn.py             # Basic ACNN
│   ├── vanilla_cnn.py              # Basic Conv
│   └── vanilla_resnet.py           # ResNet
└── utilities                   # Helper Functions
    ├── data.py                     # Dataloaders and Handlers
    ├── train_helpers.py            # Loops for training and evaluations
    └── visuals.py                  # Data Analysis, Custom Visualizations, Graphs, etc.
```