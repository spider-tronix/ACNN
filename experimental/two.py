import os
import pandas as pd

def write_to_Readme(batch_size, lr, seed, epochs, train_dir):
    """
    Writes the hyperparams and log stats to Readme.md file
    batch_size: batch_size used
    lr: learning rate
    seed: torch.seed
    epochs: epochs
    train_dir: Directory contianing train.csv and train.png
    returns: Writes a Readme.md file in train_dir
    """
    logs = pd.read_csv(os.path.join(train_dir, 'train.csv'))
    idx_best_acc = logs['train_accuracy'].idxmax()
    _, step, loss, acc = logs.iloc[idx_best_acc]

    Text = f"""
### Hyperparams
- Torch.seed = {seed}
- Epochs = {epochs}
- Batch_size = {batch_size}
- Learning Rate = {lr}
- Best Accuracy:
    - Step = {step}
    - Train Accuracy = {acc}
    - Train loss = {loss}
![Graphs](train.png)
"""
    with open('README.md', 'w') as file:
        file.write(Text)
    print('Readme.md file ready')