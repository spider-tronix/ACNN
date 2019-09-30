import os
import pandas as pd


def write_to_readme(batch_size, lr, seed, epochs, run_time, train_dir):
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
    idx_best_acc = logs['accuracy'].idxmax()
    _, step, loss, acc = logs.iloc[idx_best_acc]

    text = f"""
### Hyperparams
- Torch.seed = {seed}
- Epochs = {epochs}
- Batch_size = {batch_size}
- Learning Rate = {lr}
- Best Accuracy:
    - Step = {step}
    - Train Accuracy = {acc}
    - Train loss = {loss}
    - Training time = {run_time}s ≈ {int(run_time) / 60} min
![Graphs](train.png)
"""
    with open(os.path.join(train_dir, 'README.md'),
              'w', encoding='utf-8') as file:
        file.write(text)
    print('Readme.md file ready')
