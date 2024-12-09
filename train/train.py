# train.py

import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.NCF import NCF
from models.NGCF import NGCF
from utils.load_data import RatingsDataset
from trainer import Trainer  # Import the Trainer class from your Trainer file

def main():
    # Prepare data
    train_loader = DataLoader(RatingsDataset('data/train.csv'), batch_size=128, shuffle=True)
    val_loader = DataLoader(RatingsDataset('data/val.csv'), batch_size=128, shuffle=False)

    # Define parameters
    user_number = 6040
    movie_number = 3952
    epochs = 10
    model_name = "model1"  # Change to "model2" for NGCF

    # Initialize Trainer
    trainer = Trainer(model_name, user_number, movie_number, epochs)

    # Run the training and evaluation
    trainer.run()

if __name__ == "__main__":
    main()
