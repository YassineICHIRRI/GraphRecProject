
import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import numpy as np


class RatingsDataset(Dataset):
    """Ratings Dataset"""

    def __init__(self, datafile = '/Data/movielens100k.csv'):
        """
        Args:
            csv_file (string): Path to the csv file with ratings.
        """
        self.csv = pd.read_csv(datafile)

        self.user_ids = list(self.csv.user_id-1)
        self.movie_ids = list(self.csv.movie_id_ml-1)
        self.ratings = list(self.csv.rating)

        self.userNums = np.max(self.user_ids)+1
        self.movieNums = np.max(self.movie_ids)+1

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        return {
            'user': self.user_ids[idx],
            'movie': self.movie_ids[idx],
            'rating': self.ratings[idx]
        }

    def get_user_number(self):
        return self.userNums

    def get_movie_number(self):
        return self.movieNums
    
    def train_test_val_split(self, train_ratio=0.6, val_ratio=0.2):
            """
            Splits the dataset into train, validation, and test sets.

            Args:
                train_ratio (float): Proportion of data to use for training.
                val_ratio (float): Proportion of data to use for validation.
            
            Returns:
                tuple: (train_dataset, val_dataset, test_dataset)
            """
            test_ratio = 1 - train_ratio - val_ratio
            if test_ratio < 0:
                raise ValueError("Train and validation ratios must sum to less than 1.")

            train_size = int(train_ratio * len(self))
            val_size = int(val_ratio * len(self))
            test_size = len(self) - train_size - val_size

            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                self, [train_size, val_size, test_size]
            )
            return train_dataset, val_dataset, test_dataset

    
    @staticmethod
    def to_csv(input_file, output_file, delimiter='::', header=None):
        """
        Converts a file with a specific delimiter to a CSV file.

        Args:
            input_file (string): Path to the input file.
            output_file (string): Path to the output CSV file.
            delimiter (string): Delimiter used in the input file. Default is '::'.
            header: Header row specification. Default is None (no header).
        """
        try:
            # Read the input file with the specified delimiter
            df = pd.read_csv(input_file, delimiter=delimiter, header=header)
            
            # Save to .csv
            df.to_csv(output_file, index=False)
            print(f"File successfully converted to {output_file}")
        except Exception as e:
            print(f"An error occurred: {e}")