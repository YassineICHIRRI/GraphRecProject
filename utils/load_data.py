from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import numpy as np


class RatingsDataset(Dataset):
    """Ratings Dataset"""

    def __init__(self):
        """
        Args:
            csv_file (string): Path to the csv file with ratings.
        """
        self.csv = pd.read_csv('/Data/movielens100k.csv')

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
    
    def to_csv(input_file,output_file):
        try:
            # Adjust the delimiter and header as needed
            df = pd.read_csv(input_file, delimiter='::', header=None)  # Adjust as necessary
            # Save to .csv
            df.to_csv(output_file, index=False)
            print(f"File successfully converted to {output_file}")
        except Exception as e:
            print(f"An error occurred: {e}")