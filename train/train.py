import argparse
import torch
from torch.utils.data import DataLoader
from utils.load_data import RatingsDataset
from train.trainer import Trainer  # Import the Trainer class from your Trainer file
from utils.recommender import Recommender  # Import the Recommender class

def main(args):
    # Load dataset
    dataset = RatingsDataset(datafile=args.datafile)

    # Split into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = dataset.train_test_val_split(train_ratio=0.6, val_ratio=0.2)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Define parameters
    user_number = dataset.get_user_number()
    movie_number = dataset.get_movie_number()
    epochs = args.epochs
    model_name = args.model_name
    trainer = Trainer(args.model_name, user_number, movie_number, args.epochs)
    trainer.run()
    trained_model = trainer.model

    recommender = Recommender(trained_model)
    recommendations = recommender.recommend_k_items(test_dataset)

    for user_id, rec in recommendations.items():
        recommended_movies = rec["movies"]
        recommendation_scores = rec["scores"]

        print(f"User {user_id}:")
        for movie, score in zip(recommended_movies, recommendation_scores):
            print(f"    Movie: {movie}, Score: {score}")
 


if __name__ == "__main__":
    # Argument parser for dataset file and training parameters
    parser = argparse.ArgumentParser(description="Train a recommendation model.")
    parser.add_argument("--datafile", type=str, required=True, help="Path to the dataset file (CSV).")
    parser.add_argument("--model_name", type=str, default="model1", choices=["model1", "model2","model3"],
                        help="Model to train: 'model1' for NCF, 'model2' for NGCF,model3 fro lightgcn.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    args = parser.parse_args()

    main(args)
