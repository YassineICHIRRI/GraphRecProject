import time
import matplotlib.pyplot as plt 
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.NCF import NCF
from models.NGCF import NGCF
from utils.load_data import RatingsDataset
import torch.nn.functional as F
from utils.metrics import *
from tqdm import tqdm  # Import tqdm for progress bars


class Trainer:
    def __init__(self, model_name, user_number, movie_number, epochs):
        self.model_name = model_name
        self.user_number = user_number
        self.movie_number = movie_number
        self.epochs = epochs
        self.history = {'train_loss': [], 'train_acc': [], 'train_map': [], 'train_ndcg': [], 
                        'val_accuracies': [], 'val_precisions': [], 'val_recalls': [], 'val_f1_scores': [], 
                        'val_maps': [], 'val_ndcgs': []}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, loader, model, optim, loss):
        model.to(self.device)
        model.train()
        ls = 0.0
        all_predictions = []
        all_ratings = []

        for batch in tqdm(loader, desc="Training", unit="batch"):  # Add tqdm for progress bar
            optim.zero_grad()
            prediction = model(batch['user'].to(self.device), batch['movie'].to(self.device))
            loss_value = loss(prediction, batch['rating'].float().to(self.device))
            loss_value.backward()
            optim.step()
            ls += loss_value.item()
            all_predictions.append(prediction.cpu().detach().numpy())
            all_ratings.append(batch['rating'].numpy())

        all_predictions = np.concatenate(all_predictions)
        all_ratings = np.concatenate(all_ratings)
        accuracy = np.mean(np.round(all_predictions) == all_ratings)
        precision = precision_at_k(all_ratings, 10)  # Example: top 10 items
        recall = recall_at_k(all_predictions, 10, len(all_ratings))
        f1 = F1(precision, recall)
        map = mean_average_precision(all_ratings)
        ndcg = ndcg_at_k(all_ratings, 10)

        return ls / len(loader), accuracy, precision, recall, f1, map, ndcg

    def evaluate(self, loader, model, loss):
        model.to(self.device)
        model.eval()
        ls = 0.0
        all_predictions = []
        all_ratings = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating", unit="batch"):  # Add tqdm for progress bar
                prediction = model(batch['user'].to(self.device), batch['movie'].to(self.device))
                loss_value = loss(prediction, batch['rating'].float().to(self.device))
                ls += loss_value.item()
                all_predictions.append(prediction.cpu().detach().numpy())
                all_ratings.append(batch['rating'].numpy())

        all_predictions = np.concatenate(all_predictions)
        all_ratings = np.concatenate(all_ratings)
        accuracy = np.mean(np.round(all_predictions) == all_ratings)
        precision = precision_at_k(all_ratings, 10)
        recall = recall_at_k(all_predictions, 10, len(all_ratings))
        f1 = F1(precision, recall)
        map = mean_average_precision(all_ratings)
        ndcg = ndcg_at_k(all_ratings, 10)

        return ls / len(loader), accuracy, precision, recall, f1, map, ndcg

    def run(self):
        # Load the dataset
        dataset = RatingsDataset(datafile='data/movielens100k.csv')
        rt = dataset.csv
        rt['user_id'] = rt['user_id'] - 1
        rt['movie_id_ml'] = rt['movie_id_ml'] - 1
        
        # Split into train, validation, and test sets
        train_dataset, val_dataset, test_dataset = dataset.train_test_val_split(train_ratio=0.6, val_ratio=0.2)
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

        # Choose model
        if self.model_name == "model1":
            model = NCF(self.user_number, self.movie_number)
        elif self.model_name == "model2":
            model = NGCF(self.user_number, self.movie_number, rt)
        else:
            raise ValueError("Invalid model name")

        # Define optimizer and loss function
        optim = torch.optim.Adam(model.parameters(), lr=0.001)
        loss = torch.nn.L1Loss()  # Changed to L1Loss for regression scenario

        # Training loop
        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            train_loss, train_acc, train_prec, train_rec, train_f1, train_map, train_ndcg = self.train(train_loader, model, optim, loss)
            val_loss, val_acc, val_prec, val_rec, val_f1, val_map, val_ndcg = self.evaluate(val_loader, model, loss)
            end_time = time.time()

            # Record metrics
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_map'].append(train_map)
            self.history['train_ndcg'].append(train_ndcg)
            self.history['val_accuracies'].append(val_acc)
            self.history['val_precisions'].append(val_prec)
            self.history['val_recalls'].append(val_rec)
            self.history['val_f1_scores'].append(val_f1)
            self.history['val_maps'].append(val_map)
            self.history['val_ndcgs'].append(val_ndcg)

            print(f"Epoch {epoch}/{self.epochs} | Time: {end_time - start_time:.2f}s | "
                  f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f} | Val Accuracy: {val_acc:.4f} | Val MAP: {val_map:.4f} | Val NDCG: {val_ndcg:.4f}")

    def compare_models(self, history_model1, history_model2):
        # Compare metrics
        print("Comparison of Metrics:")
        print("Model 1 - Accuracy:", history_model1['val_accuracies'][-1])
        print("Model 2 - Accuracy:", history_model2['val_accuracies'][-1])
        print("Model 1 - Precision:", history_model1['val_precisions'][-1])
        print("Model 2 - Precision:", history_model2['val_precisions'][-1])
        print("Model 1 - Recall:", history_model1['val_recalls'][-1])
        print("Model 2 - Recall:", history_model2['val_recalls'][-1])
        print("Model 1 - F1 Score:", history_model1['val_f1_scores'][-1])
        print("Model 2 - F1 Score:", history_model2['val_f1_scores'][-1])
        print("Model 1 - MAP:", history_model1['val_maps'][-1])
        print("Model 2 - MAP:", history_model2['val_maps'][-1])
        print("Model 1 - NDCG:", history_model1['val_ndcgs'][-1])
        print("Model 2 - NDCG:", history_model2['val_ndcgs'][-1])

        # Plot loss evolution
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history_model1['train_loss'], label='Model 1 Train Loss', color='blue')
        plt.plot(history_model2['train_loss'], label='Model 2 Train Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Evolution')
        plt.legend()

        # Plot validation loss
        plt.subplot(1, 2, 2)
        plt.plot(history_model1['val_accuracies'], label='Model 1 Validation Accuracy', color='blue')
        plt.plot(history_model2['val_accuracies'], label='Model 2 Validation Accuracy', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy Evolution')
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Plot time per epoch
        plt.figure(figsize=(6, 4))
        plt.plot(history_model1['times'], label='Model 1 Time per Epoch', color='blue')
        plt.plot(history_model2['times'], label='Model 2 Time per Epoch', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.title('Time per Epoch Evolution')
        plt.legend()
        plt.show()
