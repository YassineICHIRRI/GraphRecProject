import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from models.NCF import NCF
from models.NGCF import NGCF
from utils.load_data import RatingsDataset


class Trainer:
    def __init__(self, model_name, user_number, movie_number, epochs):
        self.model_name = model_name
        self.user_number = user_number
        self.movie_number = movie_number
        self.epochs = epochs
        self.history = {'train_loss': [], 'val_accuracies': [], 'val_precisions': [], 'val_recalls': [], 'val_f1_scores': [], 'val_maps': [], 'val_ndcgs': []}


    def train(self, loader, model, optim, loss):
        model.train()
        ls = 0.0
        all_predictions = []
        all_ratings = []

        for batch in loader:
            optim.zero_grad()
            prediction = model(batch['user'].cuda(), batch['movie'].cuda())
            loss_ = loss(batch['rating'].float().cuda(), prediction)
            loss_.backward()
            optim.step()
            ls += loss_.item()
            all_predictions.append(prediction.cpu().detach().numpy())
            all_ratings.append(batch['rating'].numpy())

        all_predictions = np.concatenate(all_predictions)
        all_ratings = np.concatenate(all_ratings)
        accuracy = np.mean(np.round(all_predictions) == all_ratings)
        precision = self.precision_at_k(all_ratings, 10)  # Example: top 10 items
        recall = self.recall_at_k(all_predictions, 10, len(all_ratings))
        f1 = self.F1(precision, recall)
        map = self.mean_average_precision(all_ratings)
        ndcg = self.ndcg_at_k(all_ratings, 10)

        return ls / len(loader), accuracy, precision, recall, f1, map, ndcg

    def evaluate(self, loader, model, loss):
        model.eval()
        ls = 0.0
        all_predictions = []
        all_ratings = []

        with torch.no_grad():
            for batch in loader:
                prediction = model(batch['user'].cuda(), batch['movie'].cuda())
                loss_ = loss(batch['rating'].float().cuda(), prediction)
                ls += loss_.item()
                all_predictions.append(prediction.cpu().detach().numpy())
                all_ratings.append(batch['rating'].numpy())

        all_predictions = np.concatenate(all_predictions)
        all_ratings = np.concatenate(all_ratings)
        accuracy = np.mean(np.round(all_predictions) == all_ratings)
        precision = self.precision_at_k(all_ratings, 10)
        recall = self.recall_at_k(all_predictions, 10, len(all_ratings))
        f1 = self.F1(precision, recall)
        map = self.mean_average_precision(all_ratings)
        ndcg = self.ndcg_at_k(all_ratings, 10)

        return ls / len(loader), accuracy, precision, recall, f1, map, ndcg

    def run(self):
        # Prepare data
        train_loader = DataLoader(RatingsDataset('data/train.csv'), batch_size=128, shuffle=True)
        val_loader = DataLoader(RatingsDataset('data/val.csv'), batch_size=128, shuffle=False)

        # Choose model
        if self.model_name == "model1":
            model = NCF(self.user_number, self.movie_number).cuda()
        elif self.model_name == "model2":
            model = NGCF(self.user_number, self.movie_number).cuda()
        else:
            raise ValueError("Invalid model name")

        # Define optimizer and loss function
        optim = torch.optim.Adam(model.parameters(), lr=0.001)
        loss = torch.nn.BCELoss()

        # Training loop
        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            train_loss, train_acc, train_prec, train_rec, train_f1, train_map, train_ndcg = self.train(train_loader, model, optim, loss)
            val_loss, val_acc, val_prec, val_rec, val_f1, val_map, val_ndcg = self.evaluate(val_loader, model, loss)
            end_time = time.time()

            # Record metrics
            self.history['train_loss'].append(train_loss)
            self.history['val_accuracies'].append(val_acc)
            self.history['val_precisions'].append(val_prec)
            self.history['val_recalls'].append(val_rec)
            self.history['val_f1_scores'].append(val_f1)
            self.history['val_maps'].append(val_map)
            self.history['val_ndcgs'].append(val_ndcg)

            print(f"Epoch {epoch}/{self.epochs} | Time: {end_time - start_time:.2f}s | "
                  f"Train Loss: {train_loss:.4f} | Val Accuracy: {val_acc:.4f} | Val MAP: {val_map:.4f} | Val NDCG: {val_ndcg:.4f}")

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

if __name__ == '__main__':
    trainer = Trainer(model_name="model1", user_number=6040, movie_number=3952, epochs=10)
    trainer.run()
    trainer.compare_models(trainer.history, trainer.history)
