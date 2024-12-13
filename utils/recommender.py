import torch

class Recommender:
    def __init__(self, model):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def recommend_k_items(self, test_data, top_k=5, remove_seen=True):
        self.model.eval()
        user_recommendations = {}
        all_users = set(sample['user'] for sample in test_data)

        for user_id in all_users:
        # Create tensors for user and all possible movies
            user_tensor = torch.tensor([user_id]).to(self.device)
            movie_tensor = torch.tensor(list(range(self.model.num_movies))).to(self.device)

            with torch.no_grad():
            # Predict scores for all movies
                predictions = self.model(user_tensor.expand_as(movie_tensor), movie_tensor)

            predictions = predictions.cpu().numpy()

        # Get the top K recommended movies and their scores
            top_k_indices = predictions.argsort()[-top_k:][::-1]
            top_k_scores = predictions[top_k_indices]

        # Optionally remove already seen movies
            if remove_seen:
                seen_movies = {sample['movie'] for sample in test_data if sample['user'] == user_id}
                top_k_indices = [movie for movie in top_k_indices if movie not in seen_movies]
                top_k_scores = [score for idx, score in zip(top_k_indices, top_k_scores) if idx not in seen_movies]

        # Store recommendations and scores
            user_recommendations[user_id] = {
            "movies": [int(movie) for movie in top_k_indices],
            "scores": [float(score) for score in top_k_scores],
            }

        return user_recommendations

