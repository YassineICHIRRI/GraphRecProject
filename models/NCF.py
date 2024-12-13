import torch
import torch.nn as nn 
from torch.nn import Module 


class NCF(Module):
    def __init__(self, userNum, itemNum, dim=64, first_layer=128):
        super(NCF, self).__init__()
        self.uEmbd = nn.Embedding(userNum, dim)
        self.iEmbd = nn.Embedding(itemNum, dim)
        self.mf_uEmbd = nn.Embedding(userNum, dim)
        self.mf_iEmbd = nn.Embedding(itemNum, dim)
        self.mlp = nn.Sequential(nn.Linear(dim * 2, first_layer),
                                 nn.Dropout(0.25),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(first_layer, first_layer // 2),
                                 nn.Dropout(0.25),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(first_layer // 2, first_layer // 4),
                                 nn.Dropout(0.25),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(first_layer // 4, first_layer // 4))
        self.neumf = nn.Linear(dim + first_layer // 4, 1)

        # Ajout de num_movies
        self.num_movies = itemNum  # Définit le nombre total de films

    def forward(self, userIdx, itemIdx):
        uembd = self.uEmbd(userIdx)
        iembd = self.iEmbd(itemIdx)
        embd = torch.cat([uembd, iembd], dim=1)

        mlp = self.mlp(embd)
        mf = self.uEmbd(userIdx) * self.iEmbd(itemIdx)

        prediction = self.neumf(torch.cat([mlp, mf], dim=1))
        return prediction.flatten()


    def recommend_k_items(self, test_data, top_k=5, remove_seen=True):
        # Prepare unique users and all items
        unique_users = test_data['user_id'].unique()
        all_items = np.arange(len(np.unique(test_data['item_id'])))

        # Prepare recommendation DataFrame
        recommendations = []

        # Create a set of seen items for each user if remove_seen is True
        if remove_seen:
            seen_items = test_data.groupby('user_id')['item_id'].apply(set).to_dict()

        # Predict for each user
        for user in unique_users:
            # Create all combinations of this user with all items
            user_tensor = torch.full((len(all_items),), user, dtype=torch.long)
            items_tensor = torch.tensor(all_items, dtype=torch.long)

            # Get predictions
            with torch.no_grad():
                predictions = self.forward(user_tensor, items_tensor).cpu().numpy()

            # Filter out seen items if requested
            if remove_seen:
                user_seen_items = seen_items.get(user, set())
                mask = ~np.isin(all_items, list(user_seen_items))
                predictions = predictions[mask]
                candidate_items = all_items[mask]
            else:
                candidate_items = all_items

            # Get top-k items
            top_k_indices = np.argsort(predictions)[-top_k:][::-1]
            top_k_items = candidate_items[top_k_indices]
            top_k_scores = predictions[top_k_indices]

            # Store recommendations
            for item, score in zip(top_k_items, top_k_scores):
                recommendations.append({
                    'user_id': user,
                    'item_id': item,
                    'predicted_rating': score
                })

        # Convert to DataFrame
        return pd.DataFrame(recommendations)
