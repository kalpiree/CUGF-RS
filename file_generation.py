import pandas as pd
import torch
import os

from evaluation import metrics
from main_1 import model, validation_dataloader, device, test_dataloader

# Define dataset and interaction types
datasets = ['amazonoffice', 'epinions', 'foursquare', 'gowalla', 'lastFM', 'movielens']
interactions = ['interactions', 'popularconsumptions']

output_folder = 'output_files'
os.makedirs(output_folder, exist_ok=True)
def append_scores_and_data(model, data_loader, device):
    model.eval()
    data_collection = []

    with torch.no_grad():
        for batch in data_loader:
            # Move batch data to the device and ensure tensors are appropriately formatted
            batch = [x.to(device).squeeze() for x in batch]  # Use squeeze to remove singleton dimensions

            if data_loader.dataset.include_features:
                # When features are included, they are in the specific order as in the dataset's get_item
                users, items, labels, weights, features, ratings, item_types, groups = batch
                features = features.float()
                predictions = model(features)  # Assuming model takes features only
                predictions = predictions.squeeze()
            elif 'WMF' in model.__class__.__name__:
                # When weights are used
                users, items, labels, weights, ratings, item_types, groups = batch
                weights = weights.float()
                predictions = model(users, items, weights)  # Assuming model takes users, items, and weights
            else:
                # When features are not included
                users, items, labels, weights, ratings, item_types, groups = batch
                users, items = users.long(), items.long()
                predictions = model(users, items)  # Assuming model takes users and items only

            # Collect all the required data into a DataFrame on the go
            batch_data = {
                'userId': users.cpu().numpy(),
                'itemId': items.cpu().numpy(),
                'score': predictions.cpu().numpy(),
                'label': labels.cpu().numpy(),
                'weight': weights.cpu().numpy(),
                'rating': ratings.cpu().numpy(),
                'item_type': item_types.cpu().numpy(),
                'group': groups.cpu().numpy()
            }
            data_collection.append(pd.DataFrame(batch_data))

    # Combine all data into a single DataFrame
    full_data_df = pd.concat(data_collection, ignore_index=True)
    return full_data_df

# Loop through datasets and interactions for file generation
for dataset in datasets:
    for interaction in interactions:
        # Construct file names dynamically
        # Generate data and scores
        validation_df = append_scores_and_data(model, validation_dataloader, device)
        test_df = append_scores_and_data(model, test_dataloader, device)

        # Optionally, save to CSV
        validation_df.to_csv('validation_with_scores_int_movlens_GMF_rat_1_100.csv', index=False)  #fourth change
        test_df.to_csv('test_with_scores_int_movlens_GMF_rat_1_100.csv', index=False)    #fifth change

        top_k = 10
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Assuming you're using a device variable

        # Calculate the metrics
        avg_hr, avg_ndcg = metrics(model, test_dataloader, top_k, device)

        # Print the results in a formatted way
        print(f"Average Hit Rate Test Set(HR@{top_k}): {avg_hr:.3f}")
        print(f"Average Normalized Discounted Cumulative Gain Test set (NDCG@{top_k}): {avg_ndcg:.3f}")

        top_k = 10  # Adjust this value based on your model's evaluation requirements
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Assuming you're using a device variable

        # Calculate the metrics
        avg_hr, avg_ndcg = metrics(model, validation_dataloader, top_k, device)

        # Print the results in a formatted way
        print(f"Average Hit Rate Val Set(HR@{top_k}): {avg_hr:.3f}")
        print(f"Average Normalized Discounted Cumulative Gain Val Set (NDCG@{top_k}): {avg_ndcg:.3f}")
