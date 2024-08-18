# Device configuration
import gc

import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
import os
from evaluation import metrics
from models.BERT4Rec import BERT4Rec
from models.DeepFM import DeepFM
from models.FM import FactorizationMachine
from models.GMF import GMF
from models.LightGCN import LightGCN
from models.MLP import MLP
from models.NeuMF import NeuMF
from models.SASRec import SASRec
from models.WMF import WMF
from train import Train
from utils import FlexibleDataLoader, MovieLens

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


fraction = 0.2

# Calculate the test size
test_size = fraction


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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
datasets = ['amazonoffice', 'epinions', 'gowalla', 'lastfm', 'movielens']
interactions = ['interactions','popularconsumptions']

for dataset in datasets:
    for interaction in interactions:
        # Construct file path dynamically
        file_path = f'/Users/recom/new_score_files/{dataset.lower()}_{interaction}_data.csv'

        # Read CSV file
        df = pd.read_csv(file_path)
        print("Original dataset group distribution:")
        print(df['group'].value_counts())

        dataset_type = 'explicit' if dataset in ['amazonoffice','movielens'] else 'implicit'


        # Model selection loop
        models = ['MLP', 'GMF', 'NeuMF', 'WMF', 'FM', 'DeepFM', 'LightGCN']

        for model_name in models:
            # Model and training configuration
            epochs = 10
            batch_size = 256
            learning_rate = 0.001
            factor = 8
            use_pretrain = False
            save_model = True

            # Initialize DataLoader
            data_loader = FlexibleDataLoader(df=df, dataset_type=dataset_type)
            processed_data = data_loader.read_data()
            train_df, validation_df, test_df, total_df = data_loader.split_train_test()

            print(f"Train size: {len(train_df)}")
            print(f"TValidation size: {len(validation_df)}")
            print(f"Test size: {len(test_df)}")

            # Create dataset objects
            train_dataset = MovieLens(train_df, total_df, ng_ratio=1,
                                      include_features=(
                                                  model_name == 'FM' or model_name == 'DeepFM'))  # ,use_sequence_model=(model_name == 'SASRec' or model_name == 'BERT4Rec'))
            print(f"Dataset size: {len(train_dataset)}")
            validation_dataset = MovieLens(validation_df, total_df, ng_ratio=50,
                                           include_features=(
                                                       model_name == 'FM' or model_name == 'DeepFM'))  # ,use_sequence_model=(model_name == 'SASRec' or model_name == 'BERT4Rec'))
            test_dataset = MovieLens(test_df, total_df, ng_ratio=50,
                                     include_features=(
                                                 model_name == 'FM' or model_name == 'DeepFM'))  # ,use_sequence_model=(model_name == 'SASRec' or model_name == 'BERT4Rec'))

            # Prepare DataLoader
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            print(f"Train size: {len(train_dataloader)}")
            print(f"TValidation size: {len(validation_dataloader)}")
            print(f"Test size: {len(test_dataloader)}")
            # Dynamically initializing models using dataset metadata
            num_users = train_dataset.get_num_users()
            num_items = train_dataset.get_num_items()
            num_features = train_dataset.get_num_features()

            models = {
                'MLP': MLP(num_users=num_users, num_items=num_items, num_factor=factor),
                'GMF': GMF(num_users=num_users, num_items=num_items, num_factor=factor),
                'NeuMF': NeuMF(num_users=num_users, num_items=num_items, num_factor=factor),
                'WMF': WMF(num_users=num_users, num_items=num_items, num_factors=factor),
                'FM': FactorizationMachine(num_factors=factor, num_features=num_features),
                'DeepFM': DeepFM(num_factors=factor, num_features=num_features),
                'LightGCN': LightGCN(num_users=num_users, num_items=num_items, embedding_size=factor, n_layers=3),
                'SASRec': SASRec(num_items=num_items, embedding_size=factor, num_heads=4, num_layers=2, dropout=0.1),
                'BERT4Rec': BERT4Rec(num_items=num_items, embedding_size=factor, num_heads=4, num_layers=2)
            }

            model = models[model_name].to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = torch.nn.BCELoss()

            trainer = Train(
                model=model,
                optimizer=optimizer,
                epochs=epochs,
                dataloader=train_dataloader,
                criterion=criterion,
                test_obj=test_dataloader,
                device=device,
                print_cost=True,
                use_features=model_name in ['FM', 'DeepFM'],
                use_weights=model_name == 'WMF',
                # use_sequences=model_name in ['SASRec', 'BERT4Rec']
            )
            trainer.train()

            # Generate and save results
            output_folder = 'output_files'
            os.makedirs(output_folder, exist_ok=True)

            validation_df = append_scores_and_data(model, validation_dataloader, device)
            test_df = append_scores_and_data(model, test_dataloader, device)

            validation_df.to_csv(f'{output_folder}/validations_with_scorRes_{dataset}_{interaction}_{model_name}.csv',
                                 index=False)
            test_df.to_csv(f'{output_folder}/tests_with_scorRes_{dataset}_{interaction}_{model_name}.csv', index=False)

            # Calculate and print metrics
            top_k = 10
            avg_hr_test, avg_ndcg_test = metrics(model, test_dataloader, top_k, device)
            avg_hr_val, avg_ndcg_val = metrics(model, validation_dataloader, top_k, device)

            print(f"Dataset: {dataset}, Interaction: {interaction}, Model: {model_name}")
            print(f"Average Hit Rate Test Set (HR@{top_k}): {avg_hr_test:.3f}")
            print(f"Average Normalized Discounted Cumulative Gain Test set (NDCG@{top_k}): {avg_ndcg_test:.3f}")
            print(f"Average Hit Rate Validation Set (HR@{top_k}): {avg_hr_val:.3f}")
            print(f"Average Normalized Discounted Cumulative Gain Validation Set (NDCG@{top_k}): {avg_ndcg_val:.3f}")

            # Optionally, save metrics to a file
            with open(f'{output_folder}/metrics_{dataset}_{interaction}_{model_name}.txt', 'w') as f:
                f.write(f"Dataset: {dataset}, Interaction: {interaction}, Model: {model_name}\n")
                f.write(f"Average Hit Rate Test Set (HR@{top_k}): {avg_hr_test:.3f}\n")
                f.write(f"Average Normalized Discounted Cumulative Gain Test set (NDCG@{top_k}): {avg_ndcg_test:.3f}\n")
                f.write(f"Average Hit Rate Validation Set (HR@{top_k}): {avg_hr_val:.3f}\n")
                f.write(
                    f"Average Normalized Discounted Cumulative Gain Validation Set (NDCG@{top_k}): {avg_ndcg_val:.3f}\n")

            del train_dataset, train_dataloader, validation_dataloader, validation_dataset, test_dataloader, test_dataset
            gc.collect()

