import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

# Set this to the directory where your Excel files are located
base_folder = '/path/to/your/excel/files'
output_folder = '/Users/recom/models/plots'  # Folder to save the plots

# Updated alpha and eta values as per your request
alpha_values = [10, 20, 30]
eta_values = [10, 15, 20, 25, 30, 35, 40, 45, 50]
datasets = ['movielens', 'amazonoffice']
interactions = ['interactions', 'popularconsumption']
models = ['DeepFM', 'GMF', 'LightGCN', 'MLP', 'NeuMF']
metrics = ['Average Set Size', 'Hit Rate Diff', 'NDCG Diff']

# Ensure the main output folder exists
os.makedirs(output_folder, exist_ok=True)

# Define a fixed set of colors for the models
color_map = {
    'DeepFM': 'blue',
    'GMF': 'orange',
    'LightGCN': 'green',
    'MLP': 'red',
    'NeuMF': 'purple'
}

# Function to load the data from Excel files
def load_data(eta, alpha):
    folder = f'/Users/recom/models/alpha_{alpha}'
    file_name = f'all_evaluation_results_100_{alpha}_{eta}.xlsx'
    file_path = os.path.join(folder, file_name)
    return pd.read_excel(file_path)

# Function to filter the data based on Method = 'Conformal'
def filter_data(df):
    return df[df['Method'] == 'Conformal']

# Function to prepare and plot the data
def plot_metric(df_all, alpha, dataset, interaction, metric, models, eta_values, output_folder):
    plt.figure(figsize=(12, 8))

    lines = []
    labels = []

    for model in models:
        # For Average Set Size, we want to plot the average of Group 1 and Group 2
        if metric == 'Average Set Size':
            # Filter data for the current model and relevant groups
            subset_group1 = df_all[(df_all['File'].str.contains(dataset)) &
                                   (df_all['File'].str.contains(interaction)) &
                                   (df_all['File'].str.contains(model)) &
                                   (df_all['Group'] == 1)]
            subset_group2 = df_all[(df_all['File'].str.contains(dataset)) &
                                   (df_all['File'].str.contains(interaction)) &
                                   (df_all['File'].str.contains(model)) &
                                   (df_all['Group'] == 2)]

            # Ensure both groups have the same eta_values
            if not subset_group1.empty and not subset_group2.empty:
                # Calculate the average of Group 1 and Group 2
                avg_values = (subset_group1[metric].values + subset_group2[metric].values) / 2
                avg_values = avg_values.astype(int)  # Convert to integer

                eta_transformed = subset_group1['eta_value'].tolist()

                # Print the exact data being plotted for each line
                print(f"\nPlotting data for model: {model}, dataset: {dataset}, interaction: {interaction}, alpha: {alpha}, metric: {metric}")
                print(pd.DataFrame({'eta_value': eta_transformed, metric: avg_values}))

                line, = plt.plot(eta_transformed, avg_values, linestyle='-', marker='o',
                                 color=color_map[model])
                lines.append(line)
                labels.append(f'{model} +CUGF')

        else:
            # For other metrics, plot Group 1 as before
            subset = df_all[(df_all['File'].str.contains(dataset)) &
                            (df_all['File'].str.contains(interaction)) &
                            (df_all['File'].str.contains(model)) &
                            (df_all['Group'] == 1)]

            if subset.empty:
                print(f"No data for {model} in {dataset} {interaction} with alpha = {alpha}")
                continue

            eta_transformed = subset['eta_value'].tolist()

            # Print the exact data being plotted for each line
            print(f"\nPlotting data for model: {model}, dataset: {dataset}, interaction: {interaction}, alpha: {alpha}, metric: {metric}")
            print(subset[['eta_value', metric]])

            line, = plt.plot(eta_transformed, subset[metric], linestyle='-', marker='o',
                             color=color_map[model])
            lines.append(line)
            labels.append(f'{model} +CUGF')

    # Check if there is any line to plot before proceeding
    if not lines:
        print(f"No valid data to plot for {dataset} {interaction} with alpha = {alpha}")
        plt.close()
        return

    # Create the model legend
    model_legend = plt.legend(lines, labels, loc='best', title="Models", fontsize='small')
    plt.gca().add_artist(model_legend)  # Add model legend to the plot

    plt.xlabel(r'$\eta$')
    plt.ylabel(metric)
    # plt.grid(True)

    # Ensure the subdirectory for this alpha exists
    alpha_folder = os.path.join(output_folder, f'alpha_{alpha}')
    os.makedirs(alpha_folder, exist_ok=True)

    # Save the plot in the respective alpha folder
    file_name = f'{metric}_{dataset}_{interaction}_alpha_{alpha}.png'
    plt.savefig(os.path.join(alpha_folder, file_name))
    plt.close()  # Close the plot to free up memory

# Loop over alpha values, datasets, interactions, and metrics to generate and save plots
for alpha in alpha_values:
    for dataset in datasets:
        for interaction in interactions:
            for metric in metrics:
                df_all = pd.DataFrame()  # Placeholder to accumulate data for each metric

                for eta in eta_values:
                    df = load_data(eta, alpha)
                    df_filtered = filter_data(df).copy()
                    df_filtered['eta_value'] = eta/100
                    df_all = pd.concat([df_all, df_filtered])

                # Debug: Print the accumulated data to verify it contains all eta_values for the current metric
                print(f"\nAccumulated Data for alpha = {alpha}, dataset = {dataset}, interaction = {interaction}, metric = {metric}:")
                print(df_all[['File', 'Group', 'eta_value', metric]])

                plot_metric(df_all, alpha, dataset, interaction, metric, models, eta_values, output_folder)
