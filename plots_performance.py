

#
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D



base_folder = 'path/to/excel/files'
output_folder = 'path/to/output/plots'  # Folder to save the plots


eta_values = [10, 15, 20]
x_values = [10, 15, 20, 25, 30, 35, 40, 45, 50]
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
def load_data(eta, x):
    folder = f'{base_folder}/evaluation_results/eta_{eta}'
    file_name = f'all_evaluation_results_100_{x}_{eta}.xlsx'
    file_path = os.path.join(folder, file_name)
    return pd.read_excel(file_path)

# Function to filter the data based on Method = 'Conformal'
def filter_data(df):
    return df[df['Method'] == 'Conformal']


def plot_metric(df_all, eta, dataset, interaction, metric, models, x_values, output_folder):
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

            # Ensure both groups have the same x_values
            if not subset_group1.empty and not subset_group2.empty:
                # Calculate the average of Group 1 and Group 2
                avg_values = (subset_group1[metric].values + subset_group2[metric].values) / 2
                avg_values = avg_values.astype(int)  # Convert to integer

                x_transformed = subset_group1['x_value'].tolist()

                # Print the exact data being plotted for each line
                print(
                    f"\nPlotting data for model: {model}, dataset: {dataset}, interaction: {interaction}, eta: {eta}, metric: {metric}")
                print(pd.DataFrame({'x_value': x_transformed, metric: avg_values}))

                line, = plt.plot(x_transformed, avg_values, linestyle='-', marker='o',
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
                print(f"No data for {model} in {dataset} {interaction} with eta = {eta}")
                continue

            x_transformed = subset['x_value'].tolist()

            # Print the exact data being plotted for each line
            print(
                f"\nPlotting data for model: {model}, dataset: {dataset}, interaction: {interaction}, eta: {eta}, metric: {metric}")
            print(subset[['x_value', metric]])

            line, = plt.plot(x_transformed, subset[metric], linestyle='-', marker='o',
                             color=color_map[model])
            lines.append(line)

            labels.append(f'{model} +CUGF')

    # Check if there is any line to plot before proceeding
    if not lines:
        print(f"No valid data to plot for {dataset} {interaction} with eta = {eta}")
        plt.close()
        return

    # Create the model legend
    model_legend = plt.legend(lines, labels, loc='best', title="Models", fontsize='small')
    plt.gca().add_artist(model_legend)  # Add model legend to the plot

    # For 'Average Set Size', do not add the group legend since we're averaging them
    if metric in ['Hit Rate Diff', 'NDCG Diff']:
        plt.axhline(y=eta / 100, color='darkred', linestyle=':', linewidth=1)
    # Use LaTeX for the X-axis label to display "1 - α"
    plt.xlabel(r'$\alpha$')
    plt.ylabel(metric)
    # plt.grid(True)

    # Ensure the subdirectory for this eta exists
    eta_folder = os.path.join(output_folder, f'eta_{eta}')
    os.makedirs(eta_folder, exist_ok=True)

    # Save the plot in the respective eta folder
    file_name = f'{metric}_{dataset}_{interaction}_eta_{eta}.png'
    plt.savefig(os.path.join(eta_folder, file_name))
    plt.close()  # Close the plot to free up memory


# Loop over eta values, datasets, interactions, and metrics to generate and save plots
for eta in eta_values:
    for dataset in datasets:
        for interaction in interactions:
            for metric in metrics:
                df_all = pd.DataFrame()  # Placeholder to accumulate data for each metric

                for x in x_values:
                    df = load_data(eta, x)
                    df_filtered = filter_data(df).copy()
                    df_filtered['x_value'] =  (x / 100)
                    df_all = pd.concat([df_all, df_filtered])


                print(f"\nAccumulated Data for eta = {eta}, dataset = {dataset}, interaction = {interaction}, metric = {metric}:")
                print(df_all[['File', 'Group', 'x_value', metric]])

                plot_metric(df_all, eta, dataset, interaction, metric, models, x_values, output_folder)
