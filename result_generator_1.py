from cal_methods_1 import apply_calibration_and_evaluate
from lambda_evaluator import DataSetEvaluator
from lambda_optimizer_ import LambdaOptimizer
import os
import pandas as pd
import math
import numpy as np


class DataSetProcessor:
    def __init__(self, base_path):
        self.base_path = base_path

    def process_datasets(self):
        final_results = []
        files = os.listdir(self.base_path)
        validation_files = [f for f in files if 'validation' in f]
        test_files = [f for f in files if 'test' in f]
        print("Validation Files:", validation_files)  # Debugging
        print("Test Files:", test_files)  # Debugging

        for validation_file in validation_files:
            base_name = validation_file.replace('validation_with_scores_', '')
            corresponding_test_file = 'test_with_scores_' + base_name
            if corresponding_test_file in test_files:
                print(f"Processing {validation_file} and {corresponding_test_file}...")
                validation_df = pd.read_csv(os.path.join(self.base_path, validation_file))
                test_df = pd.read_csv(os.path.join(self.base_path, corresponding_test_file))

                # Evaluate with dynamic lambda values
                optimizer = LambdaOptimizer(validation_df)
                lambda_values = optimizer.adjust_lambda()
                evaluator = DataSetEvaluator(test_df, lambda_values, alpha=0.40, file_name=corresponding_test_file)
                results = evaluator.evaluate()

                print("Results from evaluator:", results)  # Debugging

                # Calculate the floor of the average of average set sizes from current results
                df_results = pd.DataFrame(results)
                print("df results", df_results)
                df_results = add_difference_columns(df_results)
                print("Results after adding differences:", df_results)  # Debugging
                df_results['Average Set Size'] = df_results['Average Set Size'].astype(int)

                # Calculate the average set sizes
                avg_set_sizes = df_results.groupby(['Group'])['Average Set Size'].mean()
                overall_mean = avg_set_sizes.mean().astype(int)

                # Evaluate with fixed set size
                print("Using the avg set size in next step", overall_mean)
                additional_results = apply_calibration_and_evaluate(validation_df, test_df, overall_mean,
                                                                    corresponding_test_file)
                df_uncalibrated_results = pd.DataFrame(additional_results)
                df_uncalibrated_results= add_difference_columns(df_uncalibrated_results)

                # Combine and save all results
                combined_results = pd.concat([df_results, df_uncalibrated_results], ignore_index=True)
                final_results.append(combined_results)

        # Concatenate all results from all files and save
        final_df = pd.concat(final_results, ignore_index=True)
        save_results(final_df)


def save_results(results_df):
    # Check if the directory exists before writing to it
    output_dir = ('/Users/nitinbisht/PycharmProjects/recom/models/aug_9'
                  '/evaluation_results_paper_2_')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with pd.ExcelWriter(f'{output_dir}/all_evaluation_results_100_40_25.xlsx', engine='openpyxl') as writer:
        results_df.to_excel(writer, index=False)
    print("Results saved to 'evaluation_results.xlsx'.")


def add_difference_columns(df):
    """Add columns for differences between Hit Rate and NDCG for two groups in each file."""
    if not df.empty and 'Hit Rate Loss' in df.columns and 'NDCG Loss' in df.columns:
        # Ensure DataFrame is sorted by 'File' and 'Group' for consistent ordering
        df.sort_values(by=['File', 'Group'], inplace=True)

        # Group by 'File' and 'Method' and calculate differences
        grouped = df.groupby(['File', 'Method'])
        differences = grouped.apply(lambda x: pd.Series({
            'Hit Rate Diff': abs(x['Hit Rate Loss'].diff().iloc[-1]),
            'NDCG Diff': abs(x['NDCG Loss'].diff().iloc[-1])
        }))

        # Reset index so 'differences' can be merged properly
        differences = differences.reset_index()

        # Merge the differences back into the original DataFrame
        df = df.merge(differences, on=['File', 'Method'], how='left')
    else:
        print("Required columns are missing or the DataFrame is empty.")
        df['Hit Rate Diff'] = np.nan
        df['NDCG Diff'] = np.nan

    return df





if __name__ == "__main__":
    base_path = '/Users/nitinbisht/PycharmProjects/recom/output_files_100_req' # change 5 # Specify the directory containing your CSV files
    processor = DataSetProcessor(base_path)
    processor.process_datasets()

