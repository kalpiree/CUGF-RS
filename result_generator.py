from lambda_evaluator import DataSetEvaluator
from lambda_optimizer_ import LambdaOptimizer
import os
import pandas as pd


class DataSetProcessor:
    def __init__(self, base_path):
        self.base_path = base_path

    def process_datasets(self):
        results = []
        files = os.listdir(self.base_path)
        validation_files = [f for f in files if 'validation' in f]
        test_files = [f for f in files if 'test' in f]

        for validation_file in validation_files:
            base_name = validation_file.replace('validation_with_scores_', '')
            corresponding_test_file = 'test_with_scores_' + base_name

            if corresponding_test_file in test_files:
                print(f"Processing {validation_file} and {corresponding_test_file}...")
                validation_df = pd.read_csv(os.path.join(self.base_path, validation_file))
                test_df = pd.read_csv(os.path.join(self.base_path, corresponding_test_file))

                optimizer = LambdaOptimizer(validation_df)
                lambda_values = optimizer.adjust_lambda()

                evaluator = DataSetEvaluator(test_df, lambda_values, alpha=0.50, file_name=corresponding_test_file)  # change here 1
                evaluation_results = evaluator.evaluate()
                results.extend(evaluation_results)

                # Print results for the current file
                for result in evaluation_results:
                    print(
                        f"File: {result['File']}, Group: {result['Group']},  NDCG Loss: {result['NDCG Loss']}, Hit Rate "
                        f"Loss: {result['Hit Rate Loss']}")
        # print("All results saved to 'all_evaluation_results_100.xlsx'.")

        # Check if the directory exists before writing to it
        output_dir = ('/Users/recom/models/aug_9'
                      '/evaluation_results_paper_2_')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Write all results to a single Excel file
        df_output = pd.DataFrame(results)
        with pd.ExcelWriter(f'{output_dir}/all_evaluation_results_100_20_20.xlsx', engine='openpyxl') as writer:
            df_output.to_excel(writer, index=False)
        print("All results saved to 'all_evaluation_results_100_50.xlsx'.")


if __name__ == "__main__":
    base_path = '/Users/recom/output_files_100_req'
    processor = DataSetProcessor(base_path)
    processor.process_datasets()

