
<h1>CUGF: A Reliable and Fair Recommendation Framework</h1>

<h2>Overview</h2>

<p>CUGF (Conformal Uncertainty Guarantees for Fairness) is a state-of-the-art recommendation framework that ensures guaranteed performance and fairness in recommender systems by leveraging conformal methods. This repository provides the necessary scripts and resources to preprocess data, train models, evaluate results, and visualize fairness and performance metrics.</p>

<h2>Project Structure</h2>
<ul>
<li><code>Models/</code> - Directory containing model architectures.</li>
<li><code>ratings_dataset/</code> - Directory containing the dataset files.</li>
<li><code>cal_methods.py</code> - Other Calibration methods.</li>
<li><code>evaluation.py</code> - Evaluation metrics.</li>
<li><code>lambda_evaluator.py</code> - Evaluation of gamma values on test files.</li>
<li><code>lambda_optimizer.py</code> - Scripts for optimizing lambda parameters.</li>
<li><code>plots_fairness.py</code> - Scripts for generating fairness plots.</li>
<li><code>plots_performance.py</code> - Scripts for generating performance plots.</li>
<li><code>preprocessing.py</code> - Preprocessing pipeline.</li>
<li><code>result_generator.py</code> - Generates results for final analysis.</li>
<li><code>train.py</code> - Script for training the models.</li>
<li><code>utils.py</code> - Utility functions.</li>
<li><code>run.py</code> - Main script to execute generate scores.</li>
<li><code>requirements.txt</code> - Python dependencies.</li>
</ul>

<h2>Installation</h2>

<p>First, ensure that you have Python installed. Clone this repository and install the necessary dependencies using:</p>

<pre><code>pip install -r requirements.txt
</code></pre>

<h2>How to Run</h2>

<h3>Step 1: Preprocessing</h3>

<p>Run the preprocessing script to prepare the dataset.</p>

<pre><code>python preprocessing.py --input_file '/path/to/your/data/ratings_data.txt' --output_folder './processed_data' --is_implicit --user_top_fraction 0.5 --methods popular_consumption interactions --datasets amazonoffice movielens
</code></pre>

<p><strong>Note:</strong> Replace <code>/path/to/your/data/ratings_data.txt</code> with your actual file path.</p>

<h3>Step 2: Training</h3>

<p>Use the processed data as input to train the models.</p>

<pre><code>python run.py --datasets amazonoffice movielens --models MLP LightGCN --epochs 15 --batch_size 512 --output_folder "results_folder"
</code></pre>

<h3>Step 3: Generate Results</h3>

<p>Take the output from the training step and generate final results.</p>

<pre><code>python result_generator.py --input_folder "results_folder" --output_folder "final_results"
</code></pre>

<h3>Step 4: Visualization</h3>

<p>Finally, generate the plots for fairness and performance metrics.</p>

<pre><code>python plots_fairness.py --input_folder "final_results" --output_folder "fairness_plots"
python plots_performance.py --input_folder "final_results" --output_folder "performance_plots"
</code></pre>
