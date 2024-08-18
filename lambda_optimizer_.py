
import numpy as np

class LambdaOptimizer:
    def __init__(self, df, initial_lambda=None, alpha=0.40, epsilon=0.05, stability_threshold=0.02,
                 max_stable_iterations=10, min_lambda=0, delta=0.2,
                 hit_rate_diff_threshold=0.25, ndcg_diff_threshold=0.25, score_option=1):
        self.df = df
        self.alpha = alpha
        self.epsilon = epsilon
        self.stability_threshold = stability_threshold
        self.max_stable_iterations = max_stable_iterations
        self.min_lambda = min_lambda
        self.delta = delta
        self.hit_rate_diff_threshold = hit_rate_diff_threshold
        self.ndcg_diff_threshold = ndcg_diff_threshold
        self.lambda_values = initial_lambda if initial_lambda is not None else self.initialize_lambda_values()
        self.score_option = score_option

    def initialize_lambda_values(self):
        lambda_values = {}
        for group in self.df['group'].unique():
            group_df = self.df[self.df['group'] == group]
            lambda_values[group] = group_df['score'].quantile \
                (0.8)  # Initial lambda set to the 80th percentile of scores in the group
        return lambda_values

    def score_function(self, m_u_i_true, m_u_i, gamma):
        if self.score_option == 1:
            return int(m_u_i_true >= gamma) * np.sum([m_u_i[j] * int(m_u_i[j] >= gamma) for j in range(len(m_u_i))])
        elif self.score_option == 2:
            return int(m_u_i_true >= gamma) * np.sum \
                ([m_u_i[j] * int(m_u_i[j] >= m_u_i_true) for j in range(len(m_u_i))])
        elif self.score_option == 3:
            return np.sum([m_u_i[j] * int(m_u_i[j] >= gamma) for j in range(len(m_u_i))])

    def calculate_quantile(self, scores, alpha):
        n = len(scores)
        position = np.ceil((n + 1) * (1 - alpha)) / n
        return np.quantile(scores, position)

    def calculate_metrics(self):
        results = {}
        item_sets = {group: {} for group in self.df['group'].unique()}

        for group_label in self.df['group'].unique():
            group_df = self.df[self.df['group'] == group_label].copy()
            group_df.sort_values(by='score', ascending=False, inplace=True)
            lambda_value = self.lambda_values[group_label]

            def user_loss(user_items, user_id):
                labels = group_df[(group_df['itemId'].isin(user_items)) & (group_df['userId'] == user_id)]['label']
                return 0 if any(labels == 1) else 1

            def hit(gt_item, pred_items):
                return 1 if gt_item in pred_items else 0

            def ndcg(gt_item, pred_items):
                if gt_item in pred_items:
                    index = pred_items.index(gt_item)
                    return 1 - 1 / np.log2(index + 2)
                return 1

            user_losses = []
            hits = []
            ndcgs = []
            user_scores_list = []

            # Calculate user scores for the group to determine quantile value
            for user_id in group_df['userId'].unique():
                user_data = group_df[group_df['userId'] == user_id]
                true_items = user_data[user_data['label'] == 1]['itemId'].values
                if len(true_items) > 0:
                    true_item = true_items[0]
                    scores = user_data['score'].values
                    user_score = self.score_function(true_item, scores, lambda_value)
                    user_scores_list.append(user_score)

            # Calculate the quantile value for the group
            if user_scores_list:
                quantile_value = self.calculate_quantile(user_scores_list, self.alpha)
                print(f"Group {group_label} quantile value: {quantile_value}")

                for user_id in group_df['userId'].unique():
                    user_data = group_df[group_df['userId'] == user_id]
                    true_items = user_data[user_data['label'] == 1]['itemId'].values
                    if len(true_items) > 0:
                        true_item = true_items[0]
                        sorted_items = user_data.sort_values(by='score', ascending=False)
                        included_items = []
                        cumulative_score = 0

                        for i in range(len(sorted_items)):
                            if cumulative_score >= quantile_value:
                                break
                            included_items.append(sorted_items.iloc[i]['itemId'])
                            cumulative_score += sorted_items.iloc[i]['score']

                        user_losses.append(user_loss(included_items, user_id))
                        hits.append(hit(true_item, included_items))
                        ndcgs.append(ndcg(true_item, included_items))

                        item_sets[group_label][user_id] = included_items  # Store included items per user

            average_loss = np.mean(user_losses) if user_losses else 1
            loss_variance = np.var(user_losses) if user_losses else 0
            inclusion_rate = group_df[group_df['label'] == 1]['score'].apply(lambda x: x >= lambda_value).mean() if not group_df[group_df['label'] == 1].empty else 0
            hit_rate = np.mean(hits) if hits else 0
            ndcg_rate = np.mean(ndcgs) if ndcgs else 0

            results[group_label] = {
                'Lambda': lambda_value,
                'Loss': average_loss,
                'Loss Variance': loss_variance,
                'Inclusion Rate': inclusion_rate,
                'Hit Rate': hit_rate,
                'NDCG Loss': ndcg_rate,
            }

        return results, item_sets

    def adjust_lambda(self):
        stability_count = 0
        previous_hit_rate_diff = float('inf')
        previous_ndcg_diff = float('inf')
        large_adjustment_step = 0.03
        small_adjustment_step = 0.02  # Smaller step for fine adjustments

        iteration = 0
        while stability_count < self.max_stable_iterations:
            print(f"Iteration: {iteration}")
            print(f"Current lambda values: {self.lambda_values}")
            all_groups = list(self.lambda_values.keys())
            quantile_values = {}

            for group in all_groups:
                group_scores = []
                for user_id in self.df[self.df['group'] == group]['userId'].unique():
                    user_data = self.df[(self.df['userId'] == user_id) & (self.df['group'] == group)]
                    if user_data[user_data['label'] == 1].empty:
                        print(f"User {user_id} in Group {group} has no true items.")
                        continue

                    m_u_i_true = user_data[user_data['label'] == 1]['score'].values[0]
                    m_u_i = user_data['score'].values
                    gamma = self.lambda_values[group]
                    group_scores.append(self.score_function(m_u_i_true, m_u_i, gamma))

                if group_scores:
                    quantile_value = self.calculate_quantile(group_scores, self.alpha)
                    quantile_values[group] = quantile_value
                    print(f"Group {group} quantile value: {quantile_value}")
                else:
                    # Handle case with no scores
                    quantile_values[group] = float('inf')
                    print(f"Group {group} has no scores. Setting quantile value to infinity.")

            # Construct prediction sets based on the quantile values directly in hit rate and NDCG calculations
            metrics, item_sets = self.calculate_metrics()
            print(f"Updated metrics: {metrics}")

            hit_rate_diff = abs(metrics[all_groups[0]]['Hit Rate'] - metrics[all_groups[1]]['Hit Rate'])
            print(f"Hit rate difference: {hit_rate_diff}")

            ndcg_diff = abs(metrics[all_groups[0]]['NDCG Loss'] - metrics[all_groups[1]]['NDCG Loss'])
            print(f"NDCG difference: {ndcg_diff}")

            losses_within_alpha = all(metrics[group]['Loss'] <= self.alpha for group in all_groups)
            print(f"Losses within alpha: {losses_within_alpha}")

            if hit_rate_diff <= self.hit_rate_diff_threshold and ndcg_diff <= self.ndcg_diff_threshold and losses_within_alpha:
                print("Hit rate and NDCG differences within thresholds and losses within alpha. Stopping iteration.")
                break

            if abs(hit_rate_diff - previous_hit_rate_diff) <= self.stability_threshold and abs \
                    (ndcg_diff - previous_ndcg_diff) <= self.stability_threshold:
                stability_count += 1
            else:
                stability_count = 0

            adjustment_step = large_adjustment_step if hit_rate_diff > self.hit_rate_diff_threshold or ndcg_diff > self.ndcg_diff_threshold else small_adjustment_step

            # Adjust lambda based on hit rate difference, NDCG difference, and average loss
            if not losses_within_alpha:
                group_to_adjust = [group for group in all_groups if metrics[group]['Loss'] > self.alpha]
                print(f"Adjusting lambda for loss exceeding alpha: Groups {group_to_adjust}")
            elif hit_rate_diff > self.hit_rate_diff_threshold:
                group_to_adjust = all_groups[0] if metrics[all_groups[0]]['Hit Rate'] < metrics[all_groups[1]]
                    ['Hit Rate'] else all_groups[1]
                print(f"Adjusting lambda for hit rate difference: Group {group_to_adjust}")
            elif ndcg_diff > self.ndcg_diff_threshold:
                group_to_adjust = all_groups[0] if metrics[all_groups[0]]['NDCG Loss'] < metrics[all_groups[1]]
                    ['NDCG Loss'] else all_groups[1]
                print(f"Adjusting lambda for NDCG difference: Group {group_to_adjust}")

            if isinstance(group_to_adjust, list):
                for group in group_to_adjust:
                    old_lambda = self.lambda_values[group]
                    self.lambda_values[group] = max(self.min_lambda, self.lambda_values[group] - adjustment_step)
                    print(f"Lambda for group {group} adjusted from {old_lambda} to {self.lambda_values[group]}")
            else:
                old_lambda = self.lambda_values[group_to_adjust]
                self.lambda_values[group_to_adjust] = max(self.min_lambda, self.lambda_values[group_to_adjust] - adjustment_step)
                print \
                    (f"Lambda for group {group_to_adjust} adjusted from {old_lambda} to {self.lambda_values[group_to_adjust]}")
                print \
                    (f"Group {group_to_adjust}: Hit Rate: {metrics[group_to_adjust]['Hit Rate']}, NDCG: {metrics[group_to_adjust]['NDCG Loss']}")

            previous_hit_rate_diff = hit_rate_diff
            previous_ndcg_diff = ndcg_diff
            iteration += 1

        return self.lambda_values
