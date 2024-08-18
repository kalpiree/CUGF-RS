import pandas as pd

user_top_fraction = 0.5
dataset = pd.read_csv('/Users/recom/paper/amazonoffice.txt', sep="\s+", header=0,
                      names=['userId', 'itemId', 'rating'], dtype={'userId': int, 'itemId': int, 'rating': int}, skiprows=1)
dataset = dataset.dropna()
dataset = dataset.dropna()
is_implicit = False


# Function to read item popularity based on implicit feedback
def read_implicit_item_popularity(dataset):
    items_freq = {}
    user_interactions = {}
    for eachline in dataset.itertuples(index=False):
        uid, iid, count = int(eachline.userId), int(eachline.itemId), int(eachline.rating)
        user_interactions.setdefault(uid, []).extend([iid] * count)
        items_freq[iid] = items_freq.get(iid, 0) + count
    return items_freq, user_interactions


# Function to read item popularity based on explicit feedback
def read_explicit_item_popularity(dataset):
    items_freq = {}
    user_interactions = {}
    for eachline in dataset.itertuples(index=False):
        uid, iid = int(eachline.userId), int(eachline.itemId)
        user_interactions.setdefault(uid, []).append(iid)
        items_freq[iid] = items_freq.get(iid, 0) + 1
    return items_freq, user_interactions


# Determine the top items (short_heads)
def determine_top_items(items_freq, item_top_fraction):
    sorted_items = sorted(items_freq.items(), key=lambda x: x[1], reverse=True)
    top_items_count = int(len(items_freq) * item_top_fraction)
    return set([item for item, _ in sorted_items[:top_items_count]])


# Function to group users and classify items
def update_dataset_with_groups(dataset, user_top_fraction, method, items_freq, user_interactions):
    short_heads = determine_top_items(items_freq, 0.2)  # Determine the top 20% items

    # Classify items as short heads (1) or long tails (2)
    dataset['item_type'] = dataset['itemId'].apply(lambda x: 1 if x in short_heads else 2)

    user_profile_pop_df = pd.DataFrame(
        [(uid, len(set(items) & short_heads), len(items)) for uid, items in user_interactions.items()],
        columns=['uid', 'pop_count', 'profile_size']
    )

    if method == "popular_consumption":
        user_profile_pop_df.sort_values(['pop_count', 'profile_size'], ascending=(False, False), inplace=True)
    else:  # interactions
        user_profile_pop_df.sort_values(['profile_size'], ascending=False, inplace=True)

    num_top_users = int(user_top_fraction * len(user_interactions))
    found = False
    adjustment = 0
    deviation = int(0.3 * len(user_interactions))
    while not found and adjustment <= deviation:
        index = num_top_users + (adjustment - int(deviation / 2))
        if index <= 0 or index >= len(user_interactions):
            adjustment += 1
            continue

        if method == "popular_consumption":
            condition = (user_profile_pop_df.iloc[index - 1]['pop_count'] > user_profile_pop_df.iloc[index][
                'pop_count'] + 1 and
                         user_profile_pop_df.iloc[index - 1]['profile_size'] > user_profile_pop_df.iloc[index][
                             'profile_size'] + 1)
        else:  # interactions
            condition = user_profile_pop_df.iloc[index - 1]['profile_size'] > user_profile_pop_df.iloc[index][
                'profile_size'] + 1

        if condition:
            found = True
        else:
            adjustment += 1

    advantaged_users = user_profile_pop_df.head(index)
    disadvantaged_users = user_profile_pop_df.iloc[index:]

    # Assign user groups in the original dataset
    dataset['group'] = dataset['userId'].apply(lambda x: 1 if x in advantaged_users['uid'].values else 2)

    # Print max and min total and popular items interactions for each group
    print("Advantaged Users:")
    print("Number of Users:", len(advantaged_users))
    print("Max Total Interactions:", advantaged_users['profile_size'].max())
    print("Min Total Interactions:", advantaged_users['profile_size'].min())
    print("Max Popular Item Interactions:", advantaged_users['pop_count'].max())
    print("Min Popular Item Interactions:", advantaged_users['pop_count'].min())

    print("\nDisadvantaged Users:")
    print("Number of Users:", len(disadvantaged_users))
    print("Max Total Interactions:", disadvantaged_users['profile_size'].max())
    print("Min Total Interactions:", disadvantaged_users['profile_size'].min())
    print("Max Popular Item Interactions:", disadvantaged_users['pop_count'].max())
    print("Min Popular Item Interactions:", disadvantaged_users['pop_count'].min())

    return dataset


# Choose the appropriate item popularity function based on feedback type
if is_implicit:
    items_freq, user_interactions = read_implicit_item_popularity(dataset)
else:
    items_freq, user_interactions = read_explicit_item_popularity(dataset)

# Configuration for grouping methods
methods = [
    ('popular_consumption', 0.5),  # 50% top users based on popular item consumption
    ('interactions', 0.5)  # 50% top users based on total interactions
]

# Process each method and save the resulting datasets
for method, fraction in methods:
    updated_dataset = update_dataset_with_groups(dataset.copy(), fraction, method, items_freq, user_interactions)
    updated_dataset.to_csv(f'{method}_data.csv', index=False)
