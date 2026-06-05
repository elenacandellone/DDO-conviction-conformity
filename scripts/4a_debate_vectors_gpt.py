import sys
sys.path.append('../src')
from imports import *

# 4a. Debate Vectors with GPT
# Input: debate_classification_gpt.pkl
# Output: debate vectors (d_vec_gpt.pkl) for debates with clear topic and stance

# --- Load topics from file ---
with open('../data/processed/txt/big_issues_topics.txt', 'r') as f:
    topics = [line.strip().lower() for line in f]

print(f"Total topics loaded: {len(topics)}")
print(topics)

# --- Load debate classification data ---
df = pd.read_pickle('../data/processed/pkl/debate_classification_gpt.pkl')
print(f"Number of debates classified: {df.shape[0]}")

# --- Map stance to numerical values ---
print("Stance value counts before mapping:")
print(df['stance'].value_counts())

#neutral mapped to 1
stance_mapping = {'pro': 1, 'neutral': 1, 'con': -1}
df['stance'] = df['stance'].map(stance_mapping)

# --- Filter out 'other' topics ---
df_red = df[df['topic'] != 'other']
print(f"Number of debates after removing 'other': {df_red.shape[0]} ({df_red.shape[0]/df.shape[0]*100:.2f}%)")

# --- One-hot encode topics ---
one_hot = pd.get_dummies(df_red['topic'], drop_first=False, dtype=float)
one_hot.index = df_red.debate_key

# Ensure all topics are included and reorder columns
for topic in topics:
    if topic not in one_hot.columns:
        one_hot[topic] = 0
one_hot = one_hot[topics]

# --- Assign vectors ---
df_red['vec'] = one_hot.values.tolist()
df_red['d_vec'] = df_red.apply(lambda row: [x * row['stance'] for x in row['vec']], axis=1)

# --- Save processed data ---
df_red.to_pickle('../data/processed/pkl/d_vec_gpt.pkl')
df_red.to_csv('../data/processed/csv/d_vec_gpt.csv', index=False)

df_red.head()
