import sys
sys.path.append('../src') 
from imports import *

# 2b. Debate Classification with BERT
# Input: debate_info.pkl
# Output: debate classification with Sentence-BERT + BART-large-MNLI (debate_classification_bert.pkl)

#open figs folder txt file to read the path
with open('./fig_folder.txt', 'r') as f:
    fig_folder = f.read().strip()

# Disable HuggingFace tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load debate information
debates = pd.read_pickle('../data/processed/pkl/debate_info.pkl')

#read topics from txt
topics = []
with open('../data/processed/txt/big_issues_topics.txt', 'r') as f:
    for line in f:
        topics.append(line.strip())
print(f"Total topics loaded: {len(topics)}")

#lowercase topics
topics = [topic.lower() for topic in topics]

#Sentence-BERT model for embeddings
sb_model = SentenceTransformer('all-MiniLM-L6-v2')
topic_embeddings = sb_model.encode(topics, convert_to_tensor=True)

# Zero-shot pipeline for stance detection
stance_classifier = pipeline("zero-shot-classification",
                             model="facebook/bart-large-mnli")


def classify_topic(row, threshold=0.25, sb_model=sb_model, topic_embeddings=topic_embeddings, topics=topics):
    '''
    Input: row of debate info dataframe
    Output: classified topic or 'other' if below threshold
    '''
    text = f"{row['debate_title']} {row['participant_1_rounds']} {row['participant_2_rounds']}"
    try:
        # Compute embedding and cosine similarities
        emb = sb_model.encode(text, convert_to_tensor=True)
        scores = util.cos_sim(emb, topic_embeddings)[0]
        # Take the best matching topic if above threshold
        best_idx = torch.argmax(scores).item()
        best_score = scores[best_idx].item()
        return topics[best_idx] if best_score >= threshold else "other"
    except Exception as e:
        return f"error: {e}"


def detect_stance(row, topic, stance_classifier=stance_classifier):
    '''
    Input: row of debate info dataframe and topic
    Output: detected stance ('pro', 'con', 'neutral')
    '''
    text = f"{row['debate_title']}"
    try:
        # If topic is 'other', set stance to neutral
        if topic == "other":
            return "neutral"
        result = stance_classifier(text, candidate_labels=["pro", "con", "neutral"], multi_label=False)
        return result["labels"][0]
    except Exception as e:
        return f"error: {e}"


def classify_row(row, topic_threshold=0.25):
    '''
    Input: row of debate info dataframe
    Output: debate_key, topic, stance
    '''
    topic = classify_topic(row, threshold=topic_threshold)
    stance = detect_stance(row, topic)
    return row["debate_key"], topic, stance


out_file = '../data/processed/csv/debate_classification_bert.csv'

# Load already processed debates to avoid reprocessing
if os.path.exists(out_file):
    done = pd.read_csv(out_file)
    done_ids = set(done["debate_key"])
    debates = debates[~debates["debate_key"].isin(done_ids)]
else:
    done = pd.DataFrame(columns=["debate_key", "topic", "stance"])

print(f"Remaining debates: {len(debates)}")

# -------------------------------------------------------
# Chunked parallel execution
# -------------------------------------------------------
chunk_size = 200

all_results = []

for i in range(0, len(debates), chunk_size):
    chunk = debates.iloc[i:i + chunk_size]
    chunk_id = i // chunk_size + 1
    print(f"\n=== Processing chunk {chunk_id} ({len(chunk)} rows) ===")

    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(classify_row)(row) for row in chunk.to_dict("records")
    )

    # Convert to DataFrame
    chunk_results = pd.DataFrame(results, columns=["debate_key", "topic", "stance"])

    # Print intermediate results
    for k, t, s in chunk_results.values:
        print(f"{k} → {t} | {s}")

    # Save intermediate results
    all_results.append(chunk_results)
    tmp = pd.concat([done] + all_results, ignore_index=True)
    tmp.to_csv(out_file, index=False)
    print(f"✓ Saved intermediate results: {len(tmp)} total so far")

print("\nAll chunks completed!")

#convert into pkl
#open file
df = pd.read_csv(out_file)
df.to_pickle('../data/processed/pkl/debate_classification_bert.pkl')
print(f"Saved final results: {len(df)} rows to pkl")


# Aggregate counts
#remove 'other' topic
df = df[df['topic'] != 'other']

# Aggregate counts
counts = df.groupby(['topic', 'stance']).size().unstack(fill_value=0)

# Order topics by total count (descending)
counts = counts.assign(total=counts.sum(axis=1)).sort_values('total', ascending=True)
counts = counts.drop(columns='total')

# Define consistent colors for each stance
# Replace or extend this with your stance categories
stance_colors = {
    'pro': '#0072B2',       # blue
    'con': '#D55E00',      # red
    'neutral': '#F0E442',   #yellow
}

# Filter color list to match the stance columns in counts
colors = [stance_colors[s] for s in counts.columns]

# Plot horizontally
ax = counts.plot(kind='barh', stacked=True, color=colors, figsize=(8, 7))

plt.xlabel("count (no. of debates)")
plt.ylabel("topic")
plt.title("topic and stance classification of debates")
plt.tight_layout()
plt.savefig('../plots/debates_classification_bert.pdf')
plt.savefig(f'{fig_folder}/debates_classification_bert.pdf')
