import sys
sys.path.append('../src')
from imports import *

# 2a. Debate Classification with GPT
# Input: debate_info.pkl
# Output: debate zero-shot classification with gpt-40-mini (debate_classification_gpt.pkl)

#open figs folder txt file to read the path
with open('./fig_folder.txt', 'r') as f:
    fig_folder = f.read().strip()

# --- Load debate data ---
debates = pd.read_pickle('../data/processed/pkl/debate_info.pkl')

#read topics from txt
topics = []
with open('../data/processed/txt/big_issues_topics.txt', 'r') as f:
    for line in f:
        topics.append(line.strip())
print(f"Total topics loaded: {len(topics)}")

#lowercase topics
topics = [topic.lower() for topic in topics]
#topics in Literal format
structured_topics = ',\n  '.join([f"'{topic}'" for topic in topics])


# --- Initialize model ---
os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
model = init_chat_model(
    "gpt-4o-mini",
    model_provider="openai",
    temperature=0,
    max_tokens=100
)

# --- Structured output classes ---
class DebateTopicFormat(BaseModel):
    topic: Literal['abortion', 'affirmative action', 'animal rights', 'barack obama', 'border fence', 'capitalism', 'civil unions', 'death penalty', 'drug legalization', 'electoral college', 'environmental protection', 'estate tax', 'european union', 'euthanasia', 'federal reserve', 'flat tax', 'free trade', 'gay marriage', 'global warming exists', 'globalization', 'gold standard', 'gun rights', 'homeschooling', 'internet censorship', 'iran-iraq war', 'labor union', 'legalized prostitution', 'medicaid & medicare', 'medical marijuana', 'military intervention', 'minimum wage', 'national health care', 'national retail sales tax', 'occupy movement', 'progressive tax', 'racial profiling', 'redistribution', 'smoking ban', 'social programs', 'social security', 'socialism', 'stimulus spending', 'term limits', 'torture', 'united nations', 'war in afghanistan', 'war on terror', 'welfare','other']

class CommentStanceFormat(BaseModel):
    stance: Literal["pro", "con", "neutral"]

# --- Unified prompt template ---
system_prompt = '''
You are a political debate analyzer. You will be given a debate title and participants' comments.

## TASKS ##
1. Identify the single most appropriate debate topic based on the debate title and all participant comments.
2. Classify the stance of the debate with respect to the identified topic.

## STANCE DEFINITION ##
The stance must be evaluated **relative to the identified topic**, not relative to the specific policy, object, or practice mentioned in the debate title.

- "pro": The debate supports, promotes, or advances the values, goals, or principles of the identified topic.
- "con": The debate opposes, criticizes, or undermines the values, goals, or principles of the identified topic.
- "neutral": The debate is descriptive, mixed, or unclear with respect to the identified topic.

IMPORTANT:
- If a debate argues against a practice because it conflicts with the values of a topic, the stance should be "pro" toward that topic.
  (Example: “Zoos should be banned” → topic: "animal rights" → stance: "pro")

## OUTPUT FORMAT ##
Return the output in the following JSON format and nothing else:

{{
  "topic": "<topic_label>",
  "stance": "<stance_label>"
}}

## TOPIC CONSTRAINT ##
Choose ONLY ONE topic from the following list:

['abortion', 'affirmative action', 'animal rights', 'barack obama', 'border fence', 'capitalism', 'civil unions', 'death penalty', 'drug legalization', 'electoral college', 'environmental protection', 'estate tax', 'european union', 'euthanasia', 'federal reserve', 'flat tax', 'free trade', 'gay marriage', 'global warming exists', 'globalization', 'gold standard', 'gun rights', 'homeschooling', 'internet censorship', 'iran-iraq war', 'labor union', 'legalized prostitution', 'medicaid & medicare', 'medical marijuana', 'military intervention', 'minimum wage', 'national health care', 'national retail sales tax', 'occupy movement', 'progressive tax', 'racial profiling', 'redistribution', 'smoking ban', 'social programs', 'social security', 'socialism', 'stimulus spending', 'term limits', 'torture', 'united nations', 'war in afghanistan', 'war on terror', 'welfare', 'other']

'''

prompt_template = ChatPromptTemplate([
    ("system", system_prompt),
    ("user", "{debate_info}")
])

# --- Combined structured output ---
class CombinedFormat(BaseModel):
    topic: DebateTopicFormat.__annotations__['topic']
    stance: CommentStanceFormat.__annotations__['stance']

model_structured = model.with_structured_output(CombinedFormat)

output_file = "../data/processed/pkl/debate_classification_gpt.pkl"

# --- Prepare results container ---
if os.path.exists(output_file):
    # Load existing results to continue
    results_df = pd.read_pickle(output_file)
    results = results_df.to_dict("records")
    processed_keys = set(results_df['debate_key'])
    print(f"Resuming from {len(results)} already processed debates.")
else:
    results = []
    processed_keys = set()


def truncate_text(text, max_chars=8000):
    if text is None:
        return ""
    return text[:max_chars]


# --- Process debates ---
for idx, row in tqdm(debates.iterrows(), total=len(debates), desc="Processing Debates"):
    debate_key = row['debate_key']
    if debate_key in processed_keys:
        continue  # skip already processed debates

    p1_text = truncate_text(row['participant_1_rounds'], max_chars=1000)
    p2_text = truncate_text(row['participant_2_rounds'], max_chars=1000)

    combined_info = (
    f"Debate Title:\n{row['debate_title']}\n\n"
    f"Participant 1:\n{p1_text}\n\n"
    f"Participant 2:\n{p2_text}"
    )


    prompt_request = prompt_template.invoke({"debate_info": combined_info})
    structured_response = model_structured.invoke(prompt_request)
    response_dict = dict(structured_response)

    # Append result
    results.append({
        "debate_key": debate_key,
        "topic": response_dict["topic"],
        "stance": response_dict["stance"]
    })
    #print(f"Processed debate {debate_key}: Topic - {response_dict['topic']}, Stance - {response_dict['stance']}")

    # --- Save after every iteration ---
    results_df = pd.DataFrame(results)
    results_df.to_pickle(output_file)

print(f"All results saved to {output_file}")

# Aggregate counts
#remove 'other' topic
df = results_df[results_df['topic'] != 'other']

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
plt.savefig('../plots/debates_classification_gpt.pdf')
plt.savefig(f'{fig_folder}/debates_classification_gpt.pdf')
