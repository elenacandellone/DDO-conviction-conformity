import sys
sys.path.append('../src') 
from imports import *

# 3. User Vectors and Demographics
# Input: users.json, votes.pkl
# Output: user vectors (u_vec.pkl) and demographic info (users_info.pkl)

#open figs folder txt file to read the path
with open('./fig_folder.txt', 'r') as f:
    fig_folder = f.read().strip()

# -----------------------------
# Load data
# -----------------------------
with open("../data/raw/users.json", "r") as f:
    users = pd.DataFrame(json.load(f)).T
print(f"Total users loaded: {len(users)}")

votes = pd.read_pickle('../data/processed/pkl/votes.pkl').drop_duplicates()
votes['debate_title_lower'] = votes['debate_title'].str.lower()
voters = votes.voter_name.unique()


# Obtain user vectors
# 1. `user.json` file contains the opinions (before voting) of each user that expressed it.
# 2. Map Contrary to -1, Pro to +1, and the rest to 0.
# 3. Restrict to those who voted to the debates and have an opinion on at least one topic

# -----------------------------
# Check for duplicate users
# -----------------------------
if users.index.duplicated().any():
    print("Warning: duplicated user IDs found!")

# -----------------------------
# Expand big_issues_dict and compute issue counts
# -----------------------------
issue_data = users['big_issues_dict'].apply(pd.Series)
issue_counts = issue_data.apply(pd.Series.value_counts).fillna(0)
# Remove unwanted stance categories
issue_counts = issue_counts.drop(index=['N/S', 'N/O', 'Und'], errors='ignore').T
issue_counts['total'] = issue_counts.sum(axis=1)
issue_counts = issue_counts.sort_values('total', ascending=True).drop(columns='total')

# Lowercase everything
issue_counts.columns = [col.lower() for col in issue_counts.columns]
issue_counts.index = [idx.lower() for idx in issue_counts.index]

# Define colors for stances
stance_colors = {'pro': '#0072B2', 'con': '#D55E00'}
colors = [stance_colors[s] for s in issue_counts.columns]

# Plot topic/stance distribution
ax = issue_counts.plot(
    kind='barh',
    stacked=True,
    color=colors,
    figsize=(8, 7)
)
plt.xlabel("count (no. of users)")
plt.ylabel("topic")
plt.title("topic and stance distribution of user opinions")
plt.tight_layout()
plt.savefig('../plots/users_classification.pdf')
plt.savefig(f'{fig_folder}/users_classification.pdf')

# -----------------------------
# Create user vectors
# -----------------------------
user_opinions = issue_data.replace({'Con': -1, 'Und': 0, 'Pro': 1, 'N/O': 0, 'N/S': 0}).fillna(0)
print(f'Number of users: {len(user_opinions)}')
print(f'Number of big issues (topics): {user_opinions.shape[1]}')

# Restrict to voters with at least one opinion
user_opinions_red = user_opinions.loc[user_opinions.index.isin(voters)]
user_opinions_red = user_opinions_red[(user_opinions_red != 0).any(axis=1)]
print(f'Users after filtering: {len(user_opinions_red)} ({len(user_opinions_red)/len(user_opinions)*100:.2f}% of total)')

#save user vectors
user_opinions_red['u_vec'] = user_opinions_red.values.tolist()
user_opinions_red = user_opinions_red.reset_index().rename(columns={'index': 'username'})
user_opinions_red[['username', 'u_vec']].to_pickle('../data/processed/pkl/u_vec.pkl')
user_opinions_red[['username', 'u_vec']].to_csv('../data/processed/csv/u_vec.csv', index=False)

# -----------------------------
# Extract user info and age
# -----------------------------
users_info = users.loc[user_opinions_red.username, [
    'description', 'education', 'ethnicity', 'gender', 'president',
    'religious_ideology', 'party', 'political_ideology'
]]

def extract_age(description):
    '''
    input: user description text
    output: extracted age as integer or NaN
    '''
    if pd.isna(description):
        return np.nan
    patterns = [
        r'\bI am (\d{1,3})\b', r"\bI'm (\d{1,3})\b",
        r'\bage (\d{1,3})\b', r'(\d{1,3})-years? old\b',
        r'\b(\d{1,3}) years? old\b', r'\b(\d{1,3}) yo\b'
    ]
    for pattern in patterns:
        match = re.search(pattern, description)
        if match:
            age = int(match.group(1))
            if 0 < age < 90:
                return age
    return np.nan

users_info['age'] = users_info['description'].apply(extract_age).astype('Int64')
users_info.to_pickle('../data/processed/pkl/users_info.pkl')
users_info.to_csv('../data/processed/csv/users_info.csv', index=True)

# -----------------------------
# Plot demographic distributions
# -----------------------------
fig, axs = plt.subplots(4, 2, figsize=(8, 10))

demographic_vars = [
    'education', 'ethnicity',
    'gender', 'president',
    'religious_ideology', 'party',
    'political_ideology', 'age'
]

for i, var in enumerate(demographic_vars):
    ax = axs[i // 2, i % 2]
    
    if var == 'age':
        users_info['age'].plot(
            kind='hist',
            bins=range(0, 90, 5),
            ax=ax,
            color='#56B4E9'
        )
        ax.set_title('age')
        ax.set_xlabel('age')
        ax.set_ylabel('no. of users')
    else:
        counts = users_info[var].value_counts().nlargest(10).iloc[::-1]
        counts.index = (
            counts.index
            .str.replace(r'\s*\(.*?\)', '', regex=True)
            .str.replace(r'^Christian\s*-\s*(.+)', r'\1', regex=True)
            .str.replace(r'\s*Party\b', '', regex=True)
            .str.strip()
        )



        bar_colors = [
            'grey' if str(idx).lower() in ['not saying', 'prefer not to say']
            else '#56B4E9'
            for idx in counts.index
        ]
        counts.plot(kind='barh', ax=ax, color=bar_colors)
        ax.set_title(f'top 10 of {var.replace("_", " ")}')
        ax.set_xlabel('no. of users')
        ax.tick_params(axis='y', labelsize=8)
        #ax.set_ylabel(var.capitalize())
        ax.set_ylabel(var.replace('_', ' '))


plt.tight_layout()
plt.savefig('../plots/user_demographics.pdf')
plt.savefig(f'{fig_folder}/user_demographics.pdf')