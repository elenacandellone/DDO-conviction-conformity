import sys
sys.path.append('../src') 
from imports import *

# 1. Data Cleaning
# Input: debates.json, users.json
# Output: cleaned DataFrames for debate votes (votes.pkl) and debate information (debate_info.pkl)

def process_debate_votes(debate_data_):
    '''
    Process debate votes and create a DataFrame with the following columns:
    - debate_key: unique identifier for the debate
    - debate_title: title of the debate
    - debater: name of the debater (participant or voter)
    - pro_con: stance of the debater (Pro, Con, Tied)
    - voter_name: name of the voter (if debater is a participant, this will be the same as debater)
    - vote: numerical representation of the vote (1 for Pro, -1 for Con, 0 for Tied)
    - vote_time: order of the vote (0 for debaters, incrementing for each vote by the voter)
    - is_debater: 1 if the debater is a participant, 0 if the debater is a voter
    '''
    debate_data = debate_data_.copy()
    #collect data in a list
    votes_data = []
    print("Processing debate votes...")
    print(f"Total debates votes to process: {len(debate_data)}")

    for debate_key, debate_info in tqdm(debate_data.items()):
        # title, participants, stances (pro/con)
        debate_title = debate_info['title']
        participant_1, participant_1_stance = debate_info['participant_1_name'], debate_info['participant_1_position']
        participant_2, participant_2_stance = debate_info['participant_2_name'], debate_info['participant_2_position']
        
        # Convert stances into numerical votes
        vote_1 = 1 if participant_1_stance == 'Pro' else -1 if participant_1_stance == 'Con' else 0
        vote_2 = 1 if participant_2_stance == 'Pro' else -1 if participant_2_stance == 'Con' else 0
        
        #Print zero votes for debugging
        if vote_1 == 0:
            print(f"Zero vote for {participant_1} in debate {debate_key}")
        if vote_2 == 0:
            print(f"Zero vote for {participant_2} in debate {debate_key}")

        #add participant stances to a dictionary for easy lookup
        part_dict = {participant_1: participant_1_stance, participant_2: participant_2_stance}

        #start vote time at 0 for debaters and increment for each vote by the voter
        vote_time = 0
        is_debater = 1

        # Include debate participants' votes in the votes_data
        votes_data.append([debate_key, debate_title, participant_1, participant_1_stance, participant_1, vote_1, vote_time, is_debater])
        votes_data.append([debate_key, debate_title, participant_2, participant_2_stance, participant_2, vote_2, vote_time, is_debater])
        
        # Process audience votes
        for vote in debate_info['votes']:
            voter_name = vote['user_name']
            for candidate, vote_info in vote['votes_map'].items():
                #skip votes by debaters to avoid double counting
                if voter_name == participant_1 or voter_name == participant_2:
                    continue
                
                #get the stance of the candidate (debater) from the part_dict
                stance = part_dict.get(candidate, 'Tied')  # Default to 'Tied' if debater not found

                #get if the voter agreed with the candidate after the debate
                points = vote_info.get('Agreed with after the debate', np.nan)
                vote_value = 1 if points is True else -1 if points is False else 0
                #low vote time is more recent, so we increment it for each vote by the voter
                vote_time += 1 # Increment vote time for each vote by the voter
                is_debater = 0 if voter_name != participant_1 and voter_name != participant_2 else 1
                if vote_value > 0:
                    votes_data.append([debate_key, debate_title, candidate, stance, voter_name, vote_value, vote_time, is_debater])
    
    # Create DataFrame
    df = pd.DataFrame(votes_data, columns=['debate_key', 'debate_title', 'debater', 'pro_con', 'voter_name', 'vote', 'vote_time', 'is_debater'])

    
    #order by debate_key and vote_time (higher time are older votes) and is_debater (debater votes first)
    df = (
    df
    .sort_values(by=['debate_key', 'is_debater', 'vote_time'],
                 ascending=[True, False, False])
    .reset_index(drop=True)
    )

    #older votes have higher vote_time, so we rank them in descending order to get the order of voting (1 for oldest, incrementing for newer votes)        
    mask = df['vote_time'] != 0
    df.loc[mask, 'vote_time'] = (
        df[mask]
        .groupby('debate_key')['vote_time']
        .rank(ascending=False, method='dense')
        .astype(int)
    )
    #map pro_con to vote: Pro = 1, Con = -1, Tied = 0
    df['vote'] = df['pro_con'].map({'Pro': 1, 'Con': -1, 'Tied': 0})
    return df


def process_debate_info(debate_data_):
    '''
    Process debate information and create a DataFrame with the following columns:
    - debate_key: unique identifier for the debate
    - debate_title: title of the debate
    - debate_category: category of the debate
    - debate_date: start date of the debate
    - no_comments: number of comments in the debate
    - no_views: number of views of the debate
    - no_rounds: number of rounds in the debate
    - no_votes: number of votes in the debate
    - participant_1: name of the first participant
    - participant_1_stance: stance of the first participant (Pro, Con, Tied)
    - participant_1_rounds: text of the rounds for the first participant
    - participant_2: name of the second participant
    - participant_2_stance: stance of the second participant (Pro, Con, Tied)
    - participant_2_rounds: text of the rounds for the second participant
    '''
    debate_data = debate_data_.copy()
    debate_list = []
    print("Processing debate information...")
    for debate_key, debate_info in tqdm(debate_data.items()):
        # title, category, date, no_of_comments, no_of_views, no_of_rounds, no_of_votes
        debate_title = debate_info['title']
        debate_category = debate_info['category']
        debate_date = pd.to_datetime(debate_info['start_date'])
        no_comments = debate_info['number_of_comments']
        no_views = debate_info['number_of_views']
        no_rounds = debate_info['number_of_rounds']
        no_votes = debate_info['number_of_votes']

        #remove the word times from no_of_views and make it an integer
        if isinstance(no_views, str):
            no_views = int(no_views.replace(' times', '').replace(',', ''))
            no_views = int(no_views)

        #participant stances (pro/con)
        participant_1, participant_1_stance = debate_info['participant_1_name'], debate_info['participant_1_position']
        participant_2, participant_2_stance = debate_info['participant_2_name'], debate_info['participant_2_position']
        participant_1_rounds = ""
        participant_2_rounds = ""

        #add pro and con rounds text (debate motivation)
        pro_rounds_count = 0
        con_rounds_count = 0
        for round_info in debate_info['rounds']:
            while pro_rounds_count >=1 and con_rounds_count >=1:
                break
            for stance in round_info:
                if stance['side'] == 'Pro' and pro_rounds_count < 1:
                    participant_1_rounds += stance['text']
                    pro_rounds_count += 1
                elif stance['side'] == 'Con' and con_rounds_count < 1:
                    participant_2_rounds += stance['text']
                    con_rounds_count += 1

        #text cleaning: remove new lines and extra spaces
        participant_1_rounds = ' '.join(participant_1_rounds.replace('\n', ' ').split())
        participant_2_rounds = ' '.join(participant_2_rounds.replace('\n', ' ').split())
        participant_1_rounds = ' '.join(participant_1_rounds.replace('\r', ' ').split())
        participant_2_rounds = ' '.join(participant_2_rounds.replace('\r', ' ').split())
        
        debate_list.append([debate_key, debate_title, debate_category, debate_date, no_comments, no_views, no_rounds, no_votes,
                            participant_1, participant_1_stance, participant_1_rounds,
                            participant_2, participant_2_stance, participant_2_rounds])
    
    df_debates = pd.DataFrame(debate_list, columns=[
        'debate_key', 'debate_title', 'debate_category', 'debate_date', 'no_comments', 'no_views', 'no_rounds', 'no_votes',
        'participant_1', 'participant_1_stance', 'participant_1_rounds',
        'participant_2', 'participant_2_stance', 'participant_2_rounds'
    ])
    return df_debates


### MAIN


if __name__ == "__main__":
    #read debates and users json files
    with open("../data/raw/debates.json", "r") as f:
        debates = json.load(f)
    with open("../data/raw/users.json", "r") as f:
        users = json.load(f)
    
    #make a df
    users = pd.DataFrame(users).T

    #get list of topics based on big_issues_dict keys
    topics = list(users['big_issues_dict'].apply(pd.Series).columns)
    print(f"Total big issues topics: {len(topics)}")
    #save topics
    if not os.path.exists('../data/processed/txt/'):
        os.makedirs('../data/processed/txt/')
    with open('../data/processed/txt/big_issues_topics.txt', 'w') as f:
        for topic in topics:
            f.write(f"{topic}\n")

    # DEBATE VOTES
    df_votes = process_debate_votes(debates)

    print(f"Total votes processed: {len(df_votes)}")
    print(f"Unique debates: {df_votes['debate_key'].nunique()}")
    print(f"Unique voters: {df_votes['voter_name'].nunique()}")

    if not os.path.exists('../data/processed/pkl/'):
        os.makedirs('../data/processed/pkl/')
    if not os.path.exists('../data/processed/csv/'):
        os.makedirs('../data/processed/csv/')
    
    df_votes.to_pickle('../data/processed/pkl/votes.pkl')
    df_votes.to_csv('../data/processed/csv/votes.csv', index=False)

    # DEBATE INFO
    df_debate_info = process_debate_info(debates)
    print(f"Total debates info processed: {len(df_debate_info)}")
    df_debate_info.to_pickle('../data/processed/pkl/debate_info.pkl')
    df_debate_info.to_csv('../data/processed/csv/debate_info.csv', index=False)

