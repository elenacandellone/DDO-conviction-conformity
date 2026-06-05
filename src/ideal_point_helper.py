def filter_user_debates(votes, user_vectors, debate_vectors, debate_classification, remove_both_sides_no_votes=True):
    prev_user_count, prev_debate_count = -1, -1

    # remove  neutral votes
    votes = votes[votes['vote'] != 0].reset_index(drop=True)

    while True:
        # Step 1: filter users
        users_with_opinions = set(user_vectors['username'].unique())
        users_voting = set(votes['voter_name'].unique())
        users_intersect = users_with_opinions & users_voting
        user_vectors = user_vectors[user_vectors['username'].isin(users_intersect)]
        votes = votes[votes['voter_name'].isin(users_intersect)]
        print(f"Number of users after filtering: {len(users_intersect)}")

        # Step 2: filter debates
        debates_with_vectors = set(debate_vectors.loc[debate_vectors['topic'] != 'other', 'debate_key'].unique())
        debates_votes = set(votes['debate_key'].unique())
        debates_intersect = debates_with_vectors & debates_votes
        debate_vectors = debate_vectors[debate_vectors['debate_key'].isin(debates_intersect)]
        votes = votes[votes['debate_key'].isin(debates_intersect)]
        debate_classification = debate_classification[debate_classification['debate_key'].isin(debates_intersect)]
        print(f"Number of debates after filtering: {len(debates_intersect)}")

        if remove_both_sides_no_votes:
            #keep debates with at least one -1, 0, 1 vote
            debate_vote_counts = votes.groupby(['debate_key', 'vote']).size().unstack(fill_value=0)
            debates_to_keep = debate_vote_counts[(debate_vote_counts.get(-1, 0) > 0) & 
                                                #(debate_vote_counts.get(0, 0) > 0) & 
                                                (debate_vote_counts.get(1, 0) > 0)].index
            debates_intersect = debates_intersect & set(debates_to_keep)
            debate_vectors = debate_vectors[debate_vectors['debate_key'].isin(debates_to_keep)]
            votes = votes[votes['debate_key'].isin(debates_to_keep)]
            debate_classification = debate_classification[debate_classification['debate_key'].isin(debates_to_keep)]
            print(f"Number of debates after vote type filtering: {len(debates_intersect)}")


        # Stop if counts stabilize
        if len(users_intersect) == prev_user_count and len(debates_intersect) == prev_debate_count :
            break
        prev_user_count, prev_debate_count = len(users_intersect), len(debates_intersect)


    print(f"Number of users after full intersection: {len(users_intersect)}")
    print(f"Number of debates after full intersection: {len(debates_intersect)}")
    return votes, user_vectors, debate_vectors, debate_classification


def string_to_vector(s):
    if isinstance(s, str):
        return np.fromstring(s.strip("[]"), sep=',')
    else:
        return s
    


# -------------------------------
# generate stan data
# input: votes dataframe
# output: stan data dictionary with
#         number of debates, voters, topics, votes
#         encoded indices for debates, topics
#         observed votes, phi, tau
#         mappings for topics, voters, debates
# -------------------------------
def stan_data_gen(votes_df_):
    votes_df = votes_df_.copy()

    N_debates = len(votes_df['debate_key'].unique())
    N_voters = len(votes_df['voter_name'].unique())
    N_topics = len(votes_df['topic'].unique())
    N_votes = votes_df.shape[0]
    
    print(f'Number of debates: {N_debates}')
    print(f'Number of voters: {N_voters}')
    print(f'Number of topics: {N_topics}')
    print(f'Number of votes: {N_votes}')
    
    #encode topics, voters, debates
    le_topic = LabelEncoder()
    votes_df['topic_idx'] = le_topic.fit_transform(votes_df['topic'])+1

    le_voters = LabelEncoder()
    votes_df['voter_idx'] = le_voters.fit_transform(votes_df['voter_name'])+1

    le_debate = LabelEncoder()
    votes_df['debate_idx'] = le_debate.fit_transform(votes_df['debate_key'])+1

    #save mappings
    topic_mapping = dict(zip(votes_df['topic_idx'], votes_df['topic']))
    voter_mapping =  dict(zip(votes_df['voter_idx'], votes_df['voter_name']))
    debate_mapping = dict(zip(votes_df['debate_idx'], votes_df['debate_key']))

    #sort topic mapping
    topic_mapping = dict(sorted(topic_mapping.items()))
    topics = list(topic_mapping.values())

    #votes (y=1 for PRO, 0 for CON)
    print(f'votes counts:\n{votes_df["vote"].value_counts()}')
    y_obs = votes_df['vote'].apply(lambda x: 1 if x == 1 else 0 if x == -1 else (_ for _ in ()).throw(ValueError("Invalid x"))).to_numpy()
    
    #topic id per debate (assuming each debate has one topic, take the first topic for each debate)
    topic_id_per_debate = (
    votes_df[['debate_idx', 'topic_idx']]
    .drop_duplicates('debate_idx')
    .sort_values('debate_idx')
    ['topic_idx']
    .tolist()
    )

    stan_data = {
        'N_votes': N_votes,
        'N_debates': N_debates,
        'N_topics': N_topics,
        'N_users': N_voters,
        'topic_id_debate': topic_id_per_debate,
        'debate_id': votes_df['debate_idx'].astype(int).tolist(),
        'user_id': votes_df['voter_idx'].astype(int).tolist(),
        'y': y_obs.astype(int).tolist(),
        'phi_obs': votes_df['phi'].tolist(),
        'tau_obs': votes_df['tau'].tolist()
    }
    return stan_data, debate_mapping, voter_mapping, topic_mapping, topics, votes_df




# -------------------------------
# Run Stan models and save results
# -------------------------------
def run_model(stan_file, stan_data, output_dir, results_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/{model_name}", exist_ok=True)
    os.makedirs(f"{results_dir}/{model_name}", exist_ok=True)

    if os.path.exists(f"{output_dir}/{model_name}_fit.pkl"):
        print(f"{model_name} fit already exists. Loading from file.")
        with open(f"{output_dir}/{model_name}_fit.pkl", "rb") as f:
            fit = pickle.load(f)
    else:
        model = CmdStanModel(stan_file=stan_file, force_compile=True)
        fit = model.sample(
            data=stan_data,
            seed=123,
            chains=4,
            parallel_chains=4,
            iter_warmup=1000,
            iter_sampling=1000,
            output_dir=output_dir,
            show_console=True
        )
        with open(f"{output_dir}/{model_name}_fit.pkl", "wb") as f:
            pickle.dump(fit, f)

    summary_path = f"{results_dir}/summary.csv"
    if not os.path.exists(summary_path):
        summary_df = fit.summary()
        summary_df.to_csv(summary_path)
    else:
        print(f"{model_name} summary already exists. Skipping summary save.")

    diagnose_path = f"{results_dir}/diagnose.txt"
    if not os.path.exists(diagnose_path):
        with open(diagnose_path, "w") as f:
            f.write(fit.diagnose())
        print(f"{model_name} fit complete.")
    else:
        print(f"{model_name} diagnose already exists. Skipping diagnose save.")

    return fit