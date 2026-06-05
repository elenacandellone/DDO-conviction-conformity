import sys

sys.path.append('../src') 
from imports import *
from ideal_point_helper import *
np.random.seed(42)

#open figs folder txt file to read the path
with open('./fig_folder.txt', 'r') as f:
    fig_folder = f.read().strip()


# -------------------------------
# TOPIC ALIGNMENT
# compute tau
# cosine similarity between u_vec and d_vec
# -------------------------------
def compute_tau(votes_df_):
    votes_df = votes_df_.copy()
    # user and debate vectors
    U = np.stack(votes_df['u_vec'].values)
    D = np.stack(votes_df['d_vec'].values)

    # normalize
    U_norm = U / np.linalg.norm(U, axis=1, keepdims=True)
    D_norm = D / np.linalg.norm(D, axis=1, keepdims=True)

    # compute tau
    votes_df['tau'] = np.einsum('ij,ij->i', U_norm, D_norm)
    return votes_df

# -------------------------------
# PEER INFLUENCE
# compute phi time-dependent
# sum over previous voters in debate k of cosine similarity u_i, u_k * vote_kj
# divided by number of previous voters
# -------------------------------
def compute_phi_time_dep(votes_df_):
    votes_df = votes_df_.copy()

    # user vectors matrix
    u_vecs = (
        votes_df
        .groupby("voter_name")["u_vec"]
        .first()
    )
    user_index = u_vecs.index
    # matrix of user vectors with right ordering
    u_mat = np.stack(u_vecs.loc[user_index])

    # cosine similarity matrix between users
    cos_sim = pd.DataFrame(
        cosine_similarity(u_mat),
        index=user_index,
        columns=user_index
    )
    #clip values to avoid numerical issues
    cos_sim = cos_sim.clip(-1.0, 1.0)
    
    #initialize
    phi = np.zeros(len(votes_df))
    
    # Iterate over debates
    for j, df_j in tqdm(votes_df.groupby("debate_key"), desc="Computing phi (time-dependent)"):

        # Voters, votes, and times
        voters = df_j["voter_name"].to_numpy()
        #sgn_t  = df_j["sgn_tau"].to_numpy()
        votes = df_j["vote"].to_numpy()
        times = df_j["vote_time"].to_numpy()

        #for each vote in debate j
        for idx, time in enumerate(times):
            #take voters to that debate
            voter = voters[idx]
            #if time is 0, skip (skip debaters)
            if time == 0:
                continue

            # All voters at strictly earlier times + same time (excluding self)
            mask = (times <= time) & (voters != voter)
            prev_voters = voters[mask]
            prev_votes = votes[mask]

            #get cosine similarities
            weights = cos_sim.loc[voter, prev_voters].to_numpy()
            #get the index based on debate_key and voter_name
            phi_idx = votes_df[
                (votes_df["debate_key"] == j) & 
                (votes_df["voter_name"] == voter)
            ].index[0]

            # compute phi as weighted (by cos sim) average of previous votes
            phi[phi_idx] = np.sum(weights * prev_votes) / len(prev_voters) if len(prev_voters) > 0 else 0

    votes_df["phi"] = phi
    return votes_df



# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":

    os.makedirs("../results/regression/bert", exist_ok=True)
    os.makedirs("../results/latent/bert", exist_ok=True)
    path_data = "../data/processed/pkl/"
    votes = pd.read_pickle(f"{path_data}votes.pkl")
    user_vectors = pd.read_pickle(f"{path_data}u_vec.pkl")
    debate_vectors = pd.read_pickle(f"{path_data}d_vec_bert.pkl")
    debate_classification = pd.read_pickle(f"{path_data}debate_classification_bert.pkl")

    #filter out debates with no votes on both sides (PRO and CON) and corresponding user and debate vectors
    votes, user_vectors, debate_vectors, debate_classification = filter_user_debates(
        votes, user_vectors, debate_vectors, debate_classification,
        remove_both_sides_no_votes=True
    )
    
    # Merge votes with user and debate vectors
    votes_full = pd.merge(votes[['debate_key','voter_name','vote_time','vote']],
                          debate_vectors[['debate_key','d_vec','topic']],
                          on='debate_key')
    votes_full = pd.merge(votes_full,
                          user_vectors[['username','u_vec']],
                          left_on='voter_name', right_on='username').drop(columns=['username'])
    votes_full['d_vec'] = votes_full['d_vec'].apply(string_to_vector)
    votes_full['u_vec'] = votes_full['u_vec'].apply(string_to_vector)

    # compute phi and tau, and save to file
    if os.path.exists(f"{path_data}/bert/votes_full_bert.pkl"):
        print("votes_full with phi and tau already exists. Loading from file.")
        votes_full = pd.read_pickle(f"{path_data}/bert/votes_full_bert.pkl")
    else:
        votes_full = compute_tau(votes_full)
        votes_full = compute_phi_time_dep(votes_full)
        votes_full.to_pickle(f"{path_data}votes_full_bert.pkl")

    #plot phi and tau distribution as subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(votes_full['phi'], bins=30, kde=True, ax=axs[0], color='skyblue')
    axs[0].set_xlim(-1, 1)
    #axs[0].set_title('Distribution of peer influence')
    axs[0].set_xlabel(r'peer influence ($\varphi$)')
    axs[0].set_ylabel('count')
    sns.histplot(votes_full['tau'], bins=30, kde=True, ax=axs[1], color='salmon')
    #axs[1].set_title('Distribution of topic alignment')
    axs[1].set_xlabel(r'topic alignment ($\tau$)')
    axs[1].set_xlim(-1, 1)
    axs[1].set_ylabel('count')
    plt.tight_layout()
    plt.savefig(f'{fig_folder}/phi_tau_distribution_bert.pdf')
    plt.savefig(f'../plots/phi_tau_distribution.pdf')

    #generate stan data and save to file
    stan_data, debate_mapping, voter_mapping, topic_mapping, topics, votes_full = stan_data_gen(votes_full)
    with open(f"../results/regression/bert/stan_data.pkl", "wb") as f:
        pickle.dump(stan_data, f)
    with open(f"../results/regression/bert/debate_mapping.pkl", "wb") as f:
        pickle.dump(debate_mapping, f)
    with open(f"../results/regression/bert/voter_mapping.pkl", "wb") as f:
        pickle.dump(voter_mapping, f)
    with open(f"../results/regression/bert/topic_mapping.pkl", "wb") as f:
        pickle.dump(topic_mapping, f)

    # Run regression model (observed phi/tau)
    fit_regression = run_model(
        stan_file="model.stan",
        stan_data=stan_data,
        output_dir="../results/stan_output/regression/bert/",
        results_dir="../results/regression/bert",
        model_name="model_regression"
    )
    print("All models run successfully.")