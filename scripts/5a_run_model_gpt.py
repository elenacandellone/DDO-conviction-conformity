import sys
sys.path.append('../src') 
from imports import *
from ideal_point_helper import *
np.random.seed(42)

# 5. Run Model
# Input: votes_full.pkl with computed tau (conviction), c (persuasiveness), and D (cross-cutting exposure)
# Output: fitted models and diagnostics with STANDARDIZED predictors (tau/c/D z-scored before Stan) 

#open figs folder txt file to read the path
with open('./fig_folder.txt', 'r') as f:
    fig_folder = f.read().strip()


# -------------------------------
# CONVICTION / TOPIC ALIGNMENT
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
# CROSS-CUTTING EXPOSURE
# compute D (audience diversity / cross-cutting exposure) time-dependent
# negative mean pairwise cosine similarity among prior voters in debate k
# (D in [-1,1]: -1 = fully homogeneous prior audience, +1 = fully opposed)
#
# it is a property of the prior-voter set V_{j,i} alone, so it is not 
# confounded by construction with tau (which does use u_i). Known limitation 
# (see the diversity-measure simulation): mean pairwise dissimilarity cannot 
# distinguish a large audience split into two internally-homogeneous opposing 
# camps from a large audience with genuinely dispersed, uncorrelated beliefs. 
# both converge toward D~0 as the prior-audience size grows, since same-camp 
# and cross-camp pairs contribute oppositely to the average and their counts scale together.
# -------------------------------
def compute_D_time_dep(votes_df_):
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
    D = np.zeros(len(votes_df))
    # has_prior_D: 1 where D[i] is a genuinely observed pairwise-diversity
    # value, 0 where no prior audience exists to compute it from (debaters,
    # who never reach this loop body at all since time==0 is skipped below,
    # and n<=1 voters). D itself stays 0 for those rows as a placeholder, but
    # it is never meant to be read on its own -- see the STRUCTURAL D_obs
    # HANDLING note in stan_data_gen: beta_D's slope is only applied where
    # has_prior_D==1, so this flag (not the D=0 placeholder value) is what
    # actually distinguishes "no prior audience" from "prior audience with a
    # measured diversity near 0". 
    has_prior_D = np.zeros(len(votes_df), dtype=int)

    # Iterate over debates
    for j, df_j in tqdm(votes_df.groupby("debate_key"), desc="Computing D (cross-cutting exposure)"):

        # Voters and times
        voters = df_j["voter_name"].to_numpy()
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
            prior_voters = voters[mask]
            n = len(prior_voters)

            #get the index based on debate_key and voter_name
            D_idx = votes_df[
                (votes_df["debate_key"] == j) &
                (votes_df["voter_name"] == voter)
            ].index[0]

            if n <= 1:
                # no meaningful pairwise diversity with zero or one prior voter
                D[D_idx] = 0
                continue

            # mean pairwise cosine similarity among prior voters only, negated so
            # that D > 0 = more diverse (dissimilar) prior audience, D < 0 = more
            # ideologically homogeneous prior audience
            sub = cos_sim.loc[prior_voters, prior_voters].to_numpy()
            upper = sub[np.triu_indices(n, k=1)]
            D[D_idx] = -np.mean(upper)
            has_prior_D[D_idx] = 1

    votes_df["D"] = D
    votes_df["has_prior_D"] = has_prior_D
    return votes_df



# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":

    os.makedirs("../results/regression", exist_ok=True)
    path_data = "../data/processed/pkl/"
    votes = pd.read_pickle(f"{path_data}votes.pkl")
    user_vectors = pd.read_pickle(f"{path_data}u_vec.pkl")
    debate_vectors = pd.read_pickle(f"{path_data}d_vec_gpt.pkl")
    debate_classification = pd.read_pickle(f"{path_data}debate_classification_gpt.pkl")

    #filter out debates with no votes on both sides (PRO and CON) and corresponding user and debate vectors
    votes, user_vectors, debate_vectors, debate_classification = filter_user_debates(
        votes, user_vectors, debate_vectors, debate_classification,
        remove_both_sides_no_votes=True
    )
    
    # Merge votes with user and debate vectors
    votes_full = pd.merge(votes[['debate_key','voter_name','vote_time','vote','persuasive']],
                          debate_vectors[['debate_key','d_vec','topic']],
                          on='debate_key')
    votes_full = pd.merge(votes_full,
                          user_vectors[['username','u_vec']],
                          left_on='voter_name', right_on='username').drop(columns=['username'])
    votes_full['d_vec'] = votes_full['d_vec'].apply(string_to_vector)
    votes_full['u_vec'] = votes_full['u_vec'].apply(string_to_vector)

    # persuasiveness: rename to c (matching the tau/D naming convention), and fill
    # debaters' undefined signal (NaN, since a debater doesn't credit anyone for their
    # own declared stance) with 0 -- same "no information" convention used for D at
    # vote_time==0.
    votes_full = votes_full.rename(columns={'persuasive': 'c'})
    votes_full['c'] = votes_full['c'].fillna(0)

    # compute tau, c, and D, and save to file
    if os.path.exists(f"{path_data}votes_full.pkl"):
        print("votes_full with tau, c and D already exists. Loading from file.")
        votes_full = pd.read_pickle(f"{path_data}votes_full.pkl")
    else:
        votes_full = compute_tau(votes_full)
        votes_full = compute_D_time_dep(votes_full)
        votes_full.to_pickle(f"{path_data}votes_full.pkl")

    #plot tau, c, and D distributions as subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    sns.histplot(votes_full['tau'], bins=30, kde=True, ax=axs[0], color='salmon')
    axs[0].set_xlabel(r'conviction / topic alignment ($\tau$)')
    axs[0].set_xlim(-1, 1)
    axs[0].set_ylabel('count')
    sns.histplot(votes_full['c'], bins=30, kde=True, ax=axs[1], color='#e6af2e')
    axs[1].set_xlabel(r'persuasiveness ($c$)')
    axs[1].set_xlim(-1, 1)
    axs[1].set_ylabel('count')
    sns.histplot(votes_full['D'], bins=30, kde=True, ax=axs[2], color='skyblue')
    axs[2].set_xlabel(r'cross-cutting exposure ($D$)')
    axs[2].set_xlim(-1, 1)
    axs[2].set_ylabel('count')
    plt.tight_layout()
    plt.savefig('../plots/tau_c_D_distribution.pdf')

    # -------------------------------------------------------------------
    # STANDARDIZED-PREDICTOR VERSION -- the only battery run (tau/c/D are
    # z-scored before Stan; see stan_data_gen's docstring for why). No
    # separate raw-scale fits or "_std"-suffixed paths anymore.
    # -------------------------------------------------------------------
    os.makedirs("../results/regression", exist_ok=True)

    stan_data, debate_mapping, voter_mapping, topic_mapping, topics, votes_full, standardization = \
        stan_data_gen(votes_full, standardize_predictors=True)

    print("Standardization constants used:")
    for k, v in standardization.items():
        print(f"  {k}: mean={v['mean']:.4f}, sd={v['sd']:.4f}")

    with open("../results/regression/stan_data.pkl", "wb") as f:
        pickle.dump(stan_data, f)
    with open("../results/regression/standardization.pkl", "wb") as f:
        pickle.dump(standardization, f)
    with open("../results/regression/debate_mapping.pkl", "wb") as f:
        pickle.dump(debate_mapping, f)
    with open("../results/regression/voter_mapping.pkl", "wb") as f:
        pickle.dump(voter_mapping, f)
    with open("../results/regression/topic_mapping.pkl", "wb") as f:
        pickle.dump(topic_mapping, f)

    fit_regression = run_model(
        stan_file="model.stan",
        stan_data=stan_data,
        output_dir="../results/stan_output/regression/gpt",
        results_dir="../results/regression/gpt",
        model_name="model_regression"
    )

    stan_data_D_only = stan_data.copy()
    stan_data_D_only.pop('tau_obs')
    stan_data_D_only.pop('c_obs')
    fit_D_only = run_model(
        stan_file="model_D.stan",
        stan_data=stan_data_D_only,
        output_dir="../results/stan_output/D/gpt",
        results_dir="../results/D/gpt",
        model_name="model_D"
    )

    stan_data_tau_only = stan_data.copy()
    stan_data_tau_only.pop('c_obs')
    stan_data_tau_only.pop('D_obs')
    fit_tau_only = run_model(
        stan_file="model_tau.stan",
        stan_data=stan_data_tau_only,
        output_dir="../results/stan_output/tau/gpt",
        results_dir="../results/tau/gpt",
        model_name="model_tau"
    )

    stan_data_c_only = stan_data.copy()
    stan_data_c_only.pop('tau_obs')
    stan_data_c_only.pop('D_obs')
    fit_c_only = run_model(
        stan_file="model_c.stan",
        stan_data=stan_data_c_only,
        output_dir="../results/stan_output/c/gpt",
        results_dir="../results/c/gpt",
        model_name="model_c"
    )

    # kept regardless of predictive performance -- tests a specific
    # theoretical hypothesis (Mutz-style moderation) rather than competing
    # for LOO weight; see notebook a_model_comparison_classification.ipynb
    # for why its posteriors should be read with caution even so.
    fit_interaction = run_model(
        stan_file="model_interaction.stan",
        stan_data=stan_data,
        output_dir="../results/stan_output/interaction/gpt",
        results_dir="../results/interaction/gpt",
        model_name="model_interaction"
    )

    print("All models run successfully.")