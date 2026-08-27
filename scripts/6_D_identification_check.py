import sys
sys.path.append('../src')
from imports import *
from ideal_point_helper import *
np.random.seed(42)

# 6. Cross-cutting exposure (D) identification check
#
# D is NOT confounded by construction with tau: D is a
# property of the prior-voter set V_{j,i} alone and never touches the focal
# user's own belief vector u_i. But D can still be confounded by SELECTION:
# users with certain beliefs might select into debates that already have a
# more homogeneous or more diverse audience (e.g. avoiding a debate that
# already looks like a landslide), which would make beta_D reflect who chose
# to show up rather than any effect of the audience itself.
#
# This script runs a temporal-placebo analysis to compute D from votes cast
# AFTER the focal user's own vote ("future D"). Future votes cannot causally
# or compositionally precede a past vote, so if beta_D estimated on future D
# is comparable in magnitude to beta_D on the real (past) D, that's evidence
# consistent with selection into debate audiences rather than a genuine
# association between the pre-existing audience and how a user votes.
#
# Input: votes_full.pkl
# Output: D_future column cached in votes_placebo.pkl; comparison table 


# -------------------------------
# CROSS-CUTTING EXPOSURE (FUTURE / PLACEBO)
# same as compute_D_time_dep, but the prior-voter mask is flipped: average
# over voters who voted at or AFTER the focal user's own vote (excluding
# self), instead of at or before.
# -------------------------------
def compute_D_future_time_dep(votes_df_):
    votes_df = votes_df_.copy()

    u_vecs = (
        votes_df
        .groupby("voter_name")["u_vec"]
        .first()
    )
    user_index = u_vecs.index
    u_mat = np.stack(u_vecs.loc[user_index])

    cos_sim = pd.DataFrame(
        cosine_similarity(u_mat),
        index=user_index,
        columns=user_index
    )
    cos_sim = cos_sim.clip(-1.0, 1.0)

    D_future = np.zeros(len(votes_df))
    # has_prior_D_future: same "genuinely observed" flag as has_prior_D in
    # compute_D_time_dep (5a_run_model_gpt.py), mirrored for the future/
    # placebo audience. Needed for the same reason: stan_data_gen's
    # has_prior_D gates beta_D on real data only, and this placebo run must
    # supply that flag too (renamed to has_prior_D below, same as D_future
    # -> D) or it would silently fall back to whatever has_prior_D happens
    # to already be sitting in votes_full from the real/past D computation.
    has_prior_D_future = np.zeros(len(votes_df), dtype=int)

    for j, df_j in tqdm(votes_df.groupby("debate_key"), desc="Computing D (future / placebo)"):
        voters = df_j["voter_name"].to_numpy()
        times = df_j["vote_time"].to_numpy()

        for idx, time in enumerate(times):
            voter = voters[idx]
            if time == 0:
                continue

            # All voters at strictly later times + same time (excluding self) --
            # mirrors the "<=" tie handling used for the real (past) D
            mask = (times >= time) & (voters != voter)
            fut_voters = voters[mask]
            n = len(fut_voters)

            D_idx = votes_df[
                (votes_df["debate_key"] == j) &
                (votes_df["voter_name"] == voter)
            ].index[0]

            if n <= 1:
                D_future[D_idx] = 0
                continue

            sub = cos_sim.loc[fut_voters, fut_voters].to_numpy()
            upper = sub[np.triu_indices(n, k=1)]
            D_future[D_idx] = -np.mean(upper)
            has_prior_D_future[D_idx] = 1

    votes_df["D_future"] = D_future
    votes_df["has_prior_D_future"] = has_prior_D_future
    return votes_df


# -------------------------------
# POSTERIOR CI ON THE DIFFERENCE: real beta_D - future (placebo) beta_D
#
# Comparing the two 90% CIs separately is NOT a test of whether the
# difference is credibly nonzero. Need the posterior of
# (beta_D_real - beta_D_future) directly, via paired draws (same iteration
# index) from both fits, differenced per topic.
# -------------------------------
def compute_diff_ci(fit_real, fit_future, topic_mapping, ci=0.90):
    '''
    For each topic, compute the posterior of (beta_D_real - beta_D_future)
    from paired draws, and report its mean and credible interval. Requires
    both models to have been fit with the same number of draws (same
    warmup/sampling config), since pairing is by draw index, not by matching
    on summary stats.
    '''
    draws_real = fit_real.draws_pd()
    draws_future = fit_future.draws_pd()

    def sort_key(col):
        # handles 'beta_D[7]' naming from draws_pd(); avoids the string-sort
        # bug where 'beta_D[10]' would otherwise sort before 'beta_D[2]'
        return int(col.split('[')[1].rstrip(']'))

    D_cols_real = sorted(
        [c for c in draws_real.columns if c.startswith('beta_D[')], key=sort_key
    )
    D_cols_future = sorted(
        [c for c in draws_future.columns if c.startswith('beta_D[')], key=sort_key
    )

    if len(draws_real) != len(draws_future):
        raise ValueError(
            f"Draw count mismatch: real has {len(draws_real)}, "
            f"future has {len(draws_future)}. Paired differencing requires "
            "equal draw counts from equivalent sampler configs."
        )
    if len(D_cols_real) != len(D_cols_future):
        raise ValueError(
            f"Topic count mismatch: real has {len(D_cols_real)} beta_D "
            f"columns, future has {len(D_cols_future)}. Check both models "
            "were fit on the same topic set."
        )

    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    lo_pct, hi_pct = round(lo_q * 100), round(hi_q * 100)

    rows = []
    for col_r, col_f in zip(D_cols_real, D_cols_future):
        topic_idx = sort_key(col_r)
        topic = topic_mapping.get(topic_idx, topic_idx)

        diff_draws = draws_real[col_r].to_numpy() - draws_future[col_f].to_numpy()
        rows.append({
            'topic': topic,
            'diff_mean': diff_draws.mean(),
            f'diff_{lo_pct}%': np.quantile(diff_draws, lo_q),
            f'diff_{hi_pct}%': np.quantile(diff_draws, hi_q),
            'diff_excludes_zero': (
                np.quantile(diff_draws, lo_q) > 0 or np.quantile(diff_draws, hi_q) < 0
            ),
        })
    return pd.DataFrame(rows).set_index('topic')


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":

    os.makedirs("../results/D_placebo/gpt", exist_ok=True)
    path_data = "../data/processed/pkl/"

    # load votes_full with real D/c/tau already attached (see 5a_run_model_gpt.py)
    votes_full = pd.read_pickle(f"{path_data}votes_full.pkl")

    # compute future D, cache separately so the real votes_full.pkl is never
    # overwritten
    if os.path.exists(f"{path_data}votes_placebo.pkl"):
        print("votes_full with D_future already exists. Loading from file.")
        votes_full = pd.read_pickle(f"{path_data}votes_placebo.pkl")
    else:
        votes_full = compute_D_future_time_dep(votes_full)
        votes_full.to_pickle(f"{path_data}votes_placebo.pkl")

    # sanity check: compare distributions of real D vs future D
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(votes_full['D'], bins=30, kde=True, ax=axs[0], color='skyblue')
    axs[0].set_xlim(-1, 1)
    axs[0].set_xlabel(r'cross-cutting exposure, past audience ($D$)')
    axs[0].set_ylabel('count')
    sns.histplot(votes_full['D_future'], bins=30, kde=True, ax=axs[1], color='mediumpurple')
    axs[1].set_xlim(-1, 1)
    axs[1].set_xlabel(r'cross-cutting exposure, future audience ($D_{future}$)')
    axs[1].set_ylabel('count')
    plt.tight_layout()
    plt.savefig('../plots/FigA8.pdf')

    # generate stan data using D_future IN PLACE OF D (and its matching
    # has_prior_D_future IN PLACE OF has_prior_D), standardized the same way 5a standardizes the real
    # D (see module docstring: each side is z-scored against its own
    # mean/sd, so beta_D_real/beta_D_future stay comparable as per-SD effects)
    votes_placebo = (
        votes_full
        .drop(columns=['D', 'has_prior_D'])
        .rename(columns={'D_future': 'D', 'has_prior_D_future': 'has_prior_D'})
    )
    stan_data_future, debate_mapping, voter_mapping, topic_mapping, topics, votes_placebo, standardization_future = \
        stan_data_gen(votes_placebo, standardize_predictors=True)

    with open("../results/D_placebo/gpt/stan_data.pkl", "wb") as f:
        pickle.dump(stan_data_future, f)
    with open("../results/D_placebo/gpt/standardization.pkl", "wb") as f:
        pickle.dump(standardization_future, f)
    with open("../results/D_placebo/gpt/topic_mapping.pkl", "wb") as f:
        pickle.dump(topic_mapping, f)

    # D-only stan data (drop tau_obs/c_obs, matching model_D.stan's data block)
    stan_data_future_D_only = stan_data_future.copy()
    stan_data_future_D_only.pop('tau_obs')
    stan_data_future_D_only.pop('c_obs')

    # run the SAME model spec as the real D-only model, just with D built
    # from future votes instead of past votes. Uses model_D_placebo.stan --
    # a byte-identical copy of model_D.stan under its own filename -- rather
    # than model_D.stan directly, because run_model() always compiles with
    # force_compile=True: if this script runs while 5a_run_model_gpt.py is
    # still mid-fit on model_D.stan, both processes would force-recompile
    # the SAME binary (scripts/model_D) while 5a's chains are actively
    # running it. A private copy gives this script its own compile target,
    # so the two never touch the same binary regardless of timing.
    fit_D_future = run_model(
        stan_file="model_D_placebo.stan",
        stan_data=stan_data_future_D_only,
        output_dir="../results/stan_output/D_placebo/gpt",
        results_dir="../results/D_placebo/gpt",
        model_name="model_D_future"
    )
    print("Future/placebo D model run successfully.")

    # -------------------------------------------------------------
    # Load the real, standardized D-only fit (model_D, see the
    # "STANDARDIZED-PREDICTOR VERSION" section of 5a_run_model_gpt.py) for
    # comparison. run_model() caches to output_dir/results_dir and reloads
    # from file if results already exist there, so this call will NOT refit
    # if 5a_run_model_gpt.py has already produced it -- it only compiles
    # model_D_placebo.stan (same reasoning as above) in the case where it
    # hasn't, producing a numerically identical fit (same data, same
    # seed=123) without racing 5a's own model_D binary.
    # -------------------------------------------------------------
    with open("../results/regression/stan_data.pkl", "rb") as f:
        stan_data_real = pickle.load(f)
    stan_data_real_D_only = stan_data_real.copy()
    stan_data_real_D_only.pop('tau_obs')
    stan_data_real_D_only.pop('c_obs')

    fit_D_real = run_model(
        stan_file="model_D_placebo.stan",
        stan_data=stan_data_real_D_only,
        output_dir="../results/stan_output/D/gpt",
        results_dir="../results/D/gpt",
        model_name="model_D"
    )

    # -------------------------------------------------------------
    # comparison: real beta_D vs future (placebo) beta_D, by topic
    # -------------------------------------------------------------
    def betas_df(summary_df, topic_mapping):
        beta_D_df = summary_df.loc[summary_df.index.str.startswith('beta_D')].copy()
        beta_D_df['topic_idx'] = (
            beta_D_df.index.astype(str)
            .str.extract(r'\[\s*(\d+)\s*\]', expand=False)
            .astype('Int64')
        )
        beta_D_df['topic'] = beta_D_df['topic_idx'].map(topic_mapping)
        return beta_D_df[['topic', 'Mean', '5%', '95%']]

    try:
        summary_real = pd.read_csv("../results/D/gpt/summary.csv", index_col=0)
        summary_future = pd.read_csv("../results/D_placebo/gpt/summary.csv", index_col=0)
        # D's own results dir doesn't carry a topic_mapping.pkl (run_model()
        # only writes summary.csv); it shares the same topic set/encoding as the
        # rest of 5a's standardized battery, so reuse regression's mapping.
        topic_mapping_real = pd.read_pickle("../results/regression/topic_mapping.pkl")
        topic_mapping_future = pd.read_pickle("../results/D_placebo/gpt/topic_mapping.pkl")

        beta_D_real = betas_df(summary_real, topic_mapping_real).set_index('topic')
        beta_D_future = betas_df(summary_future, topic_mapping_future).set_index('topic')

        beta_D_real = beta_D_real.rename(columns={
            'Mean': 'beta_D_real', '5%': '5%_D_real', '95%': '95%_D_real'
        })
        beta_D_future = beta_D_future.rename(columns={
            'Mean': 'beta_D_future', '5%': '5%_D_future', '95%': '95%_D_future'
        })

        comparison = beta_D_real.merge(beta_D_future, left_index=True, right_index=True)
        r_spearman = comparison['beta_D_real'].corr(comparison['beta_D_future'], method='spearman')
        print(f"Spearman r (real vs. future beta_D): {r_spearman:.3f}")

        diff_ci_df = compute_diff_ci(fit_D_real, fit_D_future, topic_mapping_real)
        comparison = comparison.merge(diff_ci_df, left_index=True, right_index=True)

        n_credible = comparison['diff_excludes_zero'].sum()
        print(f"{n_credible} / {len(comparison)} topics have a 90% CI on "
              f"(real - future) that excludes zero.")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(comparison['beta_D_real'], comparison['beta_D_future'], alpha=0.7)
        lims = [
            min(comparison['beta_D_real'].min(), comparison['beta_D_future'].min()),
            max(comparison['beta_D_real'].max(), comparison['beta_D_future'].max()),
        ]
        ax.plot(lims, lims, linestyle='--', color='gray')
        ax.set_xlabel(r"$\beta_D$ (real, past audience, standardized)")
        ax.set_ylabel(r"$\beta_D$ (placebo, future audience, standardized)")
        ax.set_title(f"Spearman r = {r_spearman:.2f}")
        plt.tight_layout()
        plt.savefig("../plots/FigA9.pdf")

        comparison.to_csv("../results/D_placebo/gpt/D_real_vs_future.csv")

    except FileNotFoundError as e:
        print(
            f"Missing expected file: {e}. Check that both the real, standardized "
            "D-only model (via 5a_run_model_gpt.py's STANDARDIZED-PREDICTOR "
            "section) and this script's future/placebo D model have been run "
            "and their summary.csv / topic_mapping.pkl outputs saved."
        )
