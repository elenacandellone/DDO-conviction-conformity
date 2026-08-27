import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# 7. Diversity-measure simulation
#
# D (cross-cutting exposure, see compute_D_time_dep in 5a_run_model_gpt.py /
# 5b_run_model_bert.py) is mean pairwise cosine dissimilarity among prior
# voters' belief vectors, negated and left on the same [-1,1] scale as
# tau and c. It was chosen over several alternatives considered during model
# development (Vendi score / effective rank, mean nearest-neighbour distance,
# trace and generalized variance, Goodman-Kruskal concordance/discordance) for
# being the simplest, cheapest, and most standard measure in the
# network-heterogeneity literature (Mutz, 2006) that D is grounded in.
#
# It has one documented limitation, demonstrated here: mean-pairwise-average
# measures cannot distinguish a large audience split into two internally
# homogeneous opposing camps from a large audience with genuinely dispersed,
# uncorrelated beliefs -- both converge toward the same value as audience
# size grows, since same-camp and cross-camp pairs contribute oppositely to
# the average and their counts scale together at the same rate. This script
# reproduces that result under three controlled synthetic-audience scenarios,
# as documented justification for the caveat noted in compute_D_time_dep.
#
# Output: ../plots/FigA7.pdf

np.random.seed(42)

K = 48        # belief-vector dimensionality, matching the paper's 48 topics
N_TRIALS = 200        # Monte Carlo repeats per (scenario, n) for a distribution, not one draw
N_VALUES = [2, 3, 5, 8, 12, 20, 35, 50, 75, 100, 150, 200]


# -------------------------------------------------------------------
# D = -mean pairwise cosine similarity among an audience's belief vectors
# (audience-level version: computed over a full set of n voters at once,
# rather than incrementally per-voter as in compute_D_time_dep -- this
# isolates how the measure behaves as a function of n and audience
# composition, independent of the timestamp/sequential machinery used in
# the real pipeline.)
# -------------------------------------------------------------------
def compute_D(belief_vectors):
    n = len(belief_vectors)
    if n <= 1:
        return 0.0
    sim = cosine_similarity(belief_vectors)
    upper = sim[np.triu_indices(n, k=1)]
    return -np.mean(upper)


# -------------------------------------------------------------------
# SCENARIOS
#
# Each returns n belief vectors in {-1, 0, +1}^K, generated to represent a
# theoretically distinct audience composition. p_flip / p_zero control how
# much each voter deviates from a base pattern (p_zero mimics the real
# data's large share of "no opinion" entries per topic).
# -------------------------------------------------------------------
def homogeneous_audience(n, K=K, p_flip=0.05, p_zero=0.3):
    '''Audience clustered around one shared belief profile -- a self-selected,
    non-cross-cutting debate audience.'''
    base = np.random.choice([-1, 1], size=K)
    base[np.random.rand(K) < p_zero] = 0
    vecs = np.tile(base, (n, 1)).astype(float)
    flip_mask = np.random.rand(n, K) < p_flip
    vecs[flip_mask] *= -1
    return vecs


def polarized_audience(n, K=K, p_flip=0.05, p_zero=0.3):
    '''Two opposing camps, each internally homogeneous -- a genuinely
    contested, bimodal audience rather than a broad continuum.'''
    base = np.random.choice([-1, 1], size=K)
    base[np.random.rand(K) < p_zero] = 0
    anti_base = -base

    n_a = n // 2
    n_b = n - n_a
    camp_a = np.tile(base, (n_a, 1)).astype(float)
    camp_b = np.tile(anti_base, (n_b, 1)).astype(float)
    vecs = np.vstack([camp_a, camp_b])

    flip_mask = np.random.rand(n, K) < p_flip
    vecs[flip_mask] *= -1
    np.random.shuffle(vecs)
    return vecs


def diverse_uncorrelated_audience(n, K=K, p_zero=0.3):
    '''Each voter's beliefs drawn independently -- no shared structure at
    all, the "maximal cross-cutting exposure" null case.'''
    vecs = np.random.choice([-1, 1], size=(n, K)).astype(float)
    zero_mask = np.random.rand(n, K) < p_zero
    vecs[zero_mask] = 0
    return vecs


SCENARIOS = {
    "Homogeneous audience": homogeneous_audience,
    "Polarized (two camps)": polarized_audience,
    "Diverse / uncorrelated": diverse_uncorrelated_audience,
}


# -------------------------------------------------------------------
# RUN SIMULATION
# -------------------------------------------------------------------
if __name__ == "__main__":

    results = {name: {"mean": [], "std": []} for name in SCENARIOS}

    for name, gen_fn in SCENARIOS.items():
        for n in N_VALUES:
            trial_vals = [compute_D(gen_fn(n)) for _ in range(N_TRIALS)]
            results[name]["mean"].append(np.mean(trial_vals))
            results[name]["std"].append(np.std(trial_vals))

    # -------------------------------------------------------------------
    # PLOT
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {
        "Homogeneous audience": "tab:blue",
        "Polarized (two camps)": "tab:red",
        "Diverse / uncorrelated": "tab:green",
    }

    for name in SCENARIOS:
        means = np.array(results[name]["mean"])
        stds = np.array(results[name]["std"])
        ax.plot(N_VALUES, means, marker='o', label=name, color=colors[name])
        ax.fill_between(N_VALUES, means - stds, means + stds, alpha=0.15, color=colors[name])

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel("Audience size (n prior voters)")
    ax.set_ylabel(r"$D$ (cross-cutting exposure)")
    ax.set_title(
        "Simulated D by scenario and audience size\n"
        "(shaded band = ±1 SD across 200 Monte Carlo trials)"
    )
    ax.set_ylim(-1.05, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.savefig("../plots/FigA7.pdf")
    plt.show()

    # Print the key diagnostic: does the polarized scenario converge toward
    # the diverse scenario's value as n grows?
    polarized_at_max_n = results["Polarized (two camps)"]["mean"][-1]
    diverse_at_max_n = results["Diverse / uncorrelated"]["mean"][-1]
    print(
        f"At n={N_VALUES[-1]}: polarized D = {polarized_at_max_n:.3f}, "
        f"diverse D = {diverse_at_max_n:.3f} "
        f"(gap = {abs(polarized_at_max_n - diverse_at_max_n):.3f})"
    )
