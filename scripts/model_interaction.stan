data {
  int<lower=1> N_votes;
  int<lower=1> N_debates;
  int<lower=1> N_topics;
  int<lower=1> N_users;

  array[N_debates] int<lower=1,upper=N_topics> topic_id_debate;
  array[N_votes] int<lower=1,upper=N_debates> debate_id;
  array[N_votes]   int<lower=1,upper=N_users>   user_id;

  // observations (y=1 if PRO, y=0 if CON)
  array[N_votes] int<lower=0,upper=1> y;

  vector[N_votes] D_obs;   // cross-cutting exposure (audience diversity)
  vector[N_votes] c_obs;   // persuasiveness
  vector[N_votes] tau_obs; // conviction / topic alignment
  // 1 if D_obs[i] is a genuinely observed prior-audience value, 0 if no
  // prior audience exists (debaters, n<=1 voters) -- see the has_prior_D
  // comment in ideal_point_helper.py's stan_data_gen. Gates both beta_D and
  // beta_tau_D below: the tau x D moderation term is exactly as undefined as
  // D itself wherever there's no prior audience to moderate anything with.
  array[N_votes] int<lower=0,upper=1> has_prior_D;
}

transformed data {
  // moderation term: does cross-cutting exposure moderate the pull of
  // conviction on the vote? (Sunstein/Mutz-style hypothesis). Computed here
  // rather than in Python so this model can reuse exactly the same stan_data
  // dict as model.stan, with no extra preprocessing step.
  vector[N_votes] tau_D_obs = tau_obs .* D_obs;
}

parameters {
  // Topic-level fixed effects
  vector[N_topics] beta_D;
  vector[N_topics] beta_c;
  vector[N_topics] beta_tau;
  vector[N_topics] beta_tau_D; // conviction x cross-cutting exposure interaction

  // Intercept shift for votes with no prior audience, replacing BOTH the
  // beta_D main effect and the beta_tau_D interaction for those rows (see
  // model.stan/model_D.stan for why this needs to be its own parameter
  // rather than relying on D_obs's placeholder value). Global rather than
  // per-topic -- see model_D.stan for the reasoning.
  real gamma_no_prior;

  // Random intercepts (non-centered)
  vector[N_users]   alpha_user_raw;
  vector[N_debates] alpha_debate_raw;

  real<lower=0> sigma_user;
  real<lower=0> sigma_debate;
}

transformed parameters {
  vector[N_users]   alpha_user   = sigma_user   * alpha_user_raw;
  vector[N_debates] alpha_debate = sigma_debate * alpha_debate_raw;
}

model {
  // Hyperpriors
  sigma_user   ~ normal(0, 1);
  sigma_debate ~ normal(0, 1);

  alpha_user_raw   ~ normal(0, 1);
  alpha_debate_raw ~ normal(0, 1);

  beta_D        ~ normal(0, 1);
  beta_c        ~ normal(0, 1);
  beta_tau      ~ normal(0, 1);
  beta_tau_D    ~ normal(0, 1);
  gamma_no_prior ~ normal(0, 1);


  // Likelihood
    for (i in 1:N_votes) {
    int d = debate_id[i];
    int u = user_id[i];
    int t = topic_id_debate[d];

    real eta =
        alpha_user[u]
      + alpha_debate[d]
      + (has_prior_D[i]
           ? beta_D[t] * D_obs[i] + beta_tau_D[t] * tau_D_obs[i]
           : gamma_no_prior)
      + beta_c[t]     * c_obs[i]
      + beta_tau[t]   * tau_obs[i];

    y[i] ~ bernoulli_logit(eta);
  }
}

generated quantities {
  vector[N_votes] log_lik;

  for (i in 1:N_votes) {
    int d = debate_id[i];
    int u = user_id[i];
    int t = topic_id_debate[d];

    real eta =
        alpha_user[u]
      + alpha_debate[d]
      + (has_prior_D[i]
           ? beta_D[t] * D_obs[i] + beta_tau_D[t] * tau_D_obs[i]
           : gamma_no_prior)
      + beta_c[t]     * c_obs[i]
      + beta_tau[t]   * tau_obs[i];

    log_lik[i] = bernoulli_logit_lpmf(y[i] | eta);
  }
}
