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

  vector[N_votes] D_obs; // cross-cutting exposure (audience diversity)
  // 1 if D_obs[i] is a genuinely observed prior-audience value, 0 if no
  // prior audience exists (debaters, n<=1 voters). beta_D's slope is gated
  // on this flag rather than trusting D_obs's placeholder (0) to mean "no
  // data" -- see the has_prior_D comment in ideal_point_helper.py's
  // stan_data_gen for why pooling imputed and real D into one continuous
  // predictor is unsafe.
  array[N_votes] int<lower=0,upper=1> has_prior_D;
}

parameters {
  // Topic-level fixed effects
  vector[N_topics] beta_D;

  // Intercept shift for votes with no prior audience (debaters, n<=1
  // voters), separate from beta_D so the "no data" mean vote-propensity
  // doesn't get attributed to wherever D_obs's placeholder value sits.
  // Global rather than per-topic: this is expected to mostly reflect a
  // structural difference (e.g. debaters trivially voting their own
  // declared side) rather than a topic-varying effect, and per-topic gamma
  // would be poorly identified in topics where has_prior_D==0 rows are
  // sparse.
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
           ? beta_D[t] * D_obs[i]
           : gamma_no_prior);

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
           ? beta_D[t] * D_obs[i]
           : gamma_no_prior);

    log_lik[i] = bernoulli_logit_lpmf(y[i] | eta);
  }
}
