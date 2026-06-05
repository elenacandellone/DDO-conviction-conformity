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

  vector[N_votes] phi_obs; // peer influence  
  vector[N_votes] tau_obs; // topic alignment
}

parameters {
  // Topic-level fixed effects
  vector[N_topics] beta_phi;
  vector[N_topics] beta_tau;

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

  beta_phi ~ normal(0, 1);
  beta_tau ~ normal(0, 1);


  // Likelihood
    for (i in 1:N_votes) {
    int d = debate_id[i];
    int u = user_id[i];
    int t = topic_id_debate[d];

    real eta =
        alpha_user[u]
      + alpha_debate[d]
      + beta_phi[t] * phi_obs[i]
      + beta_tau[t] * tau_obs[i];

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
      + beta_phi[t] * phi_obs[i]
      + beta_tau[t] * tau_obs[i];

    log_lik[i] = bernoulli_logit_lpmf(y[i] | eta);
  }
}

