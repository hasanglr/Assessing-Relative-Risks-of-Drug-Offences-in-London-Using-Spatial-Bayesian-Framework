functions {
  real icar_normal_lpdf(vector phi, int N, array[] int node1, array[] int node2) {
    return -0.5 * dot_self(phi[node1] - phi[node2]);
  }
}
data {
  int<lower=0> N;
  int<lower=0> N_edges;
  array[N_edges] int<lower=1, upper=N> node1;
  array[N_edges] int<lower=1, upper=N> node2;
  array[N] int<lower=0> Y;                                        // dependent variable (drug offences)
  matrix[N, 2] X;                                                  // independent variables (no educational qualifation and number of migrants)
  vector<lower=0>[N] E;                                           // estimated number of expected cases of drug offence
}
transformed data {
  vector[N] log_offset = log(E);                                  // use the expected cases as an offset and add to the regression model
}
parameters {
  real alpha;                                                     // define the intercept (overall risk in population)
  vector[2] beta;                                                  // define the coefficient for the independent variables
  real<lower=0> sigma;                                            // define the overall standard deviation producted with spatial effect smoothing term phi
  vector[N] phi;                                                  // spatial effect smoothing term or spatial ICAR component of the model 
}
model {
  phi ~ icar_normal(N, node1, node2);                             // prior for the spatial random effects
  Y ~ poisson_log(log_offset + alpha + X * beta + phi * sigma);    // likelihood function i.e., spatial ICAR model using Possion distribution
  alpha ~ normal(0.0, 1.0);                                       // prior for intercept   (weak/uninformative prior)
  beta ~ normal(0.0, 1.0);                                        // prior for coefficients (weak/uninformative prior)
  sigma ~ normal(0.0, 1.0);                                       // prior for SD          (weak/uninformative prior)
  sum(phi) ~ normal(0, 0.001*N);
}
generated quantities {
  vector[N] eta = alpha + X * beta + phi * sigma;                  // do eta equals alpha + beta*X + phi*sigma to get the relative risk for areas 
  vector[N] mu = exp(eta);                                        // the exponentiate eta to mu areas-specific relative risk ratios (RRs)
}


