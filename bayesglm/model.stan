data {{
	int<lower=1> K;
	int<lower=0> N;
	{y_type} y[N];
	matrix[N,K] x;
}}
parameters {{
	vector[K] beta;
	{parameter_statement}
}}
model {{
	real mu[N];
	vector[N] eta   ;
	eta <- x*beta;
	for (i in 1:N) {{
	   mu[i] <- {link_function}(eta[i]);
	}};
	{model_statement}
	{beta_priors}
}}

