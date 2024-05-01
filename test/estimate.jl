D = 3
T = 10000

θ = randn(rng, D, D)
θ ./= 2 * opnorm(θ, 2)
Σ = Diagonal(rand(rng, D))
p = rand(rng, D)
ω = 0.05

fully_observed_model = POVARModel(; θ, Σ, p=ones(D), ω)
(; X, π, Y) = rand(rng, fully_observed_model, T)

θ̂_dense = estimate(DenseEstimator(), π, Y, fully_observed_model)
@test θ̂_dense ≈ θ rtol = 0.1

θ̂_dantzig = estimate(DantzigEstimator(0.0), π, Y, fully_observed_model)
@test θ̂_dantzig ≈ θ rtol = 0.1
