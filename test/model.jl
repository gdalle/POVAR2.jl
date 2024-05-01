D = 3
T = 1000

θ = randn(rng, D, D)
θ ./= 2 * opnorm(θ, 2)
Σ = Diagonal(rand(rng, D))
p = rand(rng, D)
ω = 0.1

model = POVARModel(; θ, Σ, p, ω)
(; X, π, Y) = rand(rng, model, T)

@test size(X) == (D, T)
@test size(π) == (D, T)
@test size(Y) == (D, T)
@test eltype(X) == eltype(Y)
@test eltype(π) <: Bool
@test all(1:D) do d
    isapprox(mean(π[d, :]), p[d]; atol=0.1)
end

no_innovation_model = POVARModel(; θ, Σ=0.0 * I(D), p, ω)
(; X, π, Y) = rand(rng, no_innovation_model, T)

@test all(2:T) do t
    X[:, t] == θ * X[:, t - 1]
end

no_noise_partial_model = POVARModel(; θ, Σ, p, ω=0.0)
(; X, π, Y) = rand(rng, no_noise_partial_model, T)

@test Y[π] == X[π]
@test all(iszero, Y[.!π])

no_noise_model = POVARModel(; θ, Σ, p=ones(D), ω=0.0)
(; X, π, Y) = rand(rng, no_noise_model, T)

@test all(isone, π)
@test Y == X

just_noise_model = POVARModel(; θ, Σ=1.0 * I(D), p=zeros(D), ω)
(; X, π, Y) = rand(rng, just_noise_model, T)

@test all(iszero, π)
@test mean(Y) ≈ 0 atol = 0.05
@test std(Y) ≈ ω atol = 0.05

no_obs_model = POVARModel(; θ, Σ, p=zeros(D), ω=0.0)
(; X, π, Y) = rand(rng, no_obs_model, T)

@test all(iszero, π)
@test all(iszero, Y)
