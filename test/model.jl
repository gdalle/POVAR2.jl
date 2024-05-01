using LinearAlgebra
using POVAR2
using StableRNGs
using Statistics
using Test

rng = StableRNG(63)

D = 10
T = 1000

θ = 0.1 * randn(rng, D, D)
Σ0 = Diagonal(zeros(D))
Σ1 = Diagonal(ones(D))
p = rand(rng, D)
p0 = zeros(D)
p1 = ones(D)
ω0 = 0.0
ω1 = 1.0

model = Model(; θ, Σ=Σ1, p, ω=ω1)
(; X, π, Y) = rand(rng, model, T)

@test length(model) == D
@test size(X) == (D, T)
@test size(π) == (D, T)
@test size(Y) == (D, T)
@test eltype(X) == eltype(Y)
@test eltype(π) <: Bool
@test all(1:D) do d
    isapprox(mean(π[d, :]), p[d]; atol=0.1)
end

no_innovation_model = Model(; θ, Σ=Σ0, p, ω=ω1)
(; X, π, Y) = rand(rng, no_innovation_model, T)

@test all(2:T) do t
    X[:, t] == θ * X[:, t - 1]
end

no_noise_partial_model = Model(; θ, Σ=Σ1, p, ω=ω0)
(; X, π, Y) = rand(rng, no_noise_partial_model, T)

@test Y[π] == X[π]
@test all(iszero, Y[.!π])

no_noise_model = Model(; θ, Σ=Σ1, p=p1, ω=ω0)
(; X, π, Y) = rand(rng, no_noise_model, T)

@test all(isone, π)
@test Y == X

just_noise_model = Model(; θ, Σ=Σ1, p=p0, ω=ω1)
(; X, π, Y) = rand(rng, just_noise_model, T)

@test all(iszero, π)
@test mean(Y) ≈ 0 atol = 0.01
@test std(Y) ≈ ω1 atol = 0.01

no_obs_model = Model(; θ, Σ=Σ1, p=p0, ω=ω0)
(; X, π, Y) = rand(rng, no_obs_model, T)

@test all(iszero, π)
@test all(iszero, Y)
