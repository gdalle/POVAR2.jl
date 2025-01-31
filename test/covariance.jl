using POVAR2
using LinearAlgebra, Statistics
using StableRNGs
using StatsBase
using Test

rng = StableRNG(63)

M = rand(rng, 5, 1000)
@test POVAR2.myautocov(M, [0, 1, 2]) ≈ POVAR2.fastautocov(M, [0, 1, 2]) rtol = 0.01

D = 3
T = 10000

θ = random_transition(rng, D, D)
σ = 0.3
p = 0.2
ω = 0.1

model = POVARModel(; θ, σ, p, ω, T)
(; X, proj, Y) = rand(rng, model)[1]

Γ̂₀, Γ̂₁ = empirical_covariances(proj, Y, model, [0, 1])
@test opnorm(Γ̂₁, 2) <= opnorm(Γ̂₀, 2)

θ_sym = Symmetric(θ)
symmetric_no_noise_model = POVARModel(; θ=θ_sym, σ=2.0, p=1.0, ω=0.0, T)
(; X, proj, Y) = rand(rng, symmetric_no_noise_model)[1]

Γ̂₀, Γ̂₁ = empirical_covariances(proj, Y, symmetric_no_noise_model, [0, 1])
@test Γ̂₀ ≈ 4 * inv(I - θ_sym^2) rtol = 0.1
@test Γ̂₁ ≈ θ_sym * Γ̂₀ rtol = 0.1
