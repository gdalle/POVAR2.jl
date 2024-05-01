D = 3
T = 10000

θ = randn(rng, D, D)
θ ./= 2 * opnorm(θ, 2)
Σ = Diagonal(rand(rng, D))
p = rand(rng, D)
ω = 0.1

model = POVARModel(; θ, Σ, p, ω)
(; X, π, Y) = rand(rng, model, T)

Γ̂₀ = empirical_covariance(π, Y, model, 0)
Γ̂₁ = empirical_covariance(π, Y, model, 1)
@test opnorm(Γ̂₁, 2) <= opnorm(Γ̂₀, 2)

θ_sym = Symmetric(θ)
symmetric_no_noise_model = POVARModel(; θ=θ_sym, Σ=4.0 * I(D), p=ones(D), ω=0.0)
(; X, π, Y) = rand(rng, symmetric_no_noise_model, T)

Γ̂₀ = empirical_covariance(π, Y, symmetric_no_noise_model, 0)
Γ̂₁ = empirical_covariance(π, Y, symmetric_no_noise_model, 1)
@test Γ̂₀ ≈ 4 * inv(I - θ_sym^2) rtol = 0.1
@test Γ̂₁ ≈ θ_sym * Γ̂₀ rtol = 0.1
