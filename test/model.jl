using POVAR2
using LinearAlgebra, Statistics
using StableRNGs
using Test

rng = StableRNG(63)

D = 3
T = 1000

θ = randn(rng, D, D)
θ ./= 2 * opnorm(θ, 2)
Σ = Diagonal(rand(rng, D))
p = rand(rng, D)
ω = 0.1

model = POVARModel(; θ, Σ, p, ω, T)
(; X, proj, Y) = rand(rng, model)

@testset "Formalities" begin
    @test size(X) == (D, T)
    @test size(proj) == (D, T)
    @test size(Y) == (D, T)
    @test eltype(X) == eltype(Y)
    @test eltype(proj) == Bool
    @test all(1:D) do d
        isapprox(mean(proj[d, :]), p[d]; atol=0.1)
    end
end

@testset "No innovation" begin
    no_innovation_model = POVARModel(; θ, Σ=0.0 * I(D), p, ω, T)
    (; X, proj, Y) = rand(rng, no_innovation_model)

    @test all(2:T) do t
        X[:, t] == θ * X[:, t - 1]
    end
end

@testset "No noise" begin
    no_noise_partial_model = POVARModel(; θ, Σ, p, ω=0.0, T)
    (; X, proj, Y) = rand(rng, no_noise_partial_model)

    @test Y[proj] == X[proj]
    @test all(iszero, Y[.!proj])

    no_noise_model = POVARModel(; θ, Σ, p=ones(D), ω=0.0, T)
    (; X, proj, Y) = rand(rng, no_noise_model)

    @test all(isone, proj)
    @test Y == X
end

@testset "Just noise" begin
    just_noise_model = POVARModel(; θ, Σ=1.0 * I(D), p=zeros(D), ω, T)
    (; X, proj, Y) = rand(rng, just_noise_model)

    @test all(iszero, proj)
    @test mean(Y) ≈ 0 atol = 0.05
    @test std(Y) ≈ ω atol = 0.05
end

@testset "No obs" begin
    no_obs_model = POVARModel(; θ, Σ, p=zeros(D), ω=0.0, T)
    (; X, proj, Y) = rand(rng, no_obs_model)

    @test all(iszero, proj)
    @test all(iszero, Y)
end
