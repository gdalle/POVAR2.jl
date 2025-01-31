using POVAR2
using LinearAlgebra, Statistics
using StableRNGs
using Test

rng = StableRNG(63)

D = 3
T = 1000

θ = random_transition(rng, D, D)
σ = 0.3
p = 0.5
ω = 0.1

model = POVARModel(; θ, σ, p, ω, T)
(; X, proj, Y) = rand(rng, model)[1]

@testset "Formalities" begin
    @test size(X) == (D, T)
    @test size(proj) == (D, T)
    @test size(Y) == (D, T)
    @test eltype(X) == eltype(Y)
    @test eltype(proj) == Bool
    @test isapprox(mean(proj), p; atol=0.1)
end

@testset "No innovation" begin
    no_innovation_model = POVARModel(; θ, σ=0.0, p, ω, T)
    (; X, proj, Y) = rand(rng, no_innovation_model)[1]

    @test all(2:T) do t
        X[:, t] == θ * X[:, t - 1]
    end
end

@testset "No noise" begin
    no_noise_partial_model = POVARModel(; θ, σ, p, ω=0.0, T)
    (; X, proj, Y) = rand(rng, no_noise_partial_model)[1]

    @test Y[proj] == X[proj]
    @test all(iszero, Y[.!proj])

    no_noise_model = POVARModel(; θ, σ, p=1.0, ω=0.0, T)
    (; X, proj, Y) = rand(rng, no_noise_model)[1]

    @test all(isone, proj)
    @test Y == X
end

@testset "Just noise" begin
    just_noise_model = POVARModel(; θ, σ=1.0, p=0.0, ω, T)
    (; X, proj, Y) = rand(rng, just_noise_model)[1]

    @test all(iszero, proj)
    @test mean(Y) ≈ 0 atol = 0.05
    @test std(Y) ≈ ω atol = 0.05
end

@testset "No obs" begin
    no_obs_model = POVARModel(; θ, σ, p=0.0, ω=0.0, T)
    (; X, proj, Y) = rand(rng, no_obs_model)[1]

    @test all(iszero, proj)
    @test all(iszero, Y)
end
