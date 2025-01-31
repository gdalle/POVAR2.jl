using POVAR2
using LinearAlgebra, Statistics
using StableRNGs
using Test

rng = StableRNG(63)

@testset "Dense, fully observed" begin
    D = 3
    T = 10000

    θ = random_transition(rng, D, D)
    σ = 0.05
    p = 0.5
    ω = 0.05

    fully_observed_model = POVARModel(; θ, σ, p=1.0, ω, T)
    dataset_train = rand(rng, fully_observed_model)[1]

    θ̂_exact = estimate(ExactEstimator(), dataset_train, fully_observed_model)
    @test θ̂_exact == θ

    θ̂_dense = estimate(DenseEstimator(), dataset_train, fully_observed_model)
    @test θ̂_dense ≈ θ rtol = 0.2

    θ̂_sparse = estimate(SparseEstimator(0.0), dataset_train, fully_observed_model)
    @test θ̂_sparse ≈ θ̂_dense
end

@testset "Realistic" begin
    D = 10
    S = 2
    T = 10_000

    θ = random_transition(rng, D, S)
    σ = 0.1
    p = 0.5
    ω = 0.1

    realistic_model = POVARModel(; θ, σ, p=0.5, ω, T)
    dataset_train, dataset_test = rand(rng, realistic_model, 2)

    est_best = tune(SparseEstimator(0), dataset_train, dataset_test, realistic_model)
    err_best = evaluate(est_best, dataset_train, dataset_test, realistic_model)
    @test err_best <
        evaluate(DenseEstimator(), dataset_train, dataset_test, realistic_model)
end
