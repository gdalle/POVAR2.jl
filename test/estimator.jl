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
    dataset_train, dataset_test = rand(rng, fully_observed_model, 2)

    θ̂_dense = estimate(DenseEstimator(), dataset_train, dataset_test, fully_observed_model)
    @test θ̂_dense ≈ θ rtol = 0.2

    θ̂_sparse = estimate(
        SparseEstimator([0.0]), dataset_train, dataset_test, fully_observed_model
    )
    @test θ̂_sparse ≈ θ̂_dense
end
