using LinearAlgebra
using POVAR2
using StableRNGs
using Statistics
using Test

rng = StableRNG(63)

@testset verbose = true "POVAR2.jl" begin
    @testset "Model" begin
        include("model.jl")
    end
    @testset "Covariance" begin
        include("covariance.jl")
    end
    @testset "Estimate" begin
        include("estimate.jl")
    end
end
