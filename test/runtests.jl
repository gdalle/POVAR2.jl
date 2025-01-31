using POVAR2
using Test

@testset verbose = true "POVAR2.jl" begin
    @testset "Model" begin
        include("model.jl")
    end
    @testset "Covariance" begin
        include("covariance.jl")
    end
    @testset "Estimator" begin
        include("estimator.jl")
    end
end
