using POVAR2
using Test

@testset verbose = true "POVAR2.jl" begin
    @testset "Model" begin
        include("model.jl")
    end
end
