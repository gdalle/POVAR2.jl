module POVAR2

using CairoMakie: plot
using Distributions: Bernoulli, MvNormal, Normal
using HiGHS: HiGHS
using JuMP:
    Model,
    @constraint,
    @objective,
    @variable,
    optimize!,
    set_silent,
    termination_status,
    value
using JuMP.MOI: OPTIMAL
using LinearAlgebra: Diagonal, I, checksquare, dot, eigmin, issymmetric, mul!, opnorm, pinv
using OhMyThreads: tmap
using Random: AbstractRNG, rand!

include("model.jl")
include("covariance.jl")
include("dense.jl")
include("dantzig.jl")

export POVARModel
export scaling, empirical_covariance
export estimate, DenseEstimator, DantzigEstimator

end # module POVAR2
