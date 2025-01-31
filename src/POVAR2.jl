module POVAR2

using CairoMakie: plot
using Distributions: Bernoulli, MvNormal, Normal
using HiGHS: HiGHS
using JuMP:
    Model,
    @constraint,
    @objective,
    @variable,
    is_solved_and_feasible,
    optimize!,
    set_silent,
    termination_status,
    value
using JuMP.MOI: OPTIMAL
using LinearAlgebra: Diagonal, I, checksquare, dot, eigmin, issymmetric, mul!, opnorm, pinv
using MonteCarloMeasurements: Particles
using OhMyThreads: tmap
using Random: AbstractRNG, rand!
using Statistics: mean
using StatsBase: crosscov, sample

include("model.jl")
include("covariance.jl")
include("estimator.jl")
include("param.jl")
include("slope.jl")

export POVARModel
export scaling, empirical_covariances
export ExactEstimator, DenseEstimator, SparseEstimator
export estimate, evaluate, tune, estimation_error
export random_transition
export theil_sen

end # module POVAR2
