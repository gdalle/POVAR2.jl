module POVAR2

using CairoMakie: plot
using Distributions: Bernoulli, MvNormal, Normal
using HiGHS: HiGHS
using JuMP:
    Model,
    Parameter,
    @constraint,
    @objective,
    @variable,
    fix,
    is_solved_and_feasible,
    optimize!,
    parameter_value,
    set_parameter_value,
    set_silent,
    termination_status,
    value
using JuMP.MOI: OPTIMAL
using LinearAlgebra: Diagonal, I, checksquare, dot, eigmin, issymmetric, mul!, opnorm, pinv
using MonteCarloMeasurements: Particles
using OhMyThreads: tmap
using Random: AbstractRNG, rand!
using Statistics: mean, median
using StatsBase: crosscov, sample

include("model.jl")
include("covariance.jl")
include("estimator.jl")
include("param.jl")
include("slope.jl")

export POVARModel
export scaling, empirical_covariances
export ExactEstimator, DenseEstimator, SparseEstimator
export estimate, evaluate, tune, estimation_error, estimation_error_particles
export random_transition
export theil_sen

end # module POVAR2
