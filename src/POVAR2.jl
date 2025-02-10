module POVAR2

using CairoMakie
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
using MonteCarloMeasurements: Particles, pmean, pmedian, pquantile, pstd
using OhMyThreads: tmap
using ProgressMeter: @showprogress
using Random: AbstractRNG, rand!
using Statistics: mean, median
using StatsBase: crosscov, sample

include("model.jl")
include("covariance.jl")
include("estimator.jl")
include("param.jl")
include("slope.jl")
include("plots.jl")

export POVARModel
export scaling, empirical_covariances
export ExactEstimator, DenseEstimator, SparseEstimator
export estimate, evaluate, tune, estimation_error, estimation_error_particles
export random_transition
export theil_sen
export plot_p, plot_T, plot_D, plot_S, plot_ω, plot_SD

end # module POVAR2
