abstract type AbstractEstimator end

function evaluate(θ̂::AbstractMatrix, dataset::Dataset)
    (; X, proj, Y) = dataset
    T = duration(dataset)
    Ŷ_shifted = θ̂ * X
    errors = (Ŷ_shifted[:, 1:(T - 1)] .- Y[:, 2:T])[proj[:, 2:T]]
    return mean(abs, errors)
end

function estimation_error(rng, est::AbstractEstimator, model::POVARModel)
    dataset = rand(rng, model)[1]
    θ̂ = estimate(est, dataset, model)
    return opnorm(θ̂ - model.θ, Inf)
end

## Dense

struct DenseEstimator <: AbstractEstimator end

function estimate(::DenseEstimator, dataset::Dataset, model::POVARModel)
    (; proj, Y) = dataset
    Γ₀, Γ₁ = empirical_covariances(proj, Y, model, [0, 1])
    return Γ₁ * pinv(Γ₀)
end

## Sparse

@kwdef struct SparseEstimator <: AbstractEstimator
    λ_vals::Vector{Float64} = 10 .^ collect(-6:0.5:2)
end

function dantzig_solutions(dataset::Dataset, model::POVARModel, λ_vals)
    (; proj, Y) = dataset
    D = size(Y, 1)
    Γ₀, Γ₁ = empirical_covariances(proj, Y, model, [0, 1])

    opt = Model(HiGHS.Optimizer)
    set_silent(opt)

    @variable(opt, λ in Parameter(0.0))
    @variable(opt, θ₊[1:D, 1:D] >= 0)
    @variable(opt, θ₋[1:D, 1:D] >= 0)
    @constraint(opt, (θ₊ - θ₋) * Γ₀ - Γ₁ .<= λ)
    @constraint(opt, Γ₁ - (θ₊ - θ₋) * Γ₀ .<= λ)
    @objective(opt, Min, sum(θ₊ + θ₋))

    θ̂_solutions = Matrix{Float64}[]
    for λi in λ_vals
        set_parameter_value(λ, λi)
        optimize!(opt)
        @assert is_solved_and_feasible(opt)
        θ̂ = value.(θ₊) .- value.(θ₋)
        push!(θ̂_solutions, θ̂)
    end
    return θ̂_solutions
end

function estimate(est::SparseEstimator, dataset::Dataset, model::POVARModel)
    (; λ_vals) = est
    if length(λ_vals) == 1
        return only(dantzig_solutions(dataset, model, λ_vals))
    else
        dataset_train, dataset_test = split(dataset)
        θ̂_candidates = dantzig_solutions(dataset_train, model, λ_vals)
        errors = map(θ̂ -> evaluate(θ̂, dataset_test), θ̂_candidates)
        return θ̂_candidates[argmin(errors)]
    end
end
