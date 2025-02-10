abstract type AbstractEstimator end

function evaluate(θ̂::AbstractMatrix, dataset::Dataset)
    (; X, proj, Y) = dataset
    (D, T) = size(X)
    T = duration(dataset)
    Ŷ_shifted = θ̂ * X
    sum_abs_errors = 0.0
    count_abs_errors = 0
    for t in 2:T
        for d in 1:D
            pr = proj[d, t]
            sum_abs_errors += abs2(Ŷ_shifted[d, t - 1] - Y[d, t]) * pr
            count_abs_errors += pr
        end
    end
    return sum_abs_errors / count_abs_errors
end

function estimation_error(rng::AbstractRNG, est::AbstractEstimator, model::POVARModel)
    dataset_train, dataset_validation = rand(rng, model, 2)
    θ̂ = estimate(est, dataset_train, dataset_validation, model)
    return opnorm(θ̂ - model.θ, Inf)
end

function estimation_error_particles(
    rng::AbstractRNG,
    est::AbstractEstimator;
    D::Integer,
    S::Integer,
    T::Integer,
    p::Real,
    σ::Real,
    ω::Real,
    samples::Integer=10,
)
    errors = tmap(1:samples) do _
        yield()
        θ = random_transition(rng, D, S)
        model = POVARModel(; θ, p, σ, ω, T)
        estimation_error(rng, est, model)
    end
    return Particles(errors)
end

## Dense

struct DenseEstimator <: AbstractEstimator end

function estimate(
    ::DenseEstimator, dataset_train::Dataset, dataset_validation::Dataset, model::POVARModel
)
    (; proj, Y) = dataset_train
    Γ₀, Γ₁ = empirical_covariances(proj, Y, model, [0, 1])
    return Γ₁ * pinv(Γ₀)
end

## Sparse

@kwdef struct SparseEstimator <: AbstractEstimator
    λ_vals::Vector{Float64} = 10 .^ collect(-5:0.2:1)
end

function dantzig_solutions(dataset::Dataset, model::POVARModel, λ_vals)
    (; proj, Y) = dataset
    D = size(Y, 1)
    Γ₀, Γ₁ = empirical_covariances(proj, Y, model, [0, 1])

    opt = Model(HiGHS.Optimizer; add_bridges=false)
    set_silent(opt)

    @variable(opt, λ)
    @variable(opt, θ₊[1:D, 1:D] >= 0)
    @variable(opt, θ₋[1:D, 1:D] >= 0)
    @constraint(opt, (θ₊ - θ₋) * Γ₀ - Γ₁ .<= λ)
    @constraint(opt, Γ₁ - (θ₊ - θ₋) * Γ₀ .<= λ)
    @objective(opt, Min, sum(θ₊ + θ₋))

    θ̂_solutions = Matrix{Float64}[]
    for λi in λ_vals
        fix(λ, λi)
        optimize!(opt)
        @assert is_solved_and_feasible(opt)
        θ̂ = value.(θ₊) .- value.(θ₋)
        push!(θ̂_solutions, θ̂)
    end
    return θ̂_solutions
end

function estimate(
    est::SparseEstimator,
    dataset_train::Dataset,
    dataset_validation::Dataset,
    model::POVARModel,
)
    (; λ_vals) = est
    if length(λ_vals) == 1
        return only(dantzig_solutions(dataset_train, model, λ_vals))
    else
        θ̂_candidates = dantzig_solutions(dataset_train, model, λ_vals)
        errors = map(θ̂ -> evaluate(θ̂, dataset_validation), θ̂_candidates)
        return θ̂_candidates[argmin(errors)]
    end
end
