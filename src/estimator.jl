abstract type AbstractEstimator end

function evaluate(
    est::AbstractEstimator, dataset_train::Dataset, dataset_test::Dataset, model::POVARModel
)
    θ̂ = estimate(est, dataset_train, model)
    (; X, proj, Y) = dataset_test
    Ŷ_shifted = proj .* (θ̂ * X)
    errors = (Ŷ_shifted[1:(end - 1)] .- Y[2:end])[proj[2:end]]
    return mean(abs2, errors)
end

function tune(est::AbstractEstimator, ::Dataset, ::Dataset, ::POVARModel)
    return est
end

## Exact

struct ExactEstimator <: AbstractEstimator end

function estimate(::ExactEstimator, dataset::Dataset, model::POVARModel)
    return model.θ
end

## Dense

struct DenseEstimator <: AbstractEstimator end

function estimate(::DenseEstimator, dataset::Dataset, model::POVARModel)
    (; proj, Y) = dataset
    Γ₀ = empirical_covariance(proj, Y, model, 0)
    Γ₁ = empirical_covariance(proj, Y, model, 1)
    return Γ₁ * pinv(Γ₀)
end

## Dantzig

struct SparseEstimator{R<:Real} <: AbstractEstimator
    λ::R
end

function estimate(estimator::SparseEstimator, dataset::Dataset, model::POVARModel)
    (; proj, Y) = dataset
    (; λ) = estimator
    D = size(Y, 1)
    Γ₀ = empirical_covariance(proj, Y, model, 0)
    Γ₁ = empirical_covariance(proj, Y, model, 1)
    opt = Model(HiGHS.Optimizer)
    set_silent(opt)
    @variable(opt, θ₊[1:D, 1:D] >= 0)
    @variable(opt, θ₋[1:D, 1:D] >= 0)
    @constraint(opt, (θ₊ - θ₋) * Γ₀ - Γ₁ .<= λ)
    @constraint(opt, Γ₁ - (θ₊ - θ₋) * Γ₀ .<= λ)
    @objective(opt, Min, sum(θ₊ + θ₋))
    optimize!(opt)
    @assert is_solved_and_feasible(opt)
    θ̂ = value.(θ₊) .- value.(θ₋)
    return θ̂
end

function tune(
    ::SparseEstimator, dataset_train::Dataset, dataset_test::Dataset, model::POVARModel
)
    λ_vals = 10 .^ (-4:0.2:4)
    errors = map(λ_vals) do λ
        evaluate(SparseEstimator(λ), dataset_train, dataset_test, model)
    end
    λ_best = λ_vals[argmin(errors)]
    return SparseEstimator(λ_best)
end
