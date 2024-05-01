struct DantzigEstimator{R<:Real}
    λ::R
end

function estimate(
    estimator::DantzigEstimator,
    π::AbstractMatrix{Bool},
    Y::AbstractMatrix{<:Real},
    model::POVARModel,
)
    (; λ) = estimator
    D = size(Y, 1)
    Γ₀ = empirical_covariance(π, Y, model, 0)
    Γ₁ = empirical_covariance(π, Y, model, 1)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, θ₊[1:D, 1:D] >= 0)
    @variable(model, θ₋[1:D, 1:D] >= 0)
    @constraint(model, (θ₊ - θ₋) * Γ₀ - Γ₁ .<= λ)
    @constraint(model, Γ₁ - (θ₊ - θ₋) * Γ₀ .<= λ)
    @objective(model, Min, sum(θ₊ + θ₋))
    optimize!(model)
    @assert termination_status(model) == OPTIMAL
    θ̂ = value.(θ₊) .- value.(θ₋)
    return θ̂
end
