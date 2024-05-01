struct DenseEstimator end

function estimate(
    ::DenseEstimator, π::AbstractMatrix{Bool}, Y::AbstractMatrix{<:Real}, model::POVARModel
)
    Γ₀ = empirical_covariance(π, Y, model, 0)
    Γ₁ = empirical_covariance(π, Y, model, 1)
    return Γ₁ * pinv(Γ₀)
end
