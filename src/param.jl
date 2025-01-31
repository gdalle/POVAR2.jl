function random_transition(rng::AbstractRNG, D::Integer, S::Integer)
    θ = zeros(D, D)
    for d₁ in 1:D
        nonzero_columns = sample(1:D, S; replace=false)
        for d₂ in nonzero_columns
            θ[d₁, d₂] = randn(rng)
        end
    end
    return 0.5 * θ / opnorm(θ, 2)
end
