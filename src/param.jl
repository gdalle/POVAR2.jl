function random_transition(rng::AbstractRNG, D::Integer, S::Integer)
    @assert D >= S
    θ = randn(rng, D, D)
    for d₁ in 1:D
        zero_columns = sample(rng, 1:D, D - S; replace=false)
        for d₂ in zero_columns
            θ[d₁, d₂] = 0
        end
    end
    return 0.5 * θ / opnorm(θ, 2)
end
