function scaling(model::POVARModel, h::Integer)
    (; p) = model
    S = p * p'
    if h == 0
        return S
    elseif h == 1
        Diagonal(S) .= Diagonal(p)
        return S
    end
end

function empirical_covariance(
    π::AbstractMatrix{Bool}, Y::AbstractMatrix{<:Real}, model::POVARModel, h::Integer
)
    (; ω) = model
    D, T = size(Y)
    Γ = zeros(D, D)
    X̂ = π .* Y
    @views for t in 1:(T - h)
        Γₜ = X̂[:, t + h] * X̂[:, t]'
        Γ .+= Γₜ
    end
    Γ ./= (T - h)
    Γ ./= scaling(model, h)
    if h == 0
        Γ .-= ω^2 * I(D)
    end
    return Γ
end
