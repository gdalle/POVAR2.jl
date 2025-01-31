function theil_sen(x, y)
    n = length(x)
    slopes = [(y[j] - y[i]) / (x[j] - x[i]) for i in 1:n for j in (i + 1):n]
    α = median(slopes)
    intercepts = [y[i] - α * x[i] for i in 1:n]
    β = median(intercepts)
    return α, β
end
