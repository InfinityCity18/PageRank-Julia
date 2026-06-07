module simplerank

using Graphs
using SparseArrays
using LinearAlgebra

export graph_simplerank, matrix_A

function graph_simplerank(graph, d=1.0, iters=100000, ϵ=10E-30)
    @assert 0.0 < d <= 1.0
    A = matrix_A(graph)
    v = power_iteration(d * A, iters, ϵ)
    v[v.<ϵ] .= 0.0
    return any(v .< 0) ? -v : v
end

function power_iteration(A, iters, x₀, ϵ=10E-30)
    xᵢ = x₀
    xᵢ₊₁ = xᵢ
    for _ in 1:iters
        xᵢ = xᵢ₊₁
        xᵢ₊₁ = A * xᵢ
        normalize!(xᵢ₊₁, 1)
        if norm(xᵢ₊₁ - xᵢ) < ϵ
            @info "power iteration return because of norm < ϵ"
            break
        end
    end
    normalize!(xᵢ₊₁, 1)
    return xᵢ₊₁
end
power_iteration(A) = power_iteration(A, 100000)
power_iteration(A, iters) = power_iteration(A, iters, rand(size(A)[1]))
power_iteration(A, iters, ϵ::Real) = power_iteration(A, iters, rand(size(A)[1]), ϵ)

function matrix_A(graph::Graphs.AbstractGraph)
    n = nv(graph)
    Nᵤ = outdegree(graph)

    I = Int[]
    J = Int[]
    V = Float64[]

    sizehint!(I, ne(graph))
    sizehint!(J, ne(graph))
    sizehint!(V, ne(graph))

    for u in 1:n
        for v in inneighbors(graph, u)
            push!(I, u)
            push!(J, v)
            push!(V, 1.0 / Nᵤ[v])
        end
    end

    A = sparse(I, J, V, n, n)
    return A
end

end