using Graphs
import Graphs: pagerank as pagerank_lib
using GraphRecipes, Plots
include("./simplerank.jl")
using .simplerank
pythonplot()

bara_alb = barabasi_albert(20, 11, 5, is_directed=true)
graf = DiGraph(Edge.([(1, 2), (2, 3), (3, 2), (3, 5), (4, 3), (6, 3), (4, 1), (6, 4), (5, 2), (4, 6)]))
erdos = erdos_renyi(20, 0.2, is_directed=true)
simplerank.graph_simplerank(bara_alb, 0.85)

a = 2