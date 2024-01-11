## sample_trajectories.jl
##
## Author: Taylor Kessinger <tkess@sas.upenn.edu>
## Simulates a population of agents for a fixed gossip length.
## Iterates the processes of gameplay, view updating, gossip, and strategy updating
## until one strategic type has fixed.
## Outputs trajectories to disk.

using Revise
using Statistics
import PyPlot as plt
using LaTeXStrings
using DelimitedFiles

# Import custom module
includet("NeutralGossip.jl")
using .NeutralGossip

# Define the social norms
SH_norm = [0 0; 0 1]
SC_norm = [0 1; 0 1]
SJ_norm = [1 0; 0 1]
SS_norm = [1 1; 0 1]
norm_names = ["Shunning", "Scoring", "Stern Judging", "Simple Standing"]

# Declare the interior points of the simplex we'll be starting with
# Frequencies are given in order: ALLC, ALLD, DISC
initial_fractions = 
[
    [0.6,0.3,0.1],
    [0.4,0.4,0.2],
    [0.2,0.5,0.3],
    [0.6,0.1,0.3],
    [0.4,0.2,0.4],
    [0.2,0.3,0.5]
]

N = 100 # population size
tauvals = [0.0, 0.2] # gossip length
e_rate = 0.02 # error rate

# For each value of the gossip length tau
for (ti, tauval) in enumerate(tauvals)
    # Rescale the gossip length into units of N^3
    glen = floor(Int64,tauval*N^3)

    for (ii, i_frac) in enumerate(initial_fractions)
        # Initialize the simulation parameters and population
        sp = SimParams(N,5.0,1.0,e_rate,e_rate,1.0,0.0,BitMatrix(SJ_norm))
        pop = Population(N)
        # Pick out the number of ALLC and ALLD individuals
        num_ALLC = N*i_frac[1]
        num_ALLD = N*i_frac[2]
        # Assign strategies
        if num_ALLC > 0
            [pop.strategies[i,:] = [1,1] for i in 1:floor(Int64,num_ALLC)]
        end
        if num_ALLD > 0
            [pop.strategies[i,:] = [0,0] for i in floor(Int64,num_ALLC + 1):floor(Int64,num_ALLC + num_ALLD)]
        end
        # Prepare an array to store the trajectory values
        trajectory = Vector{Int64}[]
        gp = GossipParams(glen)

        # Allow reputations to equilibrate before simulating
        burn_time = 100
        for generation in 1:burn_time
            do_gossip!(pop,sp,gp)
        end

        # Continue simulation until one type has fixed
        while !(any(get_frequencies(pop,sp)*sp.N .== sp.N))
            evolve_modified_loop!(pop,sp,gp)
            push!(trajectory,[count([all(row .== [1,1]) for row in eachrow(pop.strategies)]),
            count([all(row .== [0,0]) for row in eachrow(pop.strategies)]),
            count([all(row .== [0,1]) for row in eachrow(pop.strategies)])])
        end
        push!(trajectory,[count([all(row .== [1,1]) for row in eachrow(pop.strategies)]),
            count([all(row .== [0,0]) for row in eachrow(pop.strategies)]),
            count([all(row .== [0,1]) for row in eachrow(pop.strategies)])])

        trajectory = hcat(trajectory...)

        # Write trajectory to file
        open("sample_trajectory_parallel_tau_$(tauval)_N_$(N)_$(ii)_precise_fractions_different_order.csv", "w") do io
            writedlm(io, trajectory, ',')  # ',' specifies the delimiter
        end
        println("$ii, $(get_frequencies(pop,sp)*sp.N)")
    end
end