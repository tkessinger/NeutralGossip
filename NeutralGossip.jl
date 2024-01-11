## NeutralGossip.jl
##
## Author: Taylor Kessinger <tkess@sas.upenn.edu>
## Generic module for Monte Carlo simulation
## of pairwise gameplay, observation, gossip,
## and strategy updating.


module NeutralGossip

    using Statistics

    export evolve!, evolve_modified_loop!, Population, SimParams, GossipParams
    export get_current_views, get_current_average_fitnesses, get_frequencies
    export do_gossip!, gossip_only!
    export interact!, update_views!


    # struct for storing simulation parameters
    struct SimParams
        N::Int64 # population size
        b::Float64 # benefit of cooperation
        c::Float64 # cost of cooperation
        assessment_error::Float64
        performance_error::Float64
        selection_strength::Float64
        strategy_mutation_rate::Float64
        social_norm::BitMatrix # contains bits corresponding to good reputation probabilities
        permitted_strategies::Vector{BitVector} # by default, this is any of ALLD, ALLC, DISC,
            # but it can be restricted to (e.g.) just DISC
    end

    # Constructor
    function SimParams(N::Int64, b::Float64, c::Float64, assessment_error::Float64,
        performance_error::Float64, selection_strength::Float64,
        strategy_mutation_rate::Float64, social_norm::BitMatrix)
        return SimParams(N, b, c, assessment_error, performance_error,
                selection_strength, strategy_mutation_rate, social_norm, [BitVector(strat) for strat in [[0,0],[1,1],[0,1]]])
        end

    # struct for storing gossip parameters
    struct GossipParams
        
        tau::Int64

        # Constructor
        function GossipParams(tau::Int64)
            new(tau)
        end

    end

    # main struct for storing population state
    mutable struct Population
        strategies::BitMatrix # each individual's strategy:
            # first bit is defect/cooperate with bad
            # second bit is defect/cooperate with good
            # [0,0] is ALLD, [1,1] is ALLC, [0,1] is DISC
        views::BitMatrix # the i,j element is i's view of j
        actions::BitMatrix # the i,j element is i's latest action toward j
        fitnesses::Vector{Float64}
        # Following are several arrays for storing the history of useful attributes
        strategy_history::Array{Vector{Float64}}
        fitness_history::Array{Vector{Float64}}
        views_history::Array{Vector{Float64}}
        action_history::Array{Vector{Float64}}
        # Constructor
        function Population(N::Int)
            strategies = BitMatrix(hcat(rand([[0,1]],N)...)')
            views = BitMatrix(rand(Bool[0, 1], N, N))
            actions = BitMatrix(rand(Bool[0, 1], N, N))
            fitnesses = zeros(Float64, N)
            strategy_history = Vector{Float64}[]
            fitness_history = Vector{Float64}[]
            views_history = Vector{Float64}[]
            action_history = Vector{Float64}[]
            new(strategies, views, actions, fitnesses, strategy_history, fitness_history, views_history, action_history)
        end
    end

    function interact!(pop::Population,sp::SimParams)

        # Update actions;
        # strategy bit 1 is defect(0)/cooperate(1) with those with bad reputations (0)
        # strategy bit 2 is defect(0)/cooperate(1) with those with good reputations (1)
        pop.actions .= pop.strategies[:,1] .* .!(pop.views) .+ pop.strategies[:,2] .* pop.views


        # Define an action error mask; this will tell us who makes mistakes
        action_error_mask = rand(sp.N,sp.N) .< sp.performance_error

        # Individuals who make mistakes defect (action error is asymmetric)
        # i.e., the only way to cooperate (1) is to intend to do so and NOT make a mistake
        pop.actions .= pop.actions .& .!action_error_mask

        avg_strategy_action = [mean(pop.actions[findall(row -> row == strategy, eachrow(pop.strategies)),:]) for strategy in sp.permitted_strategies]
        avg_action = mean(pop.actions)
        push!(pop.action_history, vcat(avg_strategy_action, avg_action))

    end

    function update_views!(pop::Population,sp::SimParams)
        
        # Each i, to update their opinion of j, chooses a random k;
        # they will use their own view of k and j's action toward k
        # to update their view of j
        k = rand(1:sp.N, sp.N, sp.N)

        # The first index of the social norm is i's view of k;
        # the second, j's action toward k
        # The +1 maps {0, 1} to {1, 2}
        pop.views = BitMatrix([sp.social_norm[(pop.views[i, k[i, j]] + 1),
            (pop.actions[j, k[i, j]] + 1)] for i in 1:sp.N, j in 1:sp.N])

        # Generate a view error mask with specified error probability
        view_error_mask = rand(sp.N, sp.N) .< sp.assessment_error

        # Introduce errors in pop.views using the XOR (⊻) operation;
        # if rand() is smaller than assessment_error, flip i's view of j
        pop.views = pop.views .⊻ view_error_mask

        # Obtain the strategy-wise and population-wide average views
        # The transpose is because we need the view the population has of each strategy,
        # not the view each strategy has of the rest of the population
        avg_strategy_views = [mean(pop.views'[findall(row -> row == strategy, eachrow(pop.strategies)),:]) for strategy in sp.permitted_strategies]
        avg_view = mean(pop.views)
        push!(pop.views_history, vcat(avg_strategy_views, avg_view))

    end

    function update_fitnesses!(pop::Population,sp::SimParams)
        # Individual i's fitness is:
        # benefit * everyone who cooperated with i
        # - cost * everyone whom i cooperated with
        pop.fitnesses = ((sp.b * sum(pop.actions,dims=1)' - sp.c * sum(pop.actions,dims=2))[:])/sp.N
        push!(pop.fitness_history, [mean(pop.fitnesses[findall(row -> row == strategy, eachrow(pop.strategies))]) for strategy in sp.permitted_strategies])

    end

    function copy_strategy!(pop::Population,sp::SimParams)
        # Select two individuals
        i,j = rand(1:sp.N), rand(1:sp.N)

        # Compute the Fermi function
        # Sanity check:
        # If Π_i >> Π_j, the exponent should be infinite, so the function value is zero
        # If Π_i << Π_j, the exponent should be zero, so the function value is 1
        fermi_function = 1/(1+exp(sp.selection_strength*(pop.fitnesses[i]-pop.fitnesses[j])))
        if rand() < fermi_function
            pop.strategies[i,:] = pop.strategies[j,:]
        end

        # Compute the strategy counts and push to pop.strategy_history
        push!(pop.strategy_history, [sum([pop.strategies[i,:] == strategy for i in 1:sp.N])/sp.N for strategy in sp.permitted_strategies])

    end

    function do_gossip!(pop::Population,sp::SimParams,gp::GossipParams)
        for _ in 1:gp.tau
            # Select three random individuals.
            # i will adopt j's view of k.
            i, j, k = rand(1:sp.N, 3)
            # for k in 1:sp.N
            #     i,j = rand(1:sp.N,2)
            pop.views[i,k] = pop.views[j,k]
            # An alternative version of this would be for i to adopt all of j's views.
    
        end
        # Compute the average strategy-wide views and append them
        # The transpose is because we need the view the population has of each strategy,
        # not the view each strategy has of the rest of the population
        avg_strategy_views = [mean(pop.views'[findall(row -> row == strategy, eachrow(pop.strategies)),:]) for strategy in sp.permitted_strategies]
        avg_view = mean(pop.views)
        push!(pop.views_history, vcat(avg_strategy_views, avg_view))
    end

    # Following are several functions that return useful averages
    function get_current_views(pop::Population,sp::SimParams)
        return [mean(pop.views'[findall(row -> row == strategy, eachrow(pop.strategies)),:]) for strategy in sp.permitted_strategies]
    end
    
    function get_current_average_fitnesses(pop::Population,sp::SimParams)
        return [mean(pop.fitnesses[findall(row -> row == strategy, eachrow(pop.strategies))]) for strategy in sp.permitted_strategies]
    end

    function get_frequencies(pop::Population,sp::SimParams)
        return [count([all(row .== strategy) for row in eachrow(pop.strategies)])/sp.N for strategy in sp.permitted_strategies]
    end

    function mutate_strategy!(pop::Population,sp::SimParams)

        mutation_mask = rand(sp.N) .< sp.strategy_mutation_rate
        if sum(mutation_mask) > 0
            pop.strategies[mutation_mask, :] .= hcat(sp.permitted_strategies[rand(1:end, sum(mutation_mask)), :]...)'
        end
    end

    function evolve!(pop::Population, sp::SimParams,gp::GossipParams)
        # Default evolution loop
        interact!(pop,sp)
        update_fitnesses!(pop,sp)
        update_views!(pop,sp)
        do_gossip!(pop,sp,gp)
        copy_strategy!(pop,sp)
        mutate_strategy!(pop,sp)
    end

    function evolve_modified_loop!(pop::Population, sp::SimParams,gp::GossipParams)
        # Modified evolution loop that does view updating first
        update_views!(pop,sp)
        do_gossip!(pop,sp,gp)
        interact!(pop,sp)
        update_fitnesses!(pop,sp)
        copy_strategy!(pop,sp)
        mutate_strategy!(pop,sp)
    end

    function gossip_only!(pop::Population, sp::SimParams,gp::GossipParams)
        # Alternative version of the above that only does gossip, primarily for debugging
        interact!(pop,sp)
        update_fitnesses!(pop,sp)
        update_views!(pop,sp)
        average_agreement = mean([mean(pop.views[i, :] .== pop.views[j, :]) for i in 1:sp.N, j in 1:sp.N])
        do_gossip!(pop,sp,gp)
        average_agreement = mean([mean(pop.views[i, :] .== pop.views[j, :]) for i in 1:sp.N, j in 1:sp.N])
    end

end