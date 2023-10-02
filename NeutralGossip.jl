
module NeutralGossip

    using Statistics


    export evolve!, Population, SimParams, GossipParams
    export get_current_views
    export do_gossip!, gossip_only!
    export interact!, update_views!

    struct SimParams
        N::Int64
        b::Float64
        c::Float64
        assessment_error::Float64
        performance_error::Float64
        selection_strength::Float64
        strategy_mutation_rate::Float64
        social_norm::BitMatrix
        permitted_strategies::Vector{BitVector}
    end

    function SimParams(N::Int64, b::Float64, c::Float64, assessment_error::Float64,
        performance_error::Float64, selection_strength::Float64,
        strategy_mutation_rate::Float64, social_norm::BitMatrix)
        return SimParams(N, b, c, assessment_error, performance_error,
                selection_strength, strategy_mutation_rate, social_norm, [BitVector(strat) for strat in [[0,1]]])
        end

    mutable struct GossipParams
        
        tau::Int64
        gossip_history::Array{BitMatrix}

        # Constructor
        function GossipParams(tau::Int64)
            new(tau,BitMatrix[])
        end

    end

    mutable struct Population
        strategies::BitMatrix
        views::BitMatrix
        actions::BitMatrix
        fitnesses::Vector{Float64}
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

        # for i in 1:size(pop.views, 1), j in 1:size(pop.views, 2)
        #     view_i_j = pop.views[i, j]
        #     view_label = view_i_j == 0 ? "bad" : "good"
        #     strategy_i = pop.strategies[i, :]
        #     strategy_bits = join([strategy_i[k] == 0 ? "D" : "C" for k in 1:2])
        #     intention_i_j = strategy_i[view_i_j + 1] ? "cooperate" : "defect"
        #     action_i_j = pop.actions[i,j] ? "cooperate" : "defect"
            
        #     println("For i=$i and j=$j:")
        #     println("i's view of j: $view_i_j ($view_label)")
        #     println("i's strategy bits: $strategy_bits")
        #     println("i intended to: $intention_i_j")
        #     println("i did: $action_i_j")
        # end 

        # Define an action error mask; this will tell us who makes mistakes
        action_error_mask = rand(sp.N,sp.N) .< sp.performance_error

        # for i in 1:sp.N, j in 1:sp.N
        #     if action_error_mask[i, j]
        #         intention = pop.strategies[i, 2] == 1 ? "cooperate" : "defect"
        #         action = pop.actions[i, j] == 1 ? "cooperates" : "defects"
        #         println("Mistake: Individual $i accidentally $action with individual $j.")
        #         println("   - Intention: $i intended to $intention with $j.")
        #         println("   - New Action: $i $action with $j.")
        #     end
        # end

        # Individuals who make mistakes defect (action error is asymmetric)
        # i.e., the only way to cooperate (1) is to intend to do so and NOT make a mistake
        pop.actions .= pop.actions .& .!action_error_mask

        avg_strategy_action = [mean(pop.actions[findall(row -> row == strategy, eachrow(pop.strategies)),:]) for strategy in sp.permitted_strategies]
        avg_action = mean(pop.actions)
        push!(pop.action_history, vcat(avg_strategy_action, avg_action))

    end

    function update_views!(pop::Population,sp::SimParams)
        
        # Each observer j chooses a random recipient k
        k = rand(1:sp.N, sp.N)

        # j examines what each i in the population did to k...
        i_actions = [pop.actions[i, k[i]] for i in 1:sp.N]

        # ...and what j's own view of k was
        j_views = [pop.views[j, k[j]] for j in 1:sp.N]

        # j's new view of i is given by the corresponding social norm index
        # The first index of the social norm is j's view of k;
        # the second, i's action toward k
        # The +1 is needed to map 0 and 1 to array indices 1 and 2
        pop.views .= sp.social_norm[j_views .+ 1, i_actions .+ 1]

        # Generate a view error mask with specified error probability
        view_error_mask = rand(sp.N, sp.N) .< sp.assessment_error

        # Introduce errors in pop.views using the XOR (^) operation;
        # if rand() is smaller than assessment_error, flip i's view of j
        pop.views = pop.views .⊻ view_error_mask

        # Obtain the strategy-wise and population-wide average views
        avg_strategy_views = [mean(pop.views[findall(row -> row == strategy, eachrow(pop.strategies)),:]) for strategy in sp.permitted_strategies]
        avg_view = mean(pop.views)
        push!(pop.views_history, vcat(avg_strategy_views, avg_view))
        # push!(pop.views_history, [avg_view])

    end

    function update_fitnesses!(pop::Population,sp::SimParams)

        # Individual i's fitness is:
        # benefit * everyone who cooperated with i
        # - cost * everyone whom i cooperated with
        pop.fitnesses = ((sp.b * sum(pop.actions,dims=1)' - sp.c * sum(pop.actions,dims=2))[:])/sp.N

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
            i, j, k = rand(1:sp.N, 3)
            # println("$i's views were $(pop.views[i,:])")
            # println("$j's views were $(pop.views[j,:])")
            # println("$k was chosen")
            pop.views[i,k] = pop.views[j,k]
            # println("views are now $(pop.views)")
            # push!(gp.gossip_history,pop.views)
            # avg_strategy_views = [count(pop.views[findall(row -> row == strategy, eachrow(pop.strategies)),:]) for strategy in sp.permitted_strategies]
            # avg_view = mean(pop.views)
            # push!(pop.views_history, [avg_view])
            #push!(pop.views_history, vcat(avg_strategy_views, avg_view))
    
        end
        avg_strategy_views = [mean(pop.views[findall(row -> row == strategy, eachrow(pop.strategies)),:]) for strategy in sp.permitted_strategies]
        avg_view = mean(pop.views)
        push!(pop.views_history, vcat(avg_strategy_views, avg_view))
    end

    function get_current_views(pop::Population,sp::SimParams)
        return [mean(pop.views[findall(row -> row == strategy, eachrow(pop.strategies)),:]) for strategy in [[0,0],[1,1],[0,1]]]
    end
    
    function get_current_average_fitnesses(pop::Population,sp::SimParams)
        return [mean(pop.fitnesses[findall(row -> row == strategy, eachrow(pop.strategies))]) for strategy in [[0,0],[1,1],[0,1]]]
    end

    function mutate_strategy!(pop::Population,sp::SimParams)

        mutation_mask = rand(sp.N) .< sp.strategy_mutation_rate
        if sum(mutation_mask) > 0
            pop.strategies[mutation_mask, :] .= hcat(sp.permitted_strategies[rand(1:end, sum(mutation_mask)), :]...)'
        end
    end

    function evolve!(pop::Population, sp::SimParams,gp::GossipParams)
        interact!(pop,sp)
        update_fitnesses!(pop,sp)
        update_views!(pop,sp)
        do_gossip!(pop,sp,gp)
        copy_strategy!(pop,sp)
        mutate_strategy!(pop,sp)
        # push!(pop.strategy_history, pop.strategies[:,:])
    end

    function gossip_only!(pop::Population, sp::SimParams,gp::GossipParams)
        # Alternative version of the above that only does gossip
        interact!(pop,sp)
        # println("good view, cooperate: $(sum((pop.views .== 1) .& (pop.actions .== 1)) / sp.N^2)")
        # println("good view, defect: $(sum((pop.views .== 1) .& (pop.actions .== 0)) / sp.N^2)")
        # println("bad view, cooperate: $(sum((pop.views .== 0) .& (pop.actions .== 1)) / sp.N^2)")
        # println("bad view, defect: $(sum((pop.views .== 0) .& (pop.actions .== 0)) / sp.N^2)")
        update_fitnesses!(pop,sp)
        update_views!(pop,sp)
        # average_agreement = mean([mean(pop.views[i, :] .== pop.views[j, :]) for i in 1:sp.N, j in 1:sp.N])
        # average_g = mean(pop.views)
        # println("before gossip: $average_agreement, $average_g")

        do_gossip!(pop,sp,gp)
        # average_agreement = mean([mean(pop.views[i, :] .== pop.views[j, :]) for i in 1:sp.N, j in 1:sp.N])
        # println("after gossip: $average_agreement, $average_g")
    end

end