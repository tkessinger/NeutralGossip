using Revise
using Statistics
using PyPlot

includet("NeutralGossip.jl")


using .NeutralGossip

N = 10
glen = 100000
simlen = 10000

sp = SimParams(N,5.0,1.0,0.02,0.02,1.0,0.01,BitMatrix([1 0; 0 1]))
pop = Population(N)
gp = GossipParams(glen)

gc_array = zeros(Float64,simlen,5)

for i in 1:simlen
    println("$i")
    gossip_only!(pop,sp,gp)
    gc_array[i,:] = [(sum((pop.views .== 1) .& (pop.actions .== 1)) / sp.N^2),
    (sum((pop.views .== 1) .& (pop.actions .== 0)) / sp.N^2),
    (sum((pop.views .== 0) .& (pop.actions .== 1)) / sp.N^2),
    (sum((pop.views .== 0) .& (pop.actions .== 0)) / sp.N^2),
    sum((pop.views .== 1)) / sp.N^2]
end

figure()
plot(gc_array[:,1], label="GC", color="#FF3366")
plot(gc_array[:,2], label="GD", color="#3366FF")
plot(gc_array[:,3], label="BC", color="#33CC33")
plot(gc_array[:,4], label="BD", color="#FF9933")
plot(gc_array[:,5], label="good", color="#303030")
plt.legend(loc=2)
# println([mean(pop.fitnesses[findall(row -> row == strategy, eachrow(pop.strategies))]) for strategy in [[0,0],[1,1],[0,1]]])
# println([var(pop.fitnesses[findall(row -> row == strategy, eachrow(pop.strategies))]) for strategy in [[0,0],[1,1],[0,1]]])

# println([mean(pop.views[findall(row -> row == strategy, eachrow(pop.strategies)),:]) for strategy in [[0,0],[1,1],[0,1]]])
# println([mean(pop.views[:,findall(row -> row == strategy, eachrow(pop.strategies))]) for strategy in [[0,0],[1,1],[0,1]]])

# figure()
# plot(hcat(pop.strategy_history...)[1,:],label="ALLD",color="#FF42A1")
# plot(hcat(pop.strategy_history...)[2,:],label="ALLC",color="#479FF8")
# plot(hcat(pop.strategy_history...)[3,:],label="DISC",color="#1DB100")
# plt.legend(loc=2)


# figure()
# plot(hcat(pop.views_history...)[1,:],label="ALLD",color="#FF42A1")
# plot(hcat(pop.views_history...)[2,:],label="ALLC",color="#479FF8")
# plot(hcat(pop.views_history...)[3,:],label="DISC",color="#1DB100")
# plot(hcat(pop.views_history...)[4,:],label="total",color="#303030")
# title("views")
# plt.legend(loc=2)

# figure()
# plot(hcat(pop.action_history...)[1,:],label="ALLD",color="#FF42A1")
# plot(hcat(pop.action_history...)[2,:],label="ALLC",color="#479FF8")
# plot(hcat(pop.action_history...)[3,:],label="DISC",color="#1DB100")
# plot(hcat(pop.action_history...)[4,:],label="total",color="#303030")
# title("actions")
# plt.legend(loc=2)

# figure()
# plot(hcat(pop.views_history...)[1,:],label="ALLD",color="#FF42A1")
# plot(hcat(pop.views_history...)[2,:],label="ALLC",color="#479FF8")
# scatter(hcat(pop.action_history...)[4,:],hcat(pop.views_history...)[4,:],label="DISC",color="#1DB100")
# plot(hcat(pop.views_history...)[4,:],label="total",color="#303030")
# title("actions vs. views")
# plt.legend(loc=2)