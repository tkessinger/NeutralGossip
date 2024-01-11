# NeutralGossip

Developed by Kessinger and Kawakatsu for:
Kawakatsu M., Kessinger T.A., and Plotkin J.B. "A mechanistic model of gossip, repuations, and cooperation."
https://arxiv.org/pdf/2312.10821.pdf

The bulk of the simulation script is kept in NeutralGossip.jl, which allows the user to define main simulation parameters, then initialize and evolve a population of agents that rely on reputations to decide how to behave.

Trajectories generated for Figure 2 were produced via sample_trajectories.jl.
An even simpler implementation is available via simple_simulation.jl.
