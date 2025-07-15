using MHDSim
using LinearAlgebra

# Basic parameters
n_el = 4 # Number of electrodes
n_mag = 4 # Number of electromagnets
σ = 5 # Conductivity (S/m)
Δt = 0.25 # Timestep (s)

path_to_fields = joinpath(@__DIR__, "fields", "8mm_4electrodes_full") # Path to the folder with the fields

data = SimData(path_to_fields, n_el, n_mag, σ, Δt) # Takes a lot of time 

state = SimState(data) # Initial state of the simulation

## Points for evaluating the solution
xs = collect(LinRange(-50e-3, 50e-3, 32)) # x coordinates of the points where the fields will be evaluated
ys = xs # y coordinates of the points where the fields will be evaluated
zs = [8e-3] # z coordinates of the points where the fields will be evaluated

peh = MHDSim.PointEvalHandler(data, xs, ys, zs)


## Running the simulation for ten steps
cache = nothing
for i in 1:10

    ϕ = zeros(n_el) # Electrode commands
    ψ = zeros(n_mag) # Coil commands

    state, cache = solve_step(data, state, ϕ , ψ, cache)

    vx, vy, vz =  measure_velocity(state, data, peh) # Measure velocity at the points

    p = measure_pressure(state, data, peh) # Measure pressure at the points

    # Do something with the velocity
    # Do something with the pressure

end
