# A3C Learning Notes

### Denotation:

V(s) --   value function

π(s) --   policy

R = γ(r) -- Discounted Reward

A = Q(s,a) - V(s) -- Advantage

A = R - V(s)  -- Advantage Estimate


### Deep learning codes structure:

* AC_Network  --  This class contains all the Tensorflow ops to create the networks themselves.

* Worker  --  This class contains a copy of AC_Network, an environment class, as well as all the logic for interacting with the environment, and updating the global network.

### Working process:

1. Worker reset to global network

2. Worker interacts with env

3. Worker calculate value and policy loss

4. Worker gets gradient from loss

5. Worker updates global network with gradients

- repeats the process
