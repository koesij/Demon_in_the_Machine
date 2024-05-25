# Demon in the Machine

## The paper
This is an implementation of Whitelam's paper, "Demon in the Machine: Learning to Extract Work and Absorb Entropy from Fluctuating Nanosystems" in python. [1]
The paper reports on the use of Monte Carlo and genetic algorithms to train neural networks for feedback control in fluctuating nanosystems.

You can see his original code written in C from the following link.


https://github.com/swhitelam/demon


One demonstration of this paper that astonished me was the Magnetization Reversal with Negative Entropy Production. The study demonstrates that magnetization reversal in the Ising model can be achieved with negative entropy production, implying that work can be extracted from the thermal bath using optimally learned trajectories.

## Important Notice

Please note that the code provided here can be hard to interpret. Detailed explanations and documentation will be added soon to help clarify its functionality and usage.

### Heavy Simulation Warning

The simulation performed by this code is computationally intensive and may take several days to complete, depending on your system's capabilities. It is recommended to use multiprocessing to leverage multiple CPUs for improved performance. I have included a function that leverages multiprocessing. Functions related to this feature have names containing 'parallel'.

Thank you for your understanding and patience.

## Reference
[1] Whitelam, Stephen. "Demon in the machine: learning to extract work and absorb entropy from fluctuating nanosystems." Physical Review X 13.2 (2023): 021005.
