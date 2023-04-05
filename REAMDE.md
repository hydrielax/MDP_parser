# MDP_Parser

A little parser for MDP algorithms.

## Installation

Install the dependencies:
```bash
pipenv install
```
or, if you don't use `pipenv`, run in a python virtualenv:
```bash
pip install numpy networkx scipy matplotlib
```

## Usage 

Run the following command:
```bash
python3 main.py <method> [--args]* <filename>
```

You can see all available methods using `python3 main.py -h`. The initial state is always defined for all methods as the first state defined in the file, but can be overridden with the `-I` argument.

A brief description of all methods:
| Method | Description | Arguments |
| -- | -- | -- |
| **print** | Show the MDP in the terminal | *None* |
| **draw** | Draw a graphical representation of the MDP and show it (or save it if a filename is given in the `save` parameter) | `save` |
| **simulate** | Run a simulation for $n$ steps in the given MDP, with a given strategy. The strategy must be defined as a function in the `strategies.py` file | `n_steps`, `strategy` |
| **check_mc** | Model Checking for Markov Chain | `terminal_state`, `n_steps` |
| **check_mc_rewards** | Model Checking for Markov Chain with rewards. | `n_steps`, `gamma` |
| **check_mdp** | Model Checking for MDP, with the min/max algorithm. | `terminal_state` |
| **SMC** | Quantitative SMC using Monte-Carlo | `terminal_state`, `n_steps`, `epsilon`, `delta` |
| **SMC_quali** | Qualitative SMC using SPRT | `terminal_state`, `n_steps`, `alpha`, `beta`, `epsilon`, `theta`, `iter_max` |
| **RL_VI** | *Reinforcement Learning* with *Value Iteration* | `gamma`, `epsilon`, `iter_max` |
| **RL_QL** | *Reinforcement Learning* with Q-learning algorithm | `gamma`, `iter_max` |

## Definition of a MDP

A MDP is defined in a file with a `.mdp` extension, following this syntax:
```python
# We defined 3 states S1, S2 and S3. S0 has a reward of 0, S1 and S2 a reward of 10.
States S0, S1:10, S2:10;
# We defined 3 available actions: a, b and c.
Actions a,b,c;
# S0 has a default transition to S1 with probability 50% and S2 with probability 50%
S0 -> 5:S1 + 5:S2;
# S1 has a transition for action b to itself with probability 20% and S0 with probability 80%
S1[b] -> 2: S1 + 8:S0;
# etc...
S1[a] -> 1:S2+3:S0+6:S1;
S2[c] -> 5:S0 + 5:S1;
S0[b] -> 7:S0 + 3:S2;
```

## Contribute

To update the grammar:
* install in your virtualenv `antlr4-tools` and `antlr4-python3-runtime`
* edit the `gram.g4` file
* compile the grammar:
    ```bash
    antlr4 -Dlanguage=Python3 gram.g4
    ```

To add a new method:
* Create a new method in the MDP class, in `mdp.py`
* Update the available methods on the second line of the `main` function in `main.py`
* Add a condition to call your method at the end of the `main` function

To add a new strategy:
* Create a new function in the `strategies.py` file, with the signature:
    ```python
    def my_strategy(history: list[int], mdp: 'MDP') -> int:
    ```
    where `history` is the list of indices of previous visited states, and `mdp` the MDP instance on which the method is executed. Your method should returns the id of the chosen action for this history.
* Then call a method with the name of your function for the argument `--strategy`
