import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import linprog

import strategies as strategy_module


class MDP:

    def __init__(self) -> None:
        self.actions_labels = []
        self.states_labels = []
        self.states_id = {}
        self.nb_states = 0
        self.add_actions([None])
        self.rewards = []

    def add_state(self, label: str, reward: int = 0) -> None:
        """Create a new state, with an associated reward.

        Args:
            label (str): The label of the new state.
            reward (int, optional): The reward of the state. Defaults to 0.

        Raises:
            Exception: raised if label already exists.
        """
        if self.states_id.get(label, None) is not None:
            raise Exception(f"State {label} is defined twice.")
        self.states_labels.append(label)
        self.nb_states += 1
        self.states_id[label] = self.nb_states - 1
        self.rewards.append(reward)
        self.probas = np.zeros((self.nb_states, self.nb_actions, self.nb_states))

    def add_actions(self, actions: list[str | None]) -> None:
        """Create multiple actions at once.

        Args:
            actions (list[str  |  None]): A list of action labels to add.
        """
        self.actions_labels += actions
        self.actions_id = {action: i for i, action in enumerate(self.actions_labels)}
        self.nb_actions = len(self.actions_labels)
        self.probas = np.zeros((self.nb_states, self.nb_actions, self.nb_states))

    def update_proba(
        self,
        source_state: str,
        target_state: str,
        action: str | None,
        proba: float
    ) -> None:
        """Update a proba from one source state to another target state through an action.

        Args:
            source_state (str): The source state label
            target_state (str): The target state label
            action (str): The action label for this edge
            proba (float): The probability to take this edge for this action
        """
        assert source_state in self.states_labels, f"Source state {source_state} is not declared."
        assert target_state in self.states_labels, f"Target state {target_state} is not declared."
        assert action in self.actions_labels, f"Action {action} is not declared."
        source_id = self.states_id[source_state]
        target_id = self.states_id[target_state]
        action_id = self.actions_id[action]
        assert self.probas[source_id, action_id, target_id]==0, (
            f"Transition from {self.states_labels[source_id]} "
            f"to {self.states_labels[target_id]} with action "
            f"{self.actions_labels[action_id]} already defined")
        self.probas[source_id, action_id, target_id] = proba

    def normalize(self) -> None:        
        """Transform transitions weights to probas"""
        S = np.sum(self.probas, axis=(2))
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                if S[i, j]:
                    for k in range(self.nb_states):
                        self.probas[i, j, k] = self.probas[i, j, k] / S[i, j]

    def actions_from(self, current_state: int) -> list[int]:
        """Returns the list of actions available from a state.

        Args:
            current_state (int): The state we want to know which actions are available from.

        Returns:
            list[int]: The list of indices of the available actions.
        """
        possible_actions = []
        for action in range(self.nb_actions):
            if np.sum(self.probas[current_state][action]):
                possible_actions.append(action)
        return possible_actions

    def check_actions_coherence(self) -> None:
        """Check that no state has transitions with and without actions

        Raises:
            Exception: _description_
            Exception: _description_
        """
        for s in range(self.nb_states):
            possible_actions = self.actions_from(s)
            if 0 in possible_actions and len(possible_actions) != 1:
                raise Exception(
                    f"The state {self.states_labels[s]} has transitions with "
                    " AND without actions.")
            if len(possible_actions) == 0:
                print(f"Aucune action possible depuis {self.states_labels[s]}. "
                      "Ajout d'une transition vers ce même état.")
                self.probas[s, None, s] = 1

    def build(self, initial_state_label: str | None = None) -> None:
        """Validate and build the MDP.

        Args:
            initial_state_label (str, optional): The initial state of the MDP, by default the first one.
        """
        self.normalize()
        self.check_actions_coherence()
        if initial_state_label is None:
            self.initial_state = 0
        else:
            self.initial_state = self.states_id[initial_state_label]

    def is_mc(self) -> bool:
        """Indicates if the model is a Markov Chain (MC) or a Markov Decision
        Process (MDP).

        Returns:
            bool: True if the model is a MC.
        """
        if len(self.actions_labels) == 1 and self.actions_labels[0] == None:
            return True
        return False

    def __repr__(self) -> str:
        """Returns a representation of the graph."""
        return "\n".join([
            f"States: {list(zip(self.states_labels, self.rewards)) if any(self.rewards) else self.states_labels}",
            f"Actions: {self.actions_labels}",
            *[
                (f"{state} [{action}] -> " if action else f"{state} -> ")
                + str({
                    s: w 
                    for s, w in zip(
                        self.states_labels,
                        self.probas[self.states_id[state], self.actions_id[action]])
                    if w
                })[1:-1]
                for action in self.actions_labels
                for state in self.states_labels
                if np.any(self.probas[self.states_id[state], self.actions_id[action]])
            ],
            f"Initial State: {self.states_labels[self.initial_state]}"
        ])

    def simulate(
        self,
        n_steps: int,
        strategy: str,
        verbose: int,
        initial: int | None = None
    ) -> tuple[list[int], int | None]:
        """Runs a simulation of the MDP.

        Args:
            n_steps (int): Number of steps to run.
            strategy (str): The strategy to use for action choices.
            verbose (int): The verbose level.
            initial (int, optional): The state to start simulation from. By default the initial state of the MDP.

        Returns:
            tuple[list[int], int | None]: The path of states, and the last action used.
        """
        if initial is None:
            initial = self.initial_state
        strategy_func = getattr(strategy_module, strategy)
        if verbose >= 1:
            label = self.states_labels[initial]
            print(f"Start simulation from {label} with {n_steps} steps...")
        path = [initial]
        action = -1
        for step in range(n_steps):
            if verbose >= 2:
                print(f"Step {step}")
                print(f"\tState : {self.states_labels[path[-1]]}")
            current_state = path[-1]
            # Sélection de l'action
            possible_actions = self.actions_from(current_state)
            if len(possible_actions) == 1:
                action = possible_actions[0]
            else:
                action = strategy_func(path, self)
            if verbose >= 2:
                print(f"\tAction : {self.actions_labels[action]}")
            # Tirage aléatoire de la transition
            next_state = np.random.choice(
                self.nb_states,
                p=self.probas[current_state, action])
            path.append(next_state)
        if verbose >= 1:
            print(f"Final State: {self.states_labels[path[-1]]}")
        return path, action
    
    def model_checking_mc(
        self,
        terminal_state_label: str,
        n_steps: int,
        verbose: int
    ) -> float:
        """Model checking algorithm for Markov Chains only.
        Compute the probability to reach a state from the initial state.

        Args:
            terminal_state_label (str): The state we want to access.
            n_steps (int): Max number of steps allowed. Use n=0 for infinity.
            verbose (int): The verbose level.

        Returns:
            float: The probability to access the state, ie P(I |= <>(<=n) T).
        """
        if verbose >= 1:
            print(f"Compute for {n_steps if n_steps > 0 else 'infinite'} steps")
        # reduce the probability matrix to 2 dim
        assert self.is_mc(), "The model is not a Markov Chain."
        P = self.probas[:, 0, :]
        # build the S1 set
        S1 = { self.states_id[terminal_state_label] }
        if not n_steps > 0:  # do not build S1 for finite number of steps
            s = 0
            while s < self.nb_states:
                if (not s in S1 and
                    all([P[s, t]==0 for t in range(self.nb_states) if not t in S1])
                ):
                    S1.add(s); s = -1
                s += 1
        # build the S0 set
        S0 = set()
        s = 0
        while s < self.nb_states:
            if (not s in S0 and
                not s in S1 and (
                P[s,s] == 1 or
                all([P[s, t]==0 for t in range(self.nb_states) if not t in S0 and t != s])
            )):
               S0.add(s); s = -1
            s += 1
        # calcul de S?
        S = set(range(self.nb_states))
        S = S.difference(S0, S1)
        # definition de A et B
        A = P[np.ix_(list(S), list(S))]
        b = np.array([np.sum(P[s, list(S1)]) for s in S])
        # cas infini : on résout y = Ay + b
        if n_steps > 0:
            y = np.zeros(len(S))
            for _ in range(n_steps):
                y = A.dot(y) + b
        else:
            y = np.linalg.solve(np.eye(len(S)) - A, b)
        # retourne le résultat
        if self.initial_state in S1:
            if verbose >= 1:
                print(f"P({self.states_labels[self.initial_state]} |= <> {terminal_state_label}) = 1")
            return 1
        if self.initial_state in S0:
            if verbose >= 1:
                print(f"P({self.states_labels[self.initial_state]} |= <> {terminal_state_label}) = 0")
            return 0
        res = y[list(S).index(self.initial_state)]
        if verbose >= 1:
            print(f"P({self.states_labels[self.initial_state]} |= <> {terminal_state_label}) = {res}")
        return res

    def model_checking_mc_rewards(
        self,
        n_steps: int,
        gamma: float,
        verbose: int
    ) -> list[float]:
        """Model checking algorithm for Markov Chains only, which computes the
        rewards associated for each terminal state.

        Args:
            n_steps (int): Max number of steps allowed. Use n=0 for infinity.
            gamma (float): The reduction factor to ensure convergence.
            verbose (int): The verbose level.

        Returns:
            list[float]: List of gains for each state.
        """
        if verbose >= 1:
            print(f"Compute for {n_steps if n_steps > 0 else 'infinite'} steps")
        # reduce the probability matrix to 2 dim
        assert self.is_mc(), "The model is not a Markov Chain."
        P = self.probas[:, 0, :]
        # 
        r = np.array(self.rewards)
        # cas infini : on résout y = Ay + b
        if n_steps > 0:
            y = np.zeros(self.nb_states)
            for _ in range(n_steps):
                y = gamma * P.dot(y) + r
        else:
            try:
                y = np.linalg.solve(np.eye(self.nb_states) - gamma*P, r)
            except Exception:
                raise Exception("Model diverges")
        # retourne le résultat
        if verbose >= 1:
            for s in range(self.nb_states):
                print(f"G({self.states_labels[s]}) = {y[s]:.2f}")
        return list(y)

    def model_checking_mdp(
        self,
        terminal_state_label: str,
        verbose: int
    ) -> list[float]:
        """Check the accessibility of a state with min/max algo for MDP.

        Args:
            terminal_state_label (str): The state to reach.
            verbose (int): The verbose level.

        Returns:
            list[float]: The probabilities to access the terminal state for each state as initial state.
        """
        terminal_state = self.states_id[terminal_state_label]
        # create the matrix A
        A = np.zeros((self.nb_states * (self.nb_actions + 2), self.nb_states))
        for s in range(self.nb_states):
            if s != terminal_state:
                for a in range(self.nb_actions):
                    for t in range(self.nb_states):
                        i, j = s * (self.nb_actions + 2) + a, t
                        if t == s:
                            A[i, j] = 1 - self.probas[s, a, t]
                        else:
                            A[i, j] = - self.probas[s, a, t]
                A[s * (self.nb_actions + 2) + self.nb_actions, s] = 1
                A[s * (self.nb_actions + 2) + self.nb_actions + 1, s] = -1
        # create matrix b
        b = np.zeros(self.nb_states * (self.nb_actions + 2))
        for s in range(self.nb_states):
            if s != terminal_state:
                for a in range(self.nb_actions):
                    b[s * (self.nb_actions + 2) + a] = self.probas[s, a ,terminal_state]
                b[s * (self.nb_actions + 2) + self.nb_actions] = 0
                b[s * (self.nb_actions + 2) + self.nb_actions + 1] = -1
        # Solving A.x <= b
        result_max = linprog(np.ones(self.nb_states), A_ub=-A, b_ub=-b)
        assert result_max['success'], "The algorithme did not converge or didn't find any solution for the MAX"
        x_max = result_max['x']
        x_max[terminal_state] = 1
        if verbose >= 1:
            print(f"A solution was found with maximal probabilities for each starting state:")
            print(x_max)
        return x_max

    def smc_mc_quantitatif(
        self,
        terminal_state_label: str,
        n_steps: int,
        eps: float,
        delta: float,
        verbose: int
    ) -> float:
        """
        Statistical Model Checking: computes an approximation of P(I |= <>(<=n) T).
        This method uses the Monte-Carlo algorithm, and works only for Markov Chains.

        Args:
            terminal_state_label (str): The state T to reach.
            n_steps (int): The max number of steps allowed. Use n=0 for infinity.
            eps (float): The precision coefficient.
            delta (float): The error rate coefficient.
            verbose (int): The verbose level.

        Returns:
            float: Returns an estimation of the probability.
        """
        assert self.is_mc(), """The model is not a Markov Chain."""
        terminal_state = self.states_id[terminal_state_label]
        N = int(np.ceil((np.log(2) - np.log(delta)) / (4*eps**2)))
        if verbose >= 1:
            print(f"N = {N}")
        count = 0
        for _ in range(N):
            path, _ = self.simulate(n_steps, 'random', verbose-1)
            if terminal_state in path:
                count += 1
        p = count / N
        if verbose >= 1:
            print(f"P({self.states_labels[self.initial_state]} |= <>(<={n_steps}) {terminal_state_label}) ≈ {p}")
        return p

    def smc_mc_qualitatif(
        self,
        terminal_state_label: str,
        n_steps: int,
        alpha: float,
        beta: float,
        eps: float,
        theta: float,
        iter_max: int,
        verbose: int
    ) -> bool | None:
        """
        Statiscal Model Checking for MDC.
        Check if the probability to reach a state in a MC is greater than
        theta, with epsilon-confidence. Uses the SPRT algorithm.

        Args:
            terminal_state_label (str): The state T to reach.
            n_steps (int): Max number of steps allowed. Use n=0 for infinity.
            alpha (float): alpha-confidence coefficient.
            beta (float): beta-confidence coefficient.
            eps (float): The precision coefficient.
            theta (float): The upper born to test.
            iter_max (int): Max number of iteration.
            verbose (int): The verbose level.

        Returns:
            bool | None: True if estimation greater than theta, False if lesser than theta, None if uncertain.
        """
        assert self.is_mc(), "The model is not a Markov Chain."
        terminal_state = self.states_id[terminal_state_label]
        logA, logB = np.log(((1 - beta) / alpha, beta / (1 - alpha)))
        gamma1, gamma0 = theta - eps, theta + eps
        logRm = (logA + logB) / 2
        dm = 0
        m = 0
        while logB < logRm and logRm < logA and m < iter_max:
            path, _ = self.simulate(n_steps, 'random', verbose-1)
            if path[-1] == terminal_state:
                dm += 1
            logRm = (dm * (np.log(gamma1) - np.log(gamma0)) +
                     (m - dm) * (np.log(1 - gamma1) - np.log(1 - gamma0)))
            m += 1
        if logRm >= logA:
            if verbose >= 1:
                print(f"P(M |= <>(≤ {n_steps}) {terminal_state_label}) ≤ {theta} (with ⍺={alpha:.2%} for confidence)")
            return False
        if logRm <= logB:
            if verbose >= 1:
                print(f"P(M |= <>(≤ {n_steps}) {terminal_state_label}) ≥ {theta} (with β={beta:.2%} for confidence)")
            return True
        if verbose >= 1:
            print(f"Not able to assert that P(M |= <>(≤ {n_steps}) {terminal_state_label}) ≥ {theta}")
        return None

    def rl_value_iteration(
        self,
        gamma: float,
        epsilon: float,
        iter_max: int,
        verbose: int
    ) -> tuple[list[float], dict[str, str]]:
        """Reinforcement Learning with Value Iterations algorithm.
        Compute the best positional strategy to maximize rewards.

        Args:
            gamma (float): The convergence factor.
            epsilon (float): The precision factor.
            iter_max (int): Max number of iterations.
            verbose (int): The verbose level.

        Returns:
            tuple[list[float], dict[str, str]]: last Vn vector and the strategy found.
        """
        if verbose >= 1:
            print("Begin optimization...")
        Vprev = np.inf * np.ones(self.nb_states)
        Vnext = np.zeros(self.nb_states)
        i = 0
        while np.linalg.norm(Vnext - Vprev) >= epsilon and i < iter_max:
            Vprev = Vnext.copy()
            for sid in range(self.nb_states):
                Vnext[sid] = np.max([
                    self.rewards[sid] + gamma * np.sum(self.probas[sid,a,:] * Vprev)
                    for a in self.actions_from(sid)
                ])
            i += 1
        if verbose >= 1:
            print(f"Computed in {i} steps.")
            print("Compute strategy...")
        strat = {}
        for sl, sid in self.states_id.items():
            possible_actions = self.actions_from(sid)
            strat[sl] = self.actions_labels[possible_actions[np.argmax([
                self.rewards[sid] + gamma * np.sum(self.probas[sid,a,:] * Vprev)
                for a in possible_actions
            ])]]
        # affichage
        if verbose >= 1:
            print("Vn = ", [f"{x:.2f}" for x in Vprev])
            print("Strat = ", strat)
        return list(Vprev), strat

    def rl_Q_learning(
        self,
        gamma: float,
        iter_max: int,
        verbose: int
    ) -> tuple[np.ndarray, dict[str, str]]:
        """Reinforcement Learning with Q-learning.
        Computes the best strategy to maximize rewards.

        Args:
            gamma (float): The convergence factor.
            iter_max (int): Max number of iterations.
            verbose (int): The verbose level.

        Returns:
            tuple[any, dict[str, str]]: The Q matrix and the strategy found.
        """
        Q = np.zeros((self.nb_states, self.nb_actions))
        Qnext = Q.copy()
        alpha = np.ones((self.nb_states, self.nb_actions))
        for _ in range(iter_max):
            st = np.random.randint(self.nb_states)
            path, at = self.simulate(1, 'random', verbose-1, initial=st)
            st1 = path[-1]
            rt = self.rewards[st]
            Q = Qnext.copy()
            deltat = rt + gamma * np.max(Q[st1]) - Q[st][at]
            Qnext[st][at] = Q[st][at] + deltat / alpha[st, at]
            alpha[st, at] += 1
        Q = Qnext.copy()
        strat = { self.states_labels[s]: self.actions_labels[np.argmax(Q[s])]
                  for s in range(self.nb_states)
                }
        if verbose >= 1:
            print("Q = \n", Q)
            print("Strat = ", strat)
        return Q, strat

    def draw_graph(self, name: str, save = False) -> None:
        """Draw a representation of the MDP with NetworkX.
        You can indicate a file name to save it.

        Args:
            output_file (str | None, optional): The name of the file to save in.
        """
        G = nx.MultiDiGraph()
        nodes = []
        labels = []
        edge_labels = []
        colors = []
        d = {}
        for s in range(self.nb_states):
            nodes.append(s)
            labels.append(self.states_labels[s])
            colors.append("cyan")
            for a in self.actions_from(s):
                if a == 0:
                    for t in range(self.nb_states):
                        if self.probas[s,a,t] != 0:
                            G.add_edge(s, t)
                            edge_labels.append(1)
                            d[(s,t)] = str(1)
                else:
                    nodes.append(f"{s}_{a}")
                    labels.append(self.actions_labels[a])
                    colors.append("yellow")
                    G.add_edge(s,f"{s}_{a}")
                    edge_labels.append("")
                    d[(s,f"{s}_{a}")] = ""
                    for t in range(self.nb_states):
                        if self.probas[s,a,t] != 0:
                            G.add_edge(f"{s}_{a}", t)
                            edge_labels.append(str(self.probas[s,a,t]))
                            d[(f"{s}_{a}",t)] = str(np.round(self.probas[s,a,t],2))
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, nodelist=nodes, pos=pos, node_color=colors) # type: ignore
        nx.draw_networkx_labels(G, pos, {n:labels[i] for i, n in enumerate(nodes)})
        nx.draw_networkx_edges(G, pos, edgelist=G.edges, connectionstyle="arc3,rad=0.1", arrows=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=d)
        plt.title(f"Graph of file {name}.mdp")
        if save:
            plt.savefig(f"images/{name}.png")
        else:
            plt.show()
