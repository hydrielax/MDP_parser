import numpy as np
import strategies as strategy_module


class MDP:

    def __init__(self):
        self.actions_labels = []
        self.states_labels = []
        self.states_id = {}
        self.nb_states = 0
        self.add_actions([None])
        self.rewards = []

    def add_state(self, label: str, reward: int = 0):
        if self.states_id.get(label, None) is not None:
            raise Exception(f"State {label} is defined twice.")
        self.states_labels.append(label)
        self.nb_states += 1
        self.states_id[label] = self.nb_states - 1
        self.rewards.append(reward)

    def add_actions(self, actions: list[str | None]) -> None:
        """Create the actions list."""
        self.actions_labels += actions
        self.actions_id = {action: i for i, action in enumerate(self.actions_labels)}
        self.nb_actions = len(self.actions_labels)
        self.probas = np.zeros((self.nb_states, self.nb_actions, self.nb_states))

    def update_proba(self, source_state: str, target_state: str, action: str | None, proba: float) -> None:
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

    def normalize(self):
        """transform transitions weights to probas"""
        S = np.sum(self.probas, axis=(2))
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                if S[i, j]:
                    for k in range(self.nb_states):
                        self.probas[i, j, k] = self.probas[i, j, k] / S[i, j]

    def actions_from(self, current_state: int) -> list[int]:
        """Returns the list of actions available from a state."""
        possible_actions = []
        for action in range(self.nb_actions):
            if np.sum(self.probas[current_state][action]):
                possible_actions.append(action)
        return possible_actions

    def check_actions_coherence(self):
        """Check that no state has transitions with and without actions"""
        for s in range(self.nb_states):
            possible_actions = self.actions_from(s)
            if 0 in possible_actions and len(possible_actions) != 1:
                raise Exception(
                    f"The state {self.states_labels[s]} has transitions with "
                    " AND without actions.")
            if len(possible_actions) == 0:
                raise Exception(
                    f"Aucune action possible depuis {self.states_labels[s]}")

    def build(self, initial_state):
        """Validate and build the network."""
        self.normalize()
        self.check_actions_coherence()
        if initial_state is None:
            self.initial_state = 0
        else:
            self.initial_state = self.states_id[initial_state]
    
    def is_mc(self) -> bool:
        """Indicates if the model is a Markov Chain (MC) or a Markov Decision
        Process (MDP)."""
        if len(self.actions_labels) == 1 and self.actions_labels[0] == None:
            return True
        return False

    def __repr__(self):
        """Returns a representation of the graph """
        return "\n".join([
            f"States: {self.states_labels}",
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

    def simulate(self, n_steps: int, strategy: str, verbose: int, initial=None) -> tuple[list[int], int | None]:
        """Simulate n_steps in the MDP and returns the list of steps followed."""
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

    def smc_mc_quantitatif(
        self,
        terminal_state_label: str,
        n_steps: int,
        eps: float,
        delta: float,
        verbose: int
    ) -> float:
        """
        Compute the probability to reach a state in a MC with Monte-Carlo.
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
            print(f"P(M|= <>(<={n_steps}) {terminal_state_label}) = {p}")
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
        Check if the probability to reach a state in a MC is greater than
        theta, with epsilon-confidence. Uses SPRT algorithm.
        """
        assert self.is_mc() == 'MC', """The model is not a Markov Chain."""
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
                print(f"P(M |= <>(≤ {n_steps}) {terminal_state_label}) ≤ {theta} (with ⍺={alpha:.2%} of confidence)")
            return False
        if logRm <= logB:
            if verbose >= 1:
                print(f"P(M |= <>(≤ {n_steps}) {terminal_state_label}) ≥ {theta} (with β={beta:.2%} of confidence)")
            return True
        if verbose >= 1:
            print(f"Not able to assert that P(M |= <>(≤ {n_steps}) {terminal_state_label}) ≥ {theta}")
        return None

    def rl_value_iteration(self, gamma: float, epsilon: float, iter_max: int, verbose: int):
        """
        Compute the best strategy to maximize the rewards,
        with value iterations algorithm.
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
        return Vprev, strat

    def rl_Q_learning(self, gamma: float, iter_max: int, verbose: int):
        """
        Compute the best strategy to maximize the rewards,
        with Q-learning algorithm.
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
        strat = { s: self.actions_labels[np.argmax(Q[s])]
                  for s in range(self.nb_states)
                }
        if verbose >= 1:
            print("Q = \n", Q)
            print("Strat = ", strat)
        return Q
