import numpy as np
import strategies as strategy_module


class MDP:

    def __init__(self):
        self.actions_labels = []

    def add_states(self, states: list[str]) -> None:
        """Create the states list."""
        self.states_labels = states
        self.states_id = {state: i for i, state in enumerate(states)}
        self.nb_states = len(self.states_labels)
        self.add_actions([None])

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

    def simulate(self, n_steps: int, strategy: str, verbose: int) -> list[int]:
        """Simulate n_steps in the MDP and returns the list of steps followed."""
        strategy_func = getattr(strategy_module, strategy)
        if verbose >= 1:
            label = self.states_labels[self.initial_state]
            print(f"Start simulation from {label} with {n_steps} steps...")
        path = [self.initial_state]
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
        return path

    def smc(
        self,
        terminal_state_label: str,
        n_steps: int,
        eps: float,
        delta: float,
        verbose: int
    ) -> float:
        terminal_state = self.states_id[terminal_state_label]
        N = int(np.ceil((np.log(2) - np.log(delta)) / (4*eps**2)))
        if verbose >= 1:
            print(f"N = {N}")
        count = 0
        for _ in range(N):
            path = self.simulate(n_steps, 'random', verbose-1)
            if path[-1] == terminal_state:
                count += 1
        p = count / N
        if verbose >= 1:
            print(f"P(M|= <>(<={n_steps}) {terminal_state_label}) = {p}")
        return p

    def sprt(
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
        terminal_state = self.states_id[terminal_state_label]
        logA, logB = np.log(((1 - beta) / alpha, beta / (1 - alpha)))
        gamma1, gamma0 = theta - eps, theta + eps
        logRm = (logA + logB) / 2
        dm = 0
        m = 0
        while logB < logRm and logRm < logA and m < iter_max:
            path = self.simulate(n_steps, 'random', verbose-1)
            if path[-1] == terminal_state:
                dm += 1
            logRm = (dm * (np.log(gamma1) - np.log(gamma0)) +
                     (m - dm) * (np.log(1 - gamma1) - np.log(1 - gamma0)))
            m += 1
        if logRm >= logA:
            res = False
        elif logRm <= logB:
            res = True
        else:
            res = None
        if verbose >= 1:
            print(f"Résultat : {res}")
        return res
