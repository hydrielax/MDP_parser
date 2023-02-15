import numpy as np

class MDP:

    def add_states(self, states: list[str]) -> None:
        """Create the states list."""
        assert not hasattr(self, 'states'), "States already defined."
        self.states = states
        self.states_id = {state: i for i, state in enumerate(states)}

    def add_actions(self, actions: list[str]) -> None:
        """Create the actions list."""
        assert not hasattr(self, 'actions'), "Actions already defined."
        self.actions = [None] + actions
        self.actions_id = {action: i for i, action in enumerate(self.actions)}
        self.probas = np.zeros((len(self.states), len(self.actions), len(self.states)))

    def update_proba(self, source_state: str, target_state: str, action: str, proba: float) -> None:
        """Update a proba from one source state to another target state through an action.

        Args:
            source_state (str): The source state label
            target_state (str): The target state label
            action (str): The action label for this edge
            proba (float): The probability to take this edge for this action
        """
        assert source_state in self.states, f"Source state {source_state} is not declared."
        assert target_state in self.states, f"Target state {target_state} is not declared."
        assert action in self.actions, f"Action {action} is not declared."
        source_id = self.states_id[source_state]
        target_id = self.states_id[target_state]
        action_id = self.actions_id[action]
        self.probas[source_id, action_id, target_id] = proba

    def __repr__(self):
        """Returns a representation of the graph """
        return "\n".join([
            f"States: {self.states}",
            f"Actions: {self.actions}",
            *[
                (f"{state} [{action}] -> " if action else f"{state} -> ")
                + str({
                    s: w 
                    for s, w in zip(
                        self.states,
                        self.probas[self.states_id[state], self.actions_id[action]])
                    if w
                })[1:-1]
                for action in self.actions
                for state in self.states
                if np.any(self.probas[self.states_id[state], self.actions_id[action]])
            ]
        ])

    def normalize(self):
        """transform transitions weights to probas"""
        S = np.sum(self.probas,axis=(2))
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                if S[i][j] != 0:
                    for k in range(len(self.probas[i][j])):
                        self.probas[i][j][k] = self.probas[i][j][k]/S[i][j]

    def actionsFrom(self,current_state):
        possible_actions = []
        for action in self.actions:
            if np.sum(self.probas[current_state][action]) != 0:
                possible_actions.append(action)
        return possible_actions

    def simulate(self,n_steps=3,strategy=None):
        path = [self.states[0]] #L'état intitial est le premeir état par défaut
        for step in range(n_steps):
            current_state = path[-1]
            #Sélection de l'action
            possible_actions = 
            if strategy is None:
                correct = False
                print(f"Choisissez l'action à réaliser dans l'état {current_state} parmis {self.actions}")
                while not correct:
                    action = input()
                    if action not in self.actions:
                        print("L'action entrée est incorrecte")
                    elif np.sum(self.probas[current_state][action]) == 0:
                        print("L'action entrée n'est pas disponible depuis cet état")
                    else:
                        correct = True
            else:
                action = strategy(current_state) #TODO : prendre en compte l'historique des actions.