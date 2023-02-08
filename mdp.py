import numpy as np

class MDP:

    def add_states(self, states: list[str]) -> None:
        self.states = states
        self.states_id = {state: i for i, state in enumerate(states)}

    def add_actions(self, actions: list[str]) -> None:
        self.actions = actions + [None]
        self.probas = {action: np.zeros((len(self.states), len(self.states)))
                       for action in self.actions}

    def update_proba(self, source_state: str, target_state: str, action: str, proba: float) -> None:
        source_id = self.states_id[source_state]
        target_id = self.states_id[target_state]
        self.probas[action][source_id, target_id] = proba
