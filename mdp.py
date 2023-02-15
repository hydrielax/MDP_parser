import numpy as np

class MDP:

    def add_states(self, states: list[str]) -> None:
        self.states = states
        self.states_id = {state: i for i, state in enumerate(states)}

    def add_actions(self, actions: list[str]) -> None:
        self.actions = [None] + actions
        self.actions_id = {action: i for i, action in enumerate(self.actions)}
        self.probas = np.zeros((len(self.states), len(self.actions), len(self.states)))

    def update_proba(self, source_state: str, target_state: str, action: str, proba: float) -> None:
        source_id = self.states_id[source_state]
        target_id = self.states_id[target_state]
        action_id = self.actions_id[action]
        self.probas[source_id, action_id, target_id] = proba

    def __repr__(self):
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
