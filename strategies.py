import random as rd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp import MDP


def ask_user(history: list[int], mdp: 'MDP') -> int:
    """Renvoie une action en fonction de l'historique."""
    action = None
    possible_actions = [mdp.actions_labels[a] for a in mdp.actions_from(history[-1])]
    while not action in possible_actions:
        action = input(f"\tChoisissez l'action parmis {possible_actions} : ")
    return mdp.actions_id[action]

def random(history: list[int], mdp: 'MDP') -> int:
    """Renvoie une action en fonction de l'historique."""
    possible_actions = [mdp.actions_labels[a] for a in mdp.actions_from(history[-1])]
    action = rd.choice(possible_actions)
    return mdp.actions_id[action]
