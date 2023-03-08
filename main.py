import argparse
from typing import Literal

from antlr4 import FileStream, CommonTokenStream, ParseTreeWalker

from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser
from mdp import MDP


class gramMDPListener(gramListener):

    def __init__(self):
        self.mdp = MDP()
        
    def enterDefstates(self, ctx):
        self.mdp.add_states([str(x) for x in ctx.ID()])

    def enterDefactions(self, ctx):
        self.mdp.add_actions([str(x) for x in ctx.ID()])

    def enterTransact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        act = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        for target, weight in zip(ids, weights):
            self.mdp.update_proba(dep, target, act, weight)
        
    def enterTransnoact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        for target, weight in zip(ids, weights):
            self.mdp.update_proba(dep, target, None, weight)


def main():
    # arg parse
    parser = argparse.ArgumentParser(
        prog = 'python main.py',
        description = 'Markov Decision Process Chains Analyser')
    parser.add_argument('method', choices=['draw', 'simulate', 'smc'],
                        help="The method to use: draw, simulate, or SMC")
    parser.add_argument('filename',
                        help="The name of the file.")
    parser.add_argument('-i', '--initial-state', default=None,
                        help="Initial state label for the simulation, defaults to the first one.")
    parser.add_argument('-n', '--number-of-steps', type=int, dest='n_steps', default=5,
                        help="Number of steps to apply for the simulation, defaults to 5.")
    parser.add_argument('-s', '--strategy',  default='ask_user',
                        help="Strategy to use for the simulation, defaults to 'ask_user'.")
    parser.add_argument('-t', '--terminal-state', default=None,
                        help="The terminal state, used for SMC only. Required.")
    parser.add_argument('-e', '--epsilon', default=0.01,
                        help="The precision, used for SMC only. Defaults to 0.01.")
    parser.add_argument('-d', '--delta', default=0.05,
                        help="The error rate, used for SMC only. Defaults to 0.05.")
    parser.add_argument('-v', '--verbose', type=int, default=1, choices=[0,1,2],
                        help="0: no prints; 1: main prints; 2: all prints")
    args = parser.parse_args()
    # lexer and grammar
    lexer = gramLexer(FileStream(args.filename))
    stream = CommonTokenStream(lexer)
    parser = gramParser(stream)
    tree = parser.program()
    MDP_parser = gramMDPListener()
    walker = ParseTreeWalker()
    # parse file
    walker.walk(MDP_parser, tree)
    mdp = MDP_parser.mdp
    mdp.build(args.initial_state)
    if args.method == 'draw':
        print(mdp)
    elif args.method == 'simulate':
        mdp.simulate(args.n_steps, args.strategy, args.verbose)
    elif args.method == 'smc':
        mdp.smc(args.terminal_state, args.n_steps, args.epsilon, args.delta, args.verbose)


if __name__ == '__main__':
    main()
