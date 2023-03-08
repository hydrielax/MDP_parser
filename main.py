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
    parser.add_argument('method', choices=['draw', 'simulate', 'smc'])
    parser.add_argument('filename')
    parser.add_argument('-i', '--initial-state', help="Initial state label for the simulation.", default=None)
    parser.add_argument('-n', '--number-of-steps', type=int, dest='n_steps', help="Number of steps to apply for the simulation.", default=5)
    parser.add_argument('-s', '--strategy', help="Strategy to use for the simulation.", default='ask_user')
    parser.add_argument('-t', '--terminal-state', default=None)
    parser.add_argument('-e', '--epsilon', default=0.01)
    parser.add_argument('-d', '--delta', default=0.05)
    parser.add_argument('-v', '--verbose', help="0: no prints; 1: main prints; 2: all prints", default=2)
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
