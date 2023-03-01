import argparse

from antlr4 import FileStream, CommonTokenStream, ParseTreeWalker

from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser
from mdp import MDP
import strategies


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


def main(filename, initial = None, n_steps = 0, strategy = 'ask_user', verbose=False):
    # lexer and grammar
    lexer = gramLexer(FileStream(filename))
    stream = CommonTokenStream(lexer)
    parser = gramParser(stream)
    tree = parser.program()
    MDP_parser = gramMDPListener()
    walker = ParseTreeWalker()
    # parse file
    walker.walk(MDP_parser, tree)
    mdp = MDP_parser.mdp
    mdp.build()
    print(mdp)
    mdp.simulate(initial, n_steps, strategy, verbose)


if __name__ == '__main__':
    # arg parse
    parser = argparse.ArgumentParser(
        prog = 'python main.py',
        description = 'Markov Decision Process Chains Analyser')
    parser.add_argument('filename')
    parser.add_argument('-i', '--initial', help="Initial state label for the simulation.")
    parser.add_argument('-n', '--number-of-steps', type=int, dest='n_steps', help="Number of steps to apply for the simulation.", default=5)
    parser.add_argument('-s', '--strategy', help="Strategy to use for the simulation.", default='ask_user')
    parser.add_argument('-v', '--verbose', help="Show all prints or not.", action='store_true', default=True)
    kwargs = vars(parser.parse_args())
    print(kwargs)
    main(**kwargs)
