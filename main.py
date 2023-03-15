import argparse

from antlr4 import FileStream, CommonTokenStream, ParseTreeWalker

from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser
from mdp import MDP


class gramMDPListener(gramListener):

    def __init__(self):
        self.mdp = MDP()
        
    def enterDefstate(self, ctx):
        if ctx.INT():
            self.mdp.add_state(str(ctx.ID()), int(str(ctx.INT())))
        else:
            self.mdp.add_state(str(ctx.ID()))

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
        prog='python main.py',
        description='Markov Decision Process Chains Analyser')
    parser.add_argument('method', choices=['draw', 'simulate', 'smc', 'sprt', 'val_iter'],
                        help="The method to use: draw, simulate, SMC, SPRT, Values Iteration")
    parser.add_argument('filename',
                        help="The name of the file.")
    parser.add_argument('-a', '--alpha', type=float, default=0.01,
                        help="The born on type 1 error, defaults to 0.01.")
    parser.add_argument('-b', '--beta', type=float, default=0.01,
                        help="The born on type 2 error, defaults to 0.01.")
    parser.add_argument('-d', '--delta', type=float, default=0.05,
                        help="The error rate, used for SMC only. Defaults to 0.05.")
    parser.add_argument('-e', '--epsilon', type=float, default=0.01,
                        help="The precision, used for SMC only. Defaults to 0.01.")
    parser.add_argument('-g', '--gamma', type=float, default=0.9,
                        help="The gamma factor, used for strategy optimization. Defaults to 0.9.")
    parser.add_argument('-t', '--theta', type=float, default=None,
                        help="The test value compaired, used for SPRT. Required.")
    parser.add_argument('-i', '--iter-max', type=float, default=10_000,
                        help="Max number of iterations for SPRT. Defaults to 10.000")
    parser.add_argument('-n', '--number-of-steps', type=int, dest='n_steps', default=5,
                        help="Number of steps to apply for the simulation, defaults to 5.")
    parser.add_argument('-s', '--strategy',  default='ask_user',
                        help="Strategy to use for the simulation, defaults to 'ask_user'.")
    parser.add_argument('-v', '--verbose', type=int, default=1, choices=[0,1,2],
                        help="0: no prints; 1: main prints; 2: all prints")
    parser.add_argument('-I', '--initial-state', default=None,
                        help="Initial state label for the simulation, defaults to the first one.")
    parser.add_argument('-T', '--terminal-state', default=None,
                        help="The terminal state, used for SMC and SPRT. Required.")
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
    elif args.method == 'sprt':
        mdp.sprt(args.terminal_state, args.n_steps, args.alpha, args.beta,
                 args.epsilon, args.theta, args.iter_max, args.verbose)
    elif args.method == 'val_iter':
        mdp.value_iteration(args.gamma, args.epsilon, args.iter_max)

# to compile grammar: $ antlr4 -Dlanguage=Python3 gram.g4

if __name__ == '__main__':
    main()
