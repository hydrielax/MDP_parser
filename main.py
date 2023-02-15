from antlr4 import StdinStream, CommonTokenStream, ParseTreeWalker
from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser
from mdp import MDP
import sys


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
    lexer = gramLexer(StdinStream())
    stream = CommonTokenStream(lexer)
    parser = gramParser(stream)
    tree = parser.program()
    MDP_parser = gramMDPListener()
    walker = ParseTreeWalker()
    walker.walk(MDP_parser, tree)
    mdp = MDP_parser.mdp
    print(mdp)

if __name__ == '__main__':
    main()
