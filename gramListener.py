# Generated from gram.g4 by ANTLR 4.12.0
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .gramParser import gramParser
else:
    from gramParser import gramParser

# This class defines a complete listener for a parse tree produced by gramParser.
class gramListener(ParseTreeListener):

    # Enter a parse tree produced by gramParser#program.
    def enterProgram(self, ctx:gramParser.ProgramContext):
        pass

    # Exit a parse tree produced by gramParser#program.
    def exitProgram(self, ctx:gramParser.ProgramContext):
        pass


    # Enter a parse tree produced by gramParser#defstates.
    def enterDefstates(self, ctx:gramParser.DefstatesContext):
        pass

    # Exit a parse tree produced by gramParser#defstates.
    def exitDefstates(self, ctx:gramParser.DefstatesContext):
        pass


    # Enter a parse tree produced by gramParser#defactions.
    def enterDefactions(self, ctx:gramParser.DefactionsContext):
        pass

    # Exit a parse tree produced by gramParser#defactions.
    def exitDefactions(self, ctx:gramParser.DefactionsContext):
        pass


    # Enter a parse tree produced by gramParser#c_defactions.
    def enterC_defactions(self, ctx:gramParser.C_defactionsContext):
        pass

    # Exit a parse tree produced by gramParser#c_defactions.
    def exitC_defactions(self, ctx:gramParser.C_defactionsContext):
        pass


    # Enter a parse tree produced by gramParser#defstate.
    def enterDefstate(self, ctx:gramParser.DefstateContext):
        pass

    # Exit a parse tree produced by gramParser#defstate.
    def exitDefstate(self, ctx:gramParser.DefstateContext):
        pass


    # Enter a parse tree produced by gramParser#transitions.
    def enterTransitions(self, ctx:gramParser.TransitionsContext):
        pass

    # Exit a parse tree produced by gramParser#transitions.
    def exitTransitions(self, ctx:gramParser.TransitionsContext):
        pass


    # Enter a parse tree produced by gramParser#trans.
    def enterTrans(self, ctx:gramParser.TransContext):
        pass

    # Exit a parse tree produced by gramParser#trans.
    def exitTrans(self, ctx:gramParser.TransContext):
        pass


    # Enter a parse tree produced by gramParser#transact.
    def enterTransact(self, ctx:gramParser.TransactContext):
        pass

    # Exit a parse tree produced by gramParser#transact.
    def exitTransact(self, ctx:gramParser.TransactContext):
        pass


    # Enter a parse tree produced by gramParser#transnoact.
    def enterTransnoact(self, ctx:gramParser.TransnoactContext):
        pass

    # Exit a parse tree produced by gramParser#transnoact.
    def exitTransnoact(self, ctx:gramParser.TransnoactContext):
        pass


    # Enter a parse tree produced by gramParser#c_transact.
    def enterC_transact(self, ctx:gramParser.C_transactContext):
        pass

    # Exit a parse tree produced by gramParser#c_transact.
    def exitC_transact(self, ctx:gramParser.C_transactContext):
        pass


    # Enter a parse tree produced by gramParser#c_transnoact.
    def enterC_transnoact(self, ctx:gramParser.C_transnoactContext):
        pass

    # Exit a parse tree produced by gramParser#c_transnoact.
    def exitC_transnoact(self, ctx:gramParser.C_transnoactContext):
        pass



del gramParser