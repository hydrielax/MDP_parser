# Generated from gram.g4 by ANTLR 4.12.0
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,13,101,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,1,0,1,0,5,0,19,8,0,10,0,12,0,22,9,0,1,0,1,0,1,0,1,1,1,
        1,1,1,1,1,5,1,31,8,1,10,1,12,1,34,9,1,1,1,1,1,1,2,1,2,1,2,1,2,5,
        2,42,8,2,10,2,12,2,45,9,2,1,2,1,2,1,3,1,3,1,3,1,3,3,3,53,8,3,1,4,
        1,4,5,4,57,8,4,10,4,12,4,60,9,4,1,5,1,5,3,5,64,8,5,1,6,1,6,1,6,1,
        6,1,6,1,6,1,6,1,6,1,6,1,6,1,6,1,6,5,6,78,8,6,10,6,12,6,81,9,6,1,
        6,1,6,1,7,1,7,1,7,1,7,1,7,1,7,1,7,1,7,1,7,5,7,94,8,7,10,7,12,7,97,
        9,7,1,7,1,7,1,7,0,0,8,0,2,4,6,8,10,12,14,0,0,100,0,16,1,0,0,0,2,
        26,1,0,0,0,4,37,1,0,0,0,6,52,1,0,0,0,8,54,1,0,0,0,10,63,1,0,0,0,
        12,65,1,0,0,0,14,84,1,0,0,0,16,20,3,2,1,0,17,19,3,4,2,0,18,17,1,
        0,0,0,19,22,1,0,0,0,20,18,1,0,0,0,20,21,1,0,0,0,21,23,1,0,0,0,22,
        20,1,0,0,0,23,24,3,8,4,0,24,25,5,0,0,1,25,1,1,0,0,0,26,27,5,1,0,
        0,27,32,3,6,3,0,28,29,5,7,0,0,29,31,3,6,3,0,30,28,1,0,0,0,31,34,
        1,0,0,0,32,30,1,0,0,0,32,33,1,0,0,0,33,35,1,0,0,0,34,32,1,0,0,0,
        35,36,5,6,0,0,36,3,1,0,0,0,37,38,5,2,0,0,38,43,5,12,0,0,39,40,5,
        7,0,0,40,42,5,12,0,0,41,39,1,0,0,0,42,45,1,0,0,0,43,41,1,0,0,0,43,
        44,1,0,0,0,44,46,1,0,0,0,45,43,1,0,0,0,46,47,5,6,0,0,47,5,1,0,0,
        0,48,53,5,12,0,0,49,50,5,12,0,0,50,51,5,4,0,0,51,53,5,11,0,0,52,
        48,1,0,0,0,52,49,1,0,0,0,53,7,1,0,0,0,54,58,3,10,5,0,55,57,3,10,
        5,0,56,55,1,0,0,0,57,60,1,0,0,0,58,56,1,0,0,0,58,59,1,0,0,0,59,9,
        1,0,0,0,60,58,1,0,0,0,61,64,3,12,6,0,62,64,3,14,7,0,63,61,1,0,0,
        0,63,62,1,0,0,0,64,11,1,0,0,0,65,66,5,12,0,0,66,67,5,9,0,0,67,68,
        5,12,0,0,68,69,5,10,0,0,69,70,5,5,0,0,70,71,5,11,0,0,71,72,5,4,0,
        0,72,79,5,12,0,0,73,74,5,8,0,0,74,75,5,11,0,0,75,76,5,4,0,0,76,78,
        5,12,0,0,77,73,1,0,0,0,78,81,1,0,0,0,79,77,1,0,0,0,79,80,1,0,0,0,
        80,82,1,0,0,0,81,79,1,0,0,0,82,83,5,6,0,0,83,13,1,0,0,0,84,85,5,
        12,0,0,85,86,5,5,0,0,86,87,5,11,0,0,87,88,5,4,0,0,88,95,5,12,0,0,
        89,90,5,8,0,0,90,91,5,11,0,0,91,92,5,4,0,0,92,94,5,12,0,0,93,89,
        1,0,0,0,94,97,1,0,0,0,95,93,1,0,0,0,95,96,1,0,0,0,96,98,1,0,0,0,
        97,95,1,0,0,0,98,99,5,6,0,0,99,15,1,0,0,0,8,20,32,43,52,58,63,79,
        95
    ]

class gramParser ( Parser ):

    grammarFileName = "gram.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'States'", "'Actions'", "'transition'", 
                     "':'", "'->'", "';'", "','", "'+'", "'['", "']'" ]

    symbolicNames = [ "<INVALID>", "STATES", "ACTIONS", "TRANSITION", "DPOINT", 
                      "FLECHE", "SEMI", "VIRG", "PLUS", "LCROCH", "RCROCH", 
                      "INT", "ID", "WS" ]

    RULE_program = 0
    RULE_defstates = 1
    RULE_defactions = 2
    RULE_defstate = 3
    RULE_transitions = 4
    RULE_trans = 5
    RULE_transact = 6
    RULE_transnoact = 7

    ruleNames =  [ "program", "defstates", "defactions", "defstate", "transitions", 
                   "trans", "transact", "transnoact" ]

    EOF = Token.EOF
    STATES=1
    ACTIONS=2
    TRANSITION=3
    DPOINT=4
    FLECHE=5
    SEMI=6
    VIRG=7
    PLUS=8
    LCROCH=9
    RCROCH=10
    INT=11
    ID=12
    WS=13

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.12.0")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class ProgramContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def defstates(self):
            return self.getTypedRuleContext(gramParser.DefstatesContext,0)


        def transitions(self):
            return self.getTypedRuleContext(gramParser.TransitionsContext,0)


        def EOF(self):
            return self.getToken(gramParser.EOF, 0)

        def defactions(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(gramParser.DefactionsContext)
            else:
                return self.getTypedRuleContext(gramParser.DefactionsContext,i)


        def getRuleIndex(self):
            return gramParser.RULE_program

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterProgram" ):
                listener.enterProgram(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitProgram" ):
                listener.exitProgram(self)




    def program(self):

        localctx = gramParser.ProgramContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_program)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 16
            self.defstates()
            self.state = 20
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==2:
                self.state = 17
                self.defactions()
                self.state = 22
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 23
            self.transitions()
            self.state = 24
            self.match(gramParser.EOF)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class DefstatesContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def STATES(self):
            return self.getToken(gramParser.STATES, 0)

        def defstate(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(gramParser.DefstateContext)
            else:
                return self.getTypedRuleContext(gramParser.DefstateContext,i)


        def SEMI(self):
            return self.getToken(gramParser.SEMI, 0)

        def VIRG(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.VIRG)
            else:
                return self.getToken(gramParser.VIRG, i)

        def getRuleIndex(self):
            return gramParser.RULE_defstates

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterDefstates" ):
                listener.enterDefstates(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitDefstates" ):
                listener.exitDefstates(self)




    def defstates(self):

        localctx = gramParser.DefstatesContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_defstates)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 26
            self.match(gramParser.STATES)
            self.state = 27
            self.defstate()
            self.state = 32
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==7:
                self.state = 28
                self.match(gramParser.VIRG)
                self.state = 29
                self.defstate()
                self.state = 34
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 35
            self.match(gramParser.SEMI)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class DefactionsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ACTIONS(self):
            return self.getToken(gramParser.ACTIONS, 0)

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.ID)
            else:
                return self.getToken(gramParser.ID, i)

        def SEMI(self):
            return self.getToken(gramParser.SEMI, 0)

        def VIRG(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.VIRG)
            else:
                return self.getToken(gramParser.VIRG, i)

        def getRuleIndex(self):
            return gramParser.RULE_defactions

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterDefactions" ):
                listener.enterDefactions(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitDefactions" ):
                listener.exitDefactions(self)




    def defactions(self):

        localctx = gramParser.DefactionsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_defactions)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 37
            self.match(gramParser.ACTIONS)
            self.state = 38
            self.match(gramParser.ID)
            self.state = 43
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==7:
                self.state = 39
                self.match(gramParser.VIRG)
                self.state = 40
                self.match(gramParser.ID)
                self.state = 45
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 46
            self.match(gramParser.SEMI)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class DefstateContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(gramParser.ID, 0)

        def DPOINT(self):
            return self.getToken(gramParser.DPOINT, 0)

        def INT(self):
            return self.getToken(gramParser.INT, 0)

        def getRuleIndex(self):
            return gramParser.RULE_defstate

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterDefstate" ):
                listener.enterDefstate(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitDefstate" ):
                listener.exitDefstate(self)




    def defstate(self):

        localctx = gramParser.DefstateContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_defstate)
        try:
            self.state = 52
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,3,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 48
                self.match(gramParser.ID)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 49
                self.match(gramParser.ID)
                self.state = 50
                self.match(gramParser.DPOINT)
                self.state = 51
                self.match(gramParser.INT)
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TransitionsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def trans(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(gramParser.TransContext)
            else:
                return self.getTypedRuleContext(gramParser.TransContext,i)


        def getRuleIndex(self):
            return gramParser.RULE_transitions

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTransitions" ):
                listener.enterTransitions(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTransitions" ):
                listener.exitTransitions(self)




    def transitions(self):

        localctx = gramParser.TransitionsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_transitions)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 54
            self.trans()
            self.state = 58
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==12:
                self.state = 55
                self.trans()
                self.state = 60
                self._errHandler.sync(self)
                _la = self._input.LA(1)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TransContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def transact(self):
            return self.getTypedRuleContext(gramParser.TransactContext,0)


        def transnoact(self):
            return self.getTypedRuleContext(gramParser.TransnoactContext,0)


        def getRuleIndex(self):
            return gramParser.RULE_trans

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTrans" ):
                listener.enterTrans(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTrans" ):
                listener.exitTrans(self)




    def trans(self):

        localctx = gramParser.TransContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_trans)
        try:
            self.state = 63
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,5,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 61
                self.transact()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 62
                self.transnoact()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TransactContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.ID)
            else:
                return self.getToken(gramParser.ID, i)

        def LCROCH(self):
            return self.getToken(gramParser.LCROCH, 0)

        def RCROCH(self):
            return self.getToken(gramParser.RCROCH, 0)

        def FLECHE(self):
            return self.getToken(gramParser.FLECHE, 0)

        def INT(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.INT)
            else:
                return self.getToken(gramParser.INT, i)

        def DPOINT(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.DPOINT)
            else:
                return self.getToken(gramParser.DPOINT, i)

        def SEMI(self):
            return self.getToken(gramParser.SEMI, 0)

        def PLUS(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.PLUS)
            else:
                return self.getToken(gramParser.PLUS, i)

        def getRuleIndex(self):
            return gramParser.RULE_transact

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTransact" ):
                listener.enterTransact(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTransact" ):
                listener.exitTransact(self)




    def transact(self):

        localctx = gramParser.TransactContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_transact)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 65
            self.match(gramParser.ID)
            self.state = 66
            self.match(gramParser.LCROCH)
            self.state = 67
            self.match(gramParser.ID)
            self.state = 68
            self.match(gramParser.RCROCH)
            self.state = 69
            self.match(gramParser.FLECHE)
            self.state = 70
            self.match(gramParser.INT)
            self.state = 71
            self.match(gramParser.DPOINT)
            self.state = 72
            self.match(gramParser.ID)
            self.state = 79
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==8:
                self.state = 73
                self.match(gramParser.PLUS)
                self.state = 74
                self.match(gramParser.INT)
                self.state = 75
                self.match(gramParser.DPOINT)
                self.state = 76
                self.match(gramParser.ID)
                self.state = 81
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 82
            self.match(gramParser.SEMI)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TransnoactContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.ID)
            else:
                return self.getToken(gramParser.ID, i)

        def FLECHE(self):
            return self.getToken(gramParser.FLECHE, 0)

        def INT(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.INT)
            else:
                return self.getToken(gramParser.INT, i)

        def DPOINT(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.DPOINT)
            else:
                return self.getToken(gramParser.DPOINT, i)

        def SEMI(self):
            return self.getToken(gramParser.SEMI, 0)

        def PLUS(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.PLUS)
            else:
                return self.getToken(gramParser.PLUS, i)

        def getRuleIndex(self):
            return gramParser.RULE_transnoact

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTransnoact" ):
                listener.enterTransnoact(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTransnoact" ):
                listener.exitTransnoact(self)




    def transnoact(self):

        localctx = gramParser.TransnoactContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_transnoact)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 84
            self.match(gramParser.ID)
            self.state = 85
            self.match(gramParser.FLECHE)
            self.state = 86
            self.match(gramParser.INT)
            self.state = 87
            self.match(gramParser.DPOINT)
            self.state = 88
            self.match(gramParser.ID)
            self.state = 95
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==8:
                self.state = 89
                self.match(gramParser.PLUS)
                self.state = 90
                self.match(gramParser.INT)
                self.state = 91
                self.match(gramParser.DPOINT)
                self.state = 92
                self.match(gramParser.ID)
                self.state = 97
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 98
            self.match(gramParser.SEMI)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx





