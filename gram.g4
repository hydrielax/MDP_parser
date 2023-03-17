grammar gram;

program
    : defstates (defactions | c_defactions)* transitions EOF
    ;

defstates : STATES defstate (',' defstate)* ';';

defactions : ACTIONS ID (',' ID)* ';';
c_defactions : COMMENT ACTIONS ID (',' ID)* ';';

defstate : ID | (ID ':' INT);

transitions : trans (trans)* ;


trans : transact | transnoact | c_transact | c_transnoact;

transact : ID '[' ID ']' FLECHE INT ':' ID 
    ('+' INT ':' ID)* ';';
transnoact : ID FLECHE INT ':' ID 
    ('+' INT ':' ID)* ';';

c_transact : COMMENT ID '[' ID ']' FLECHE INT ':' ID 
    ('+' INT ':' ID)* ';';
c_transnoact : COMMENT ID FLECHE INT ':' ID 
    ('+' INT ':' ID)* ';';

STATES : 'States';
ACTIONS : 'Actions' ;
TRANSITION : 'transition' ;
DPOINT : ':' ;
FLECHE : '->';
SEMI : ';' ;
VIRG : ',';
PLUS : '+';
LCROCH : '[' ;
RCROCH : ']' ;
COMMENT : '#' ;

INT : [0-9]+ ;
ID: [a-zA-Z_][a-zA-Z_0-9]* ;
WS: [ \t\n\r\f]+ -> skip ;





