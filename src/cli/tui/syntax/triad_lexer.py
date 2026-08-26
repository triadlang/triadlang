from pygments.lexer import RegexLexer
from pygments.token import (
    Comment,
    Error,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    Text,
)


class TriadLexer(RegexLexer):
    name = 'TriadLang'
    aliases = ['tri', 'triad']
    filenames = ['*.tri']
    mimetypes = ['text/x-triad']

    tokens = {
        'root': [
            (r'//[^\n]*', Comment.Single),
            (r'#[^\n]*', Comment.Single),

            (r'(r|rf|rb|R|RF|RB)(\"\"\"|\'\'\')', String.Doc, 'tdq_raw'),
            (r'(r|rf|rb|R|RF|RB)(\"|\')', String.Regex, 'sq_raw'),
            (r'\"\"\"', String.Doc, 'tdq'),
            (r'\'\'\'', String.Doc, 'sdq'),
            (r'f\"', String.Affix, 'fstring_dq'),
            (r"f\'", String.Affix, 'fstring_sq'),
            (r'\"', String.Double, 'dq'),
            (r'\'', String.Single, 'sq'),

            (r'0[xX][0-9a-fA-F_]+', Number.Hex),
            (r'0[bB][0-1_]+', Number.Bin),
            (r'[0-9][0-9_]*\.[0-9][0-9_]*[jJ]', Number.Float),
            (r'[0-9][0-9_]*[jJ]', Number.Float),
            (r'[0-9][0-9_]*\.[0-9][0-9_]*([eE][+-]?[0-9_]+)?', Number.Float),
            (r'[0-9][0-9_]*[eE][+-]?[0-9_]+', Number.Float),
            (r'[0-9][0-9_]+', Number.Integer),

            (r'(assert|pass|del|is|in|and|or|not|true|false|none)\b', Keyword.Constant),
            (r'(let|const|fn|return|if|else|elif|for|while|break|continue|match|case|yield|async|await|try|catch|finally|throw|with|class|type|import|from|as|self|super|inherits)\b', Keyword.Reserved),
            (r'(reg|pair|ring|evolve|OBSERVE|observe|run|couple|entity|world|sequence|via|each_for|substrate|composed_of)\b', Keyword.Namespace),
            (r'(assert|persistent|extended|structurally_open|mem_memory|atomic|anti_collapsed|over_seeds)\b', Keyword.Pseudo),

            (r'=>|->|<<=|>>=|//=|\*\*=|@=|&=|\|=|\^=|%=|\+=|-=|\*=|/=|<<|>>|==|!=|<=|>=|\*\*|//', Operator),
            (r'[+\-*/%@&|^~<>=!]', Operator),

            (r'[(){}\[\];:,.?]', Punctuation),

            (r'@\w+', Name.Decorator),

            (r'[A-Z_][A-Za-z0-9_]*', Name.Class),
            (r'_[A-Za-z0-9_]*', Name.Private),
            (r'[a-z_][A-Za-z0-9_]*', Name),

            (r'\s+', Text),
            (r'.', Error),
        ],

        'dq': [(r'[^\"\\]+', String.Double), (r'\\[ntr\\\"\'0]', String.Escape), (r'\"', String.Double, '#pop'), (r'.', String.Double)],
        'sq': [(r"[^\'\\]+", String.Single), (r"\\[ntr\\\"\'0]", String.Escape), (r"\'", String.Single, '#pop'), (r'.', String.Single)],
        'tdq': [(r'[^\"\\]+', String.Doc), (r'\\[ntr\\\"\'0]', String.Escape), (r'\"\"\"', String.Doc, '#pop'), (r'.', String.Doc)],
        'sdq': [(r"[^\'\\]+", String.Doc), (r"\\[ntr\\\"\'0]", String.Escape), (r"\'\'\'", String.Doc, '#pop'), (r'.', String.Doc)],
        'sq_raw': [(r'[^\'\\]+', String.Regex), (r'\\.', String.Regex), (r"\'", String.Regex, '#pop')],
        'tdq_raw': [(r'[^\"\\]+', String.Doc), (r'\"\"\"', String.Doc, '#pop'), (r'.', String.Doc)],
        'fstring_dq': [(r'[^\"\\{]+', String.Affix), (r'\\[ntr\\\"\'0]', String.Escape), (r'\{', String.Affix, 'fstring_expr_dq'), (r'\"', String.Affix, '#pop'), (r'.', String.Affix)],
        'fstring_sq': [(r"[^\'\\{]+", String.Affix), (r"\\[ntr\\\"\'0]", String.Escape), (r'\{', String.Affix, 'fstring_expr_sq'), (r"\'", String.Affix, '#pop'), (r'.', String.Affix)],
        'fstring_expr_dq': [(r'[^}]+', String.Interpol), (r'\}', String.Affix, '#pop')],
        'fstring_expr_sq': [(r'[^}]+', String.Interpol), (r'\}', String.Affix, '#pop')],
    }

