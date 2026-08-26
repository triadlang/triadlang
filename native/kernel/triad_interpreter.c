#include "triad_interpreter.h"
#include "triad_serial.h"
#include "triad_mm.h"
#include "triad_syscall.h"
#include "triad_ai.h"
#include "triad_equilibrium.h"
#include <string.h>

static TriadVal *val_new(ValType type) {
    TriadVal *v = (TriadVal *)triad_mm_alloc(sizeof(TriadVal));
    if (!v) return NULL;
    memset(v, 0, sizeof(TriadVal));
    v->type = type;
    return v;
}

static TriadVal *val_int(int64_t i) {
    TriadVal *v = val_new(VAL_INT);
    if (v) v->ival = i;
    return v;
}

static TriadVal *val_float(double f) {
    TriadVal *v = val_new(VAL_FLOAT);
    if (v) v->fval = f;
    return v;
}

static TriadVal *val_string(const char *s) {
    TriadVal *v = val_new(VAL_STRING);
    if (v) {
        int k = 0;
        while (s[k] && k < 255) { v->sval[k] = s[k]; k++; }
        v->sval[k] = 0;
    }
    return v;
}

static TriadVal *val_bool(bool b) {
    TriadVal *v = val_new(VAL_BOOL);
    if (v) v->bval = b;
    return v;
}

static TriadVal *val_none(void) {
    return val_new(VAL_NONE);
}

static bool val_truthy(TriadVal *v) {
    if (!v) return false;
    switch (v->type) {
    case VAL_NONE: return false;
    case VAL_BOOL: return v->bval;
    case VAL_INT: return v->ival != 0;
    case VAL_FLOAT: return v->fval != 0.0;
    case VAL_STRING: return v->sval[0] != 0;
    default: return true;
    }
}

static int str_eq(const char *a, const char *b) {
    int k = 0;
    while (a[k] && b[k]) {
        if (a[k] != b[k]) return 0;
        k++;
    }
    return a[k] == 0 && b[k] == 0;
}

void triad_interp_print(TriadVal *val) {
    if (!val) return;
    switch (val->type) {
    case VAL_NONE:
        triad_serial_puts("none");
        break;
    case VAL_BOOL:
        triad_serial_puts(val->bval ? "true" : "false");
        break;
    case VAL_INT:
        if (val->ival < 0) {
            triad_serial_putc('-');
            triad_serial_dec((uint64_t)(-val->ival));
        } else {
            triad_serial_dec((uint64_t)val->ival);
        }
        break;
    case VAL_FLOAT:
        triad_serial_float(val->fval);
        break;
    case VAL_STRING:
        triad_serial_puts(val->sval);
        break;
    default:
        triad_serial_puts("?");
    }
}

void triad_interp_init(TriadInterp *interp, int proc_id) {
    memset(interp, 0, sizeof(TriadInterp));
    interp->proc_id = proc_id;
    interp->scope_top = 0;
    interp->scopes[0].n_vars = 0;
    interp->call_top = 0;
    interp->stack_top = 0;
    interp->running = true;
    interp->error = false;
}

void triad_interp_push_scope(TriadInterp *interp) {
    if (interp->scope_top < INTERP_MAX_SCOPE - 1) {
        interp->scope_top++;
        interp->scopes[interp->scope_top].n_vars = 0;
        interp->scopes[interp->scope_top].parent = interp->scope_top - 1;
    }
}

void triad_interp_pop_scope(TriadInterp *interp) {
    if (interp->scope_top > 0) interp->scope_top--;
}

void triad_interp_set_var(TriadInterp *interp, const char *name, TriadVal *val) {
    TriadScope *s = &interp->scopes[interp->scope_top];
    for (int i = 0; i < s->n_vars; i++) {
        if (str_eq(s->vars[i].name, name)) {
            s->vars[i].value = val;
            return;
        }
    }
    if (s->n_vars < INTERP_MAX_VARS) {
        int k = 0;
        while (name[k] && k < 63) { s->vars[s->n_vars].name[k] = name[k]; k++; }
        s->vars[s->n_vars].name[k] = 0;
        s->vars[s->n_vars].value = val;
        s->n_vars++;
    }
}

TriadVal *triad_interp_get_var(TriadInterp *interp, const char *name) {
    int scope = interp->scope_top;
    while (scope >= 0) {
        TriadScope *s = &interp->scopes[scope];
        for (int i = 0; i < s->n_vars; i++) {
            if (str_eq(s->vars[i].name, name)) return s->vars[i].value;
        }
        scope = s->parent;
    }
    return NULL;
}

static void interp_error(TriadInterp *interp, const char *msg) {
    interp->error = true;
    interp->running = false;
    int k = 0;
    while (msg[k] && k < 255) { interp->error_msg[k] = msg[k]; k++; }
    interp->error_msg[k] = 0;
    triad_serial_puts("[interp] error: ");
    triad_serial_puts(interp->error_msg);
    triad_serial_puts("\n");
}

static const char *INTERP_KEYWORDS[] = {
    "let", "const", "fn", "return", "if", "else", "elif", "for", "in",
    "while", "break", "continue", "true", "false", "none",
    "and", "or", "not",
    NULL
};

static int is_interp_keyword(const char *s) {
    for (int i = 0; INTERP_KEYWORDS[i]; i++) {
        if (str_eq(s, INTERP_KEYWORDS[i])) return 1;
    }
    return 0;
}

static void lex_init(InterpLexer *lex, const char *src) {
    lex->src = src;
    lex->pos = 0;
    lex->line = 1;
    lex->col = 1;
    lex->len = 0;
    while (src[lex->len]) lex->len++;
    lex->current.type = TRI_TOKEN_EOF;
    lex->current.value[0] = 0;
    lex->peek.type = TRI_TOKEN_EOF;
    lex->peek.value[0] = 0;
}

static int lex_skip_ws(InterpLexer *lex) {
    while (lex->pos < lex->len) {
        char c = lex->src[lex->pos];
        if (c == '\n') { lex->line++; lex->col = 1; lex->pos++; }
        else if (c == ' ' || c == '\t' || c == '\r') { lex->col++; lex->pos++; }
        else if (c == '/' && lex->pos + 1 < lex->len && lex->src[lex->pos + 1] == '/') {
            while (lex->pos < lex->len && lex->src[lex->pos] != '\n') lex->pos++;
        }
        else if (c == '#') {
            while (lex->pos < lex->len && lex->src[lex->pos] != '\n') lex->pos++;
        }
        else break;
    }
    return lex->pos >= lex->len;
}

static int lex_next(InterpLexer *lex, TriToken *out) {
    if (lex_skip_ws(lex)) {
        out->type = TRI_TOKEN_EOF;
        out->value[0] = 0;
        return 0;
    }
    char c = lex->src[lex->pos];
    out->line = lex->line;
    out->col = lex->col;

    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_') {
        int n = 0;
        while (lex->pos < lex->len) {
            c = lex->src[lex->pos];
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') || c == '_') {
                if (n < 127) out->value[n] = c;
                n++; lex->pos++; lex->col++;
            } else break;
        }
        out->value[n] = 0;
        if (is_interp_keyword(out->value)) {
            out->type = TRI_TOKEN_KEYWORD;
        } else {
            out->type = TRI_TOKEN_IDENT;
        }
        return 1;
    }
    if (c >= '0' && c <= '9') {
        int n = 0;
        while (lex->pos < lex->len) {
            c = lex->src[lex->pos];
            if (c == '.') { if (n < 127) out->value[n] = c; n++; lex->pos++; lex->col++; }
            else if (c >= '0' && c <= '9') { if (n < 127) out->value[n] = c; n++; lex->pos++; lex->col++; }
            else break;
        }
        out->value[n] = 0;
        out->type = TRI_TOKEN_NUMBER;
        return 1;
    }
    if (c == '"') {
        lex->pos++; lex->col++;
        int n = 0;
        while (lex->pos < lex->len && lex->src[lex->pos] != '"') {
            if (lex->src[lex->pos] == '\\' && lex->pos + 1 < lex->len) {
                lex->pos++; lex->col++;
                char esc = lex->src[lex->pos];
                char actual = esc;
                if (esc == 'n') actual = '\n';
                else if (esc == 't') actual = '\t';
                else if (esc == '\\') actual = '\\';
                else if (esc == '"') actual = '"';
                if (n < 127) out->value[n] = actual;
                n++; lex->pos++; lex->col++;
            } else {
                if (n < 127) out->value[n] = lex->src[lex->pos];
                n++; lex->pos++; lex->col++;
            }
        }
        if (lex->pos < lex->len) { lex->pos++; lex->col++; }
        out->value[n] = 0;
        out->type = TRI_TOKEN_STRING;
        return 1;
    }

    if (c == '=' && lex->pos + 1 < lex->len && lex->src[lex->pos + 1] == '=') {
        out->value[0] = '='; out->value[1] = '='; out->value[2] = 0;
        out->type = TRI_TOKEN_SYMBOL;
        lex->pos += 2; lex->col += 2;
        return 1;
    }
    if (c == '!' && lex->pos + 1 < lex->len && lex->src[lex->pos + 1] == '=') {
        out->value[0] = '!'; out->value[1] = '='; out->value[2] = 0;
        out->type = TRI_TOKEN_SYMBOL;
        lex->pos += 2; lex->col += 2;
        return 1;
    }
    if (c == '<' && lex->pos + 1 < lex->len && lex->src[lex->pos + 1] == '=') {
        out->value[0] = '<'; out->value[1] = '='; out->value[2] = 0;
        out->type = TRI_TOKEN_SYMBOL;
        lex->pos += 2; lex->col += 2;
        return 1;
    }
    if (c == '>' && lex->pos + 1 < lex->len && lex->src[lex->pos + 1] == '=') {
        out->value[0] = '>'; out->value[1] = '='; out->value[2] = 0;
        out->type = TRI_TOKEN_SYMBOL;
        lex->pos += 2; lex->col += 2;
        return 1;
    }
    if (c == '&' && lex->pos + 1 < lex->len && lex->src[lex->pos + 1] == '&') {
        out->value[0] = '&'; out->value[1] = '&'; out->value[2] = 0;
        out->type = TRI_TOKEN_SYMBOL;
        lex->pos += 2; lex->col += 2;
        return 1;
    }
    if (c == '|' && lex->pos + 1 < lex->len && lex->src[lex->pos + 1] == '|') {
        out->value[0] = '|'; out->value[1] = '|'; out->value[2] = 0;
        out->type = TRI_TOKEN_SYMBOL;
        lex->pos += 2; lex->col += 2;
        return 1;
    }

    out->value[0] = c;
    out->value[1] = 0;
    out->type = TRI_TOKEN_SYMBOL;
    lex->pos++; lex->col++;
    return 1;
}

static TriadVal *parse_expr(InterpLexer *lex, TriadInterp *interp);
static int exec_stmt(InterpLexer *lex, TriadInterp *interp);

static TriadVal *parse_primary(InterpLexer *lex, TriadInterp *interp) {
    TriToken tok;
    if (!lex_next(lex, &tok)) return val_none();

    if (tok.type == TRI_TOKEN_NUMBER) {
        bool is_float = false;
        int k = 0;
        while (tok.value[k]) {
            if (tok.value[k] == '.') { is_float = true; break; }
            k++;
        }
        if (is_float) {
            double val = 0;
            double frac = 0;
            int after_dot = 0;
            bool past_dot = false;
            k = 0;
            while (tok.value[k]) {
                if (tok.value[k] == '.') { past_dot = true; k++; continue; }
                if (!past_dot) {
                    val = val * 10.0 + (double)(tok.value[k] - '0');
                } else {
                    frac = frac * 10.0 + (double)(tok.value[k] - '0');
                    after_dot++;
                }
                k++;
            }
            for (int i = 0; i < after_dot; i++) frac /= 10.0;
            return val_float(val + frac);
        } else {
            int64_t val = 0;
            k = 0;
            while (tok.value[k]) { val = val * 10 + (tok.value[k] - '0'); k++; }
            return val_int(val);
        }
    }

    if (tok.type == TRI_TOKEN_STRING) {
        return val_string(tok.value);
    }

    if (tok.type == TRI_TOKEN_KEYWORD) {
        if (str_eq(tok.value, "true")) return val_bool(true);
        if (str_eq(tok.value, "false")) return val_bool(false);
        if (str_eq(tok.value, "none")) return val_none();
    }

    if (tok.type == TRI_TOKEN_IDENT) {
        TriadVal *v = triad_interp_get_var(interp, tok.value);
        if (v) return v;
        return val_none();
    }

    if (tok.type == TRI_TOKEN_SYMBOL && tok.value[0] == '(') {
        TriadVal *v = parse_expr(lex, interp);
        lex_next(lex, &tok);
        return v;
    }

    return val_none();
}

static TriadVal *parse_unary(InterpLexer *lex, TriadInterp *interp) {
    TriToken tok;
    InterpLexer saved = *lex;
    if (lex_next(lex, &tok)) {
        if (tok.type == TRI_TOKEN_SYMBOL && tok.value[0] == '-') {
            TriadVal *operand = parse_unary(lex, interp);
            if (operand && operand->type == VAL_INT) {
                int64_t v = operand->ival;
                return val_int(-v);
            }
            if (operand && operand->type == VAL_FLOAT) {
                double v = operand->fval;
                return val_float(-v);
            }
            return val_none();
        }
        if (tok.type == TRI_TOKEN_SYMBOL && tok.value[0] == '!') {
            TriadVal *operand = parse_unary(lex, interp);
            return val_bool(!val_truthy(operand));
        }
    }
    *lex = saved;
    return parse_primary(lex, interp);
}

static int prec(const char *op) {
    if (op[0] == '|' && op[1] == '|') return 1;
    if (op[0] == '&' && op[1] == '&') return 2;
    if (op[0] == '=' && op[1] == '=') return 3;
    if (op[0] == '!' && op[1] == '=') return 3;
    if (op[0] == '<' && op[1] == '=') return 4;
    if (op[0] == '>' && op[1] == '=') return 4;
    if (op[0] == '<' && op[1] == 0) return 4;
    if (op[0] == '>' && op[1] == 0) return 4;
    switch (op[0]) {
    case '|': return 1;
    case '^': return 2;
    case '&': return 3;
    case '<': case '>': return 4;
    case '+': case '-': return 5;
    case '*': case '/': case '%': return 6;
    default: return 0;
    }
}

static TriadVal *parse_binop(InterpLexer *lex, TriadInterp *interp, int min_prec) {
    TriadVal *left = parse_unary(lex, interp);
    if (!left) return val_none();

    while (interp->running) {
        InterpLexer saved = *lex;
        TriToken tok;
        if (!lex_next(lex, &tok)) { *lex = saved; break; }
        if (tok.type != TRI_TOKEN_SYMBOL) { *lex = saved; break; }

        int p = prec(tok.value);
        if (p < min_prec || p == 0) { *lex = saved; break; }

        TriadVal *right = parse_binop(lex, interp, p + 1);
        if (!right) { *lex = saved; break; }

        char op0 = tok.value[0];
        char op1 = tok.value[1];

        if (op0 == '=' && op1 == '=') {
            left = val_bool(val_truthy(left) == val_truthy(right));
            continue;
        }
        if (op0 == '!' && op1 == '=') {
            left = val_bool(val_truthy(left) != val_truthy(right));
            continue;
        }
        if (op0 == '&' && op1 == '&') {
            left = val_bool(val_truthy(left) && val_truthy(right));
            continue;
        }
        if (op0 == '|' && op1 == '|') {
            left = val_bool(val_truthy(left) || val_truthy(right));
            continue;
        }

        if (left->type == VAL_INT && right->type == VAL_INT) {
            int64_t a = left->ival, b = right->ival;
            switch (op0) {
            case '+': left = val_int(a + b); break;
            case '-': left = val_int(a - b); break;
            case '*': left = val_int(a * b); break;
            case '/': if (b != 0) left = val_int(a / b); break;
            case '%': if (b != 0) left = val_int(a % b); break;
            case '<': left = val_bool(a < b); break;
            case '>': left = val_bool(a > b); break;
            case '&': left = val_int(a & b); break;
            case '|': left = val_int(a | b); break;
            case '^': left = val_int(a ^ b); break;
            default: break;
            }
        } else if (left->type == VAL_FLOAT || right->type == VAL_FLOAT) {
            double a = left->type == VAL_FLOAT ? left->fval : (double)left->ival;
            double b = right->type == VAL_FLOAT ? right->fval : (double)right->ival;
            switch (op0) {
            case '+': left = val_float(a + b); break;
            case '-': left = val_float(a - b); break;
            case '*': left = val_float(a * b); break;
            case '/': if (b != 0) left = val_float(a / b); break;
            case '<': left = val_bool(a < b); break;
            case '>': left = val_bool(a > b); break;
            default: break;
            }
        } else if (left->type == VAL_STRING && right->type == VAL_STRING && op0 == '+') {
            char buf[512];
            int k = 0;
            int j = 0;
            while (left->sval[j] && k < 511) { buf[k] = left->sval[j]; k++; j++; }
            j = 0;
            while (right->sval[j] && k < 511) { buf[k] = right->sval[j]; k++; j++; }
            buf[k] = 0;
            left = val_string(buf);
        }
    }
    return left;
}

static TriadVal *parse_expr(InterpLexer *lex, TriadInterp *interp) {
    return parse_binop(lex, interp, 1);
}

static TriadVal *builtin_call(TriadInterp *interp, const char *name,
                               TriadVal **args, int n_args) {
    (void)interp;
    if (str_eq(name, "print")) {
        for (int i = 0; i < n_args; i++) {
            triad_interp_print(args[i]);
            if (i < n_args - 1) triad_serial_putc(' ');
        }
        triad_serial_putc('\n');
        return val_none();
    }
    if (str_eq(name, "str") && n_args >= 1) {
        char buf[256];
        buf[0] = 0;
        if (args[0]->type == VAL_INT) {
            int64_t v = args[0]->ival;
            int pos = 0;
            if (v < 0) { buf[pos] = '-'; pos++; v = -v; }
            char tmp[21];
            int t = 20;
            tmp[t] = 0;
            if (v == 0) { tmp[--t] = '0'; }
            while (v > 0 && t > 0) { tmp[--t] = '0' + (v % 10); v /= 10; }
            int j = t;
            while (tmp[j] && pos < 255) { buf[pos] = tmp[j]; pos++; j++; }
            buf[pos] = 0;
        } else if (args[0]->type == VAL_FLOAT) {
            int pos = 0;
            double v = args[0]->fval;
            if (v < 0) { buf[pos] = '-'; pos++; v = -v; }
            int64_t ipart = (int64_t)v;
            double fpart = v - (double)ipart;
            int64_t tmp_i = ipart;
            char tmp[21];
            int t = 20;
            tmp[t] = 0;
            if (tmp_i == 0) tmp[--t] = '0';
            while (tmp_i > 0 && t > 0) { tmp[--t] = '0' + (tmp_i % 10); tmp_i /= 10; }
            int j = t;
            while (tmp[j] && pos < 255) { buf[pos] = tmp[j]; pos++; j++; }
            buf[pos] = '.'; pos++;
            for (int d = 0; d < 4 && pos < 255; d++) {
                fpart *= 10;
                int digit = (int)fpart;
                buf[pos] = '0' + digit;
                pos++;
                fpart -= digit;
            }
            buf[pos] = 0;
        } else if (args[0]->type == VAL_STRING) {
            int k = 0;
            while (args[0]->sval[k] && k < 255) { buf[k] = args[0]->sval[k]; k++; }
            buf[k] = 0;
        } else if (args[0]->type == VAL_BOOL) {
            const char *s = args[0]->bval ? "true" : "false";
            int k = 0;
            while (s[k] && k < 255) { buf[k] = s[k]; k++; }
            buf[k] = 0;
        }
        return val_string(buf);
    }
    if (str_eq(name, "int") && n_args >= 1) {
        if (args[0]->type == VAL_INT) return val_int(args[0]->ival);
        if (args[0]->type == VAL_FLOAT) return val_int((int64_t)args[0]->fval);
        if (args[0]->type == VAL_STRING) {
            int64_t v = 0;
            int k = 0;
            while (args[0]->sval[k] >= '0' && args[0]->sval[k] <= '9') {
                v = v * 10 + (args[0]->sval[k] - '0');
                k++;
            }
            return val_int(v);
        }
    }
    if (str_eq(name, "float") && n_args >= 1) {
        if (args[0]->type == VAL_FLOAT) return val_float(args[0]->fval);
        if (args[0]->type == VAL_INT) return val_float((double)args[0]->ival);
    }
    if (str_eq(name, "len") && n_args >= 1) {
        if (args[0]->type == VAL_STRING) {
            int k = 0;
            while (args[0]->sval[k]) k++;
            return val_int(k);
        }
        return val_int(0);
    }
    if (str_eq(name, "eq_read") && n_args >= 1) {
        if (args[0]->type == VAL_INT) {
            double u = 0.0;
            triad_eq_read((EqHwDomain)args[0]->ival, &u);
            return val_float(u);
        }
    }
    if (str_eq(name, "ai_query") && n_args >= 1) {
        if (args[0]->type == VAL_STRING) {
            char buf[1024];
            triad_ai_query(args[0]->sval, buf, sizeof(buf));
            return val_string(buf);
        }
    }
    return val_none();
}

static int exec_let(InterpLexer *lex, TriadInterp *interp) {
    TriToken tok;
    lex_next(lex, &tok);
    if (tok.type != TRI_TOKEN_IDENT) {
        interp_error(interp, "let: expected identifier");
        return -1;
    }
    char name[64];
    int k = 0;
    while (tok.value[k] && k < 63) { name[k] = tok.value[k]; k++; }
    name[k] = 0;

    lex_next(lex, &tok);
    if (tok.type != TRI_TOKEN_SYMBOL || tok.value[0] != '=') {
        interp_error(interp, "let: expected =");
        return -1;
    }

    TriadVal *val = parse_expr(lex, interp);
    triad_interp_set_var(interp, name, val);
    return 0;
}

static int exec_print(InterpLexer *lex, TriadInterp *interp) {
    TriToken tok;
    lex_next(lex, &tok);
    if (tok.type != TRI_TOKEN_SYMBOL || tok.value[0] != '(') {
        interp_error(interp, "print: expected (");
        return -1;
    }
    TriadVal *val = parse_expr(lex, interp);
    triad_interp_print(val);
    triad_serial_putc('\n');
    lex_next(lex, &tok);
    return 0;
}

static int skip_block(InterpLexer *lex, TriadInterp *interp) {
    TriToken tok;
    int depth = 1;
    while (interp->running && depth > 0) {
        if (!lex_next(lex, &tok)) break;
        if (tok.type == TRI_TOKEN_SYMBOL && tok.value[0] == '{') depth++;
        if (tok.type == TRI_TOKEN_SYMBOL && tok.value[0] == '}') depth--;
    }
    return 0;
}

static int exec_if(InterpLexer *lex, TriadInterp *interp) {
    TriToken tok;
    TriadVal *cond = parse_expr(lex, interp);
    lex_next(lex, &tok);
    if (tok.type != TRI_TOKEN_SYMBOL || tok.value[0] != '{') {
        interp_error(interp, "if: expected {");
        return -1;
    }
    if (val_truthy(cond)) {
        triad_interp_push_scope(interp);
        while (interp->running) {
            InterpLexer saved = *lex;
            if (!lex_next(lex, &tok)) break;
            if (tok.type == TRI_TOKEN_SYMBOL && tok.value[0] == '}') break;
            *lex = saved;
            exec_stmt(lex, interp);
        }
        triad_interp_pop_scope(interp);

        while (interp->running) {
            InterpLexer saved = *lex;
            if (!lex_next(lex, &tok)) break;
            if (tok.type == TRI_TOKEN_KEYWORD && (str_eq(tok.value, "else") || str_eq(tok.value, "elif"))) {
                if (str_eq(tok.value, "elif")) {
                    *lex = saved;
                    lex->pos = saved.pos;
                    exec_if(lex, interp);
                } else {
                    lex_next(lex, &tok);
                    if (tok.type == TRI_TOKEN_SYMBOL && tok.value[0] == '{') {
                        skip_block(lex, interp);
                    }
                }
                break;
            }
            *lex = saved;
            break;
        }
    } else {
        skip_block(lex, interp);

        InterpLexer saved = *lex;
        if (lex_next(lex, &tok)) {
            if (tok.type == TRI_TOKEN_KEYWORD && str_eq(tok.value, "else")) {
                lex_next(lex, &tok);
                if (tok.type == TRI_TOKEN_KEYWORD && str_eq(tok.value, "if")) {
                    *lex = saved;
                    lex->pos = saved.pos;
                    exec_if(lex, interp);
                } else if (tok.type == TRI_TOKEN_SYMBOL && tok.value[0] == '{') {
                    triad_interp_push_scope(interp);
                    while (interp->running) {
                        InterpLexer s2 = *lex;
                        if (!lex_next(lex, &tok)) break;
                        if (tok.type == TRI_TOKEN_SYMBOL && tok.value[0] == '}') break;
                        *lex = s2;
                        exec_stmt(lex, interp);
                    }
                    triad_interp_pop_scope(interp);
                }
            } else {
                *lex = saved;
            }
        }
    }
    return 0;
}

static int exec_while(InterpLexer *lex, TriadInterp *interp) {
    int cond_pos = lex->pos;
    TriToken tok;

    while (interp->running) {
        lex->pos = cond_pos;
        TriadVal *cond = parse_expr(lex, interp);
        if (!val_truthy(cond)) break;

        lex_next(lex, &tok);
        if (tok.type != TRI_TOKEN_SYMBOL || tok.value[0] != '{') {
            interp_error(interp, "while: expected {");
            return -1;
        }

        int body_start = lex->pos;
        triad_interp_push_scope(interp);
        while (interp->running) {
            InterpLexer saved = *lex;
            if (!lex_next(lex, &tok)) break;
            if (tok.type == TRI_TOKEN_SYMBOL && tok.value[0] == '}') break;
            *lex = saved;
            exec_stmt(lex, interp);
        }
        triad_interp_pop_scope(interp);
    }

    while (interp->running) {
        if (!lex_next(lex, &tok)) break;
        if (tok.type == TRI_TOKEN_SYMBOL && tok.value[0] == '}') break;
    }
    return 0;
}

static int exec_for(InterpLexer *lex, TriadInterp *interp) {
    TriToken tok;
    lex_next(lex, &tok);
    if (tok.type != TRI_TOKEN_IDENT) {
        interp_error(interp, "for: expected var name");
        return -1;
    }
    char var_name[64];
    int k = 0;
    while (tok.value[k] && k < 63) { var_name[k] = tok.value[k]; k++; }
    var_name[k] = 0;

    lex_next(lex, &tok);
    if (tok.type != TRI_TOKEN_KEYWORD || !str_eq(tok.value, "in")) {
        interp_error(interp, "for: expected in");
        return -1;
    }

    TriadVal *iter_val = parse_expr(lex, interp);
    lex_next(lex, &tok);
    if (tok.type != TRI_TOKEN_SYMBOL || tok.value[0] != '{') {
        interp_error(interp, "for: expected {");
        return -1;
    }

    int body_start = lex->pos;
    int64_t count = 0;
    if (iter_val && iter_val->type == VAL_INT) count = iter_val->ival;

    for (int64_t i = 0; i < count && interp->running; i++) {
        triad_interp_set_var(interp, var_name, val_int(i));
        InterpLexer body_lex = *lex;
        body_lex.pos = body_start;
        triad_interp_push_scope(interp);
        while (interp->running) {
            InterpLexer saved = body_lex;
            if (!lex_next(&body_lex, &tok)) break;
            if (tok.type == TRI_TOKEN_SYMBOL && tok.value[0] == '}') break;
            body_lex = saved;
            exec_stmt(&body_lex, interp);
        }
        triad_interp_pop_scope(interp);
    }

    while (interp->running) {
        if (!lex_next(lex, &tok)) break;
        if (tok.type == TRI_TOKEN_SYMBOL && tok.value[0] == '}') break;
    }
    return 0;
}

static int exec_stmt(InterpLexer *lex, TriadInterp *interp) {
    TriToken tok;
    InterpLexer saved = *lex;
    if (!lex_next(lex, &tok)) return -1;

    if (tok.type == TRI_TOKEN_KEYWORD) {
        if (str_eq(tok.value, "let") || str_eq(tok.value, "const")) {
            *lex = saved;
            lex->pos = saved.pos;
            return exec_let(lex, interp);
        }
        if (str_eq(tok.value, "if")) {
            return exec_if(lex, interp);
        }
        if (str_eq(tok.value, "for")) {
            return exec_for(lex, interp);
        }
        if (str_eq(tok.value, "while")) {
            return exec_while(lex, interp);
        }
    }

    if (tok.type == TRI_TOKEN_IDENT) {
        if (str_eq(tok.value, "print")) {
            return exec_print(lex, interp);
        }
        TriadVal *args[16];
        int n_args = 0;
        TriToken t2;
        InterpLexer s2 = *lex;
        if (lex_next(lex, &t2)) {
            if (t2.type == TRI_TOKEN_SYMBOL && t2.value[0] == '(') {
                while (interp->running) {
                    InterpLexer s3 = *lex;
                    if (!lex_next(lex, &t2)) break;
                    if (t2.type == TRI_TOKEN_SYMBOL && t2.value[0] == ')') break;
                    *lex = s3;
                    if (n_args < 16) {
                        args[n_args] = parse_expr(lex, interp);
                        n_args++;
                    }
                    lex_next(lex, &t2);
                    if (t2.type == TRI_TOKEN_SYMBOL && t2.value[0] == ')') break;
                }
                builtin_call(interp, tok.value, args, n_args);
                return 0;
            }
        }
        *lex = s2;
    }

    *lex = saved;
    parse_expr(lex, interp);
    return 0;
}

int triad_interp_run(TriadInterp *interp, const char *src) {
    InterpLexer lex;
    lex_init(&lex, src);

    while (interp->running) {
        InterpLexer saved = lex;
        TriToken tok;
        if (!lex_next(&lex, &tok)) break;
        if (tok.type == TRI_TOKEN_EOF) break;
        if (tok.type == TRI_TOKEN_SYMBOL && tok.value[0] == ';') continue;
        lex = saved;
        exec_stmt(&lex, interp);
    }

    return interp->error ? -1 : 0;
}

TriadVal *triad_interp_eval(TriadInterp *interp, const char *expr) {
    InterpLexer lex;
    lex_init(&lex, expr);
    return parse_expr(&lex, interp);
}