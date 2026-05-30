from __future__ import annotations
from compiler.ir import *
_C_PREAMBLE = '#include "triad_rt.h"\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include <math.h>\n\nstatic double _solver_dget(TriadValue cfg, const char *key, double def) {\n    TriadValue v = triad_dict_get(cfg.as.dval, triad_str_new(key));\n    if (v.tag == TRIAD_NONE) return def;\n    return (v.tag == TRIAD_INT) ? (double)v.as.ival : v.as.fval;\n}\n\nstatic int32_t _solver_iget(TriadValue cfg, const char *key, int32_t def) {\n    TriadValue v = triad_dict_get(cfg.as.dval, triad_str_new(key));\n    if (v.tag == TRIAD_NONE) return def;\n    return (v.tag == TRIAD_INT) ? (int32_t)v.as.ival : (int32_t)v.as.fval;\n}\n\nstatic int64_t _solver_lget(TriadValue cfg, const char *key, int64_t def) {\n    TriadValue v = triad_dict_get(cfg.as.dval, triad_str_new(key));\n    if (v.tag == TRIAD_NONE) return def;\n    return (v.tag == TRIAD_INT) ? v.as.ival : (int64_t)v.as.fval;\n}\n\nstatic TriadValue _triad_solver_solve(TriadValue cfg) {\n    static double _nu_def[] = {2.0, 0.5, 0.1};\n    static double _lam_def[] = {-0.3, -0.2, -0.1};\n    TriadSolverC p;\n    memset(&p, 0, sizeof(p));\n    p.N      = _solver_iget(cfg, "N", 128);\n    p.L      = _solver_dget(cfg, "L", 32.0);\n    p.dt     = _solver_dget(cfg, "dt", 0.005);\n    p.T      = _solver_dget(cfg, "T", 2.0);\n    p.hbar   = _solver_dget(cfg, "hbar", 1.0);\n    p.m      = _solver_dget(cfg, "m", 1.0);\n    p.omega  = _solver_dget(cfg, "omega", 0.05);\n    p.Lambda = _solver_dget(cfg, "Lambda", -0.5);\n    p.alpha  = _solver_dget(cfg, "alpha", 0.15);\n    p.sigma  = _solver_dget(cfg, "sigma", 1.5);\n    p.Gamma  = _solver_dget(cfg, "Gamma", 0.05);\n    p.f_FDT  = _solver_dget(cfg, "f_FDT", 0.002);\n    p.M      = 3;\n    p.nu     = _nu_def;\n    p.lam    = _lam_def;\n    p.mode   = _solver_iget(cfg, "mode", 2);\n    p.seed   = (uint64_t)_solver_lget(cfg, "seed", 42);\n    {\n        TriadValue vv = triad_dict_get(cfg.as.dval, triad_str_new("V_ext"));\n        if (vv.tag == TRIAD_STRING) p.V_ext = vv.as.sval->data;\n        else p.V_ext = "harmonic";\n    }\n\n    TriadSolverResult r = triad_solve_1d(&p);\n\n    double norm = 0, peak = 0;\n    for (int32_t i = 0; i < p.N; i++) {\n        norm += r.density_final[i] * r.dx;\n        if (r.density_final[i] > peak) peak = r.density_final[i];\n    }\n\n    TriadDict *rd = triad_dict_new();\n    triad_dict_set(rd, triad_str_new("norm"),   TRIAD_FLOAT(norm));\n    triad_dict_set(rd, triad_str_new("peak"),   TRIAD_FLOAT(peak));\n    triad_dict_set(rd, triad_str_new("N"),      TRIAD_INT(p.N));\n    triad_dict_set(rd, triad_str_new("dx"),     TRIAD_FLOAT(r.dx));\n\n    {\n        TriadList *dlist = triad_list_new_cap(p.N);\n        for (int32_t i = 0; i < p.N; i++)\n            triad_list_push(dlist, TRIAD_FLOAT(r.density_final[i]));\n        triad_dict_set(rd, triad_str_new("density"),\n                       (TriadValue){.tag = TRIAD_LIST, .as = {.lval = dlist}});\n    }\n    {\n        TriadList *xlist = triad_list_new_cap(p.N);\n        for (int32_t i = 0; i < p.N; i++)\n            triad_list_push(xlist, TRIAD_FLOAT(r.x[i]));\n        triad_dict_set(rd, triad_str_new("x"),\n                       (TriadValue){.tag = TRIAD_LIST, .as = {.lval = xlist}});\n    }\n\n    triad_solver_result_free(&r);\n    return (TriadValue){.tag = TRIAD_DICT, .as = {.dval = rd}};\n}\n\n/* ═══ ML builtins (tensor, linear, sequential, adam, etc.) ═══ */\n#include "triad_ml.h"\n\nstatic double _ml_vf(TriadValue v) {\n    return (v.tag == TRIAD_INT) ? (double)v.as.ival : v.as.fval;\n}\n\n/* tensor(list, requires_grad?) -> TriadTensor* via TRIAD_PTR */\nstatic TriadValue _ml_tensor(TriadValue data_v, TriadValue rg_v) {\n    int rg = (rg_v.tag == TRIAD_BOOL) ? rg_v.as.bval : 0;\n    if (data_v.tag == TRIAD_LIST) {\n        TriadList *l = data_v.as.lval;\n        /* Check if 2D (list of lists) */\n        if (l->len > 0 && l->items[0].tag == TRIAD_LIST) {\n            int32_t rows = l->len;\n            int32_t cols = triad_list_len(l->items[0].as.lval);\n            int32_t shape[] = {rows, cols};\n            TriadTensor *t = triad_tensor_new(2, shape, rg);\n            for (int32_t i = 0; i < rows; i++) {\n                TriadList *row = l->items[i].as.lval;\n                for (int32_t j = 0; j < cols; j++)\n                    t->data[i * cols + j] = _ml_vf(row->items[j]);\n            }\n            return TRIAD_PTR_VAL(t);\n        }\n        /* 1D */\n        int32_t shape[] = {l->len};\n        TriadTensor *t = triad_tensor_new(1, shape, rg);\n        for (int32_t i = 0; i < l->len; i++)\n            t->data[i] = _ml_vf(l->items[i]);\n        return TRIAD_PTR_VAL(t);\n    }\n    /* scalar */\n    TriadTensor *t = triad_tensor_scalar(_ml_vf(data_v), rg);\n    return TRIAD_PTR_VAL(t);\n}\n\nstatic TriadValue _ml_tensor_zeros(TriadValue rows_v, TriadValue cols_v, TriadValue rg_v) {\n    int rg = (rg_v.tag == TRIAD_BOOL) ? rg_v.as.bval : 0;\n    if (cols_v.tag == TRIAD_NONE) {\n        int32_t s[] = {(int32_t)rows_v.as.ival};\n        return TRIAD_PTR_VAL(triad_tensor_zeros(1, s, rg));\n    }\n    int32_t s[] = {(int32_t)rows_v.as.ival, (int32_t)cols_v.as.ival};\n    return TRIAD_PTR_VAL(triad_tensor_zeros(2, s, rg));\n}\n\nstatic TriadValue _ml_tensor_randn(TriadValue rows_v, TriadValue cols_v, TriadValue rg_v) {\n    int rg = (rg_v.tag == TRIAD_BOOL) ? rg_v.as.bval : 0;\n    if (cols_v.tag == TRIAD_NONE) {\n        int32_t s[] = {(int32_t)rows_v.as.ival};\n        return TRIAD_PTR_VAL(triad_tensor_randn(1, s, rg));\n    }\n    int32_t s[] = {(int32_t)rows_v.as.ival, (int32_t)cols_v.as.ival};\n    return TRIAD_PTR_VAL(triad_tensor_randn(2, s, rg));\n}\n\nstatic TriadValue _ml_linear_new(TriadValue in_v, TriadValue out_v) {\n    return TRIAD_PTR_VAL(triad_linear_new((int32_t)in_v.as.ival, (int32_t)out_v.as.ival, 1));\n}\n\nstatic TriadValue _ml_linear_forward(TriadValue layer_v, TriadValue x_v) {\n    return TRIAD_PTR_VAL(triad_linear_forward((TriadLinear*)layer_v.as.ptr, (TriadTensor*)x_v.as.ptr));\n}\n\nstatic TriadValue _ml_relu(TriadValue x_v) {\n    return TRIAD_PTR_VAL(triad_tensor_relu((TriadTensor*)x_v.as.ptr));\n}\n\nstatic TriadValue _ml_sigmoid(TriadValue x_v) {\n    return TRIAD_PTR_VAL(triad_tensor_sigmoid((TriadTensor*)x_v.as.ptr));\n}\n\nstatic TriadValue _ml_tanh(TriadValue x_v) {\n    return TRIAD_PTR_VAL(triad_tensor_tanh((TriadTensor*)x_v.as.ptr));\n}\n\nstatic TriadValue _ml_softmax(TriadValue x_v) {\n    return TRIAD_PTR_VAL(triad_tensor_softmax((TriadTensor*)x_v.as.ptr));\n}\n\nstatic TriadValue _ml_mse_loss(TriadValue pred_v, TriadValue tgt_v) {\n    return TRIAD_PTR_VAL(triad_tensor_mse_loss((TriadTensor*)pred_v.as.ptr, (TriadTensor*)tgt_v.as.ptr));\n}\n\nstatic TriadValue _ml_cross_entropy(TriadValue logits_v, TriadValue tgt_v) {\n    return TRIAD_PTR_VAL(triad_tensor_cross_entropy((TriadTensor*)logits_v.as.ptr, (TriadTensor*)tgt_v.as.ptr));\n}\n\nstatic TriadValue _ml_backward(TriadValue t_v) {\n    triad_tensor_backward((TriadTensor*)t_v.as.ptr, NULL);\n    return TRIAD_NONE_VAL;\n}\n\nstatic TriadValue _ml_tensor_data(TriadValue t_v, TriadValue idx_v) {\n    TriadTensor *t = (TriadTensor*)t_v.as.ptr;\n    return TRIAD_FLOAT(t->data[(int32_t)idx_v.as.ival]);\n}\n\nstatic TriadValue _ml_tensor_item(TriadValue t_v) {\n    TriadTensor *t = (TriadTensor*)t_v.as.ptr;\n    return TRIAD_FLOAT(t->data[0]);\n}\n\n/* Sequential: seq_new(n), seq_add(seq, type, layer), seq_forward(seq, x) */\nstatic TriadValue _ml_sequential_new(TriadValue n_v) {\n    return TRIAD_PTR_VAL(triad_sequential_new((int32_t)n_v.as.ival));\n}\n\nstatic TriadValue _ml_sequential_set(TriadValue seq_v, TriadValue idx_v,\n                                      TriadValue type_v, TriadValue layer_v) {\n    triad_sequential_set((TriadSequential*)seq_v.as.ptr, (int32_t)idx_v.as.ival,\n                          (TriadLayerType)(int32_t)type_v.as.ival,\n                          (type_v.as.ival == TRIAD_LAYER_LINEAR) ? layer_v.as.ptr : NULL);\n    return TRIAD_NONE_VAL;\n}\n\nstatic TriadValue _ml_sequential_forward(TriadValue seq_v, TriadValue x_v) {\n    return TRIAD_PTR_VAL(triad_sequential_forward((TriadSequential*)seq_v.as.ptr,\n                                                    (TriadTensor*)x_v.as.ptr));\n}\n\n/* Adam: adam_new(seq, lr), adam_step(adam), adam_zero_grad(adam) */\nstatic TriadValue _ml_adam_new(TriadValue seq_v, TriadValue lr_v) {\n    TriadTensor *params[256];\n    int32_t np = triad_sequential_params((TriadSequential*)seq_v.as.ptr, params, 256);\n    return TRIAD_PTR_VAL(triad_adam_new(params, np, _ml_vf(lr_v), 0.9, 0.999, 1e-8));\n}\n\nstatic TriadValue _ml_sgd_new(TriadValue seq_v, TriadValue lr_v) {\n    TriadTensor *params[256];\n    int32_t np = triad_sequential_params((TriadSequential*)seq_v.as.ptr, params, 256);\n    return TRIAD_PTR_VAL(triad_sgd_new(params, np, _ml_vf(lr_v), 0.0));\n}\n\nstatic TriadValue _ml_adam_step(TriadValue opt_v) {\n    triad_adam_step((TriadAdam*)opt_v.as.ptr);\n    return TRIAD_NONE_VAL;\n}\n\nstatic TriadValue _ml_adam_zero_grad(TriadValue opt_v) {\n    triad_adam_zero_grad((TriadAdam*)opt_v.as.ptr);\n    return TRIAD_NONE_VAL;\n}\n\nstatic TriadValue _ml_sgd_step(TriadValue opt_v) {\n    triad_sgd_step((TriadSGD*)opt_v.as.ptr);\n    return TRIAD_NONE_VAL;\n}\n\nstatic TriadValue _ml_sgd_zero_grad(TriadValue opt_v) {\n    triad_sgd_zero_grad((TriadSGD*)opt_v.as.ptr);\n    return TRIAD_NONE_VAL;\n}\n\nstatic TriadValue _ml_tensor_add(TriadValue a_v, TriadValue b_v) {\n    return TRIAD_PTR_VAL(triad_tensor_add((TriadTensor*)a_v.as.ptr, (TriadTensor*)b_v.as.ptr));\n}\n\nstatic TriadValue _ml_tensor_sub(TriadValue a_v, TriadValue b_v) {\n    return TRIAD_PTR_VAL(triad_tensor_sub((TriadTensor*)a_v.as.ptr, (TriadTensor*)b_v.as.ptr));\n}\n\nstatic TriadValue _ml_tensor_mul(TriadValue a_v, TriadValue b_v) {\n    return TRIAD_PTR_VAL(triad_tensor_mul((TriadTensor*)a_v.as.ptr, (TriadTensor*)b_v.as.ptr));\n}\n\nstatic TriadValue _ml_tensor_matmul(TriadValue a_v, TriadValue b_v) {\n    return TRIAD_PTR_VAL(triad_tensor_matmul((TriadTensor*)a_v.as.ptr, (TriadTensor*)b_v.as.ptr));\n}\n\nstatic TriadValue _ml_tensor_sum(TriadValue a_v) {\n    return TRIAD_PTR_VAL(triad_tensor_sum((TriadTensor*)a_v.as.ptr));\n}\n\nstatic TriadValue _ml_tensor_mean(TriadValue a_v) {\n    return TRIAD_PTR_VAL(triad_tensor_mean((TriadTensor*)a_v.as.ptr));\n}\n\nstatic TriadValue _ml_tensor_print(TriadValue t_v) {\n    TriadTensor *t = (TriadTensor*)t_v.as.ptr;\n    if (t->ndim == 0 || t->size == 1) {\n        printf("%.6f\\n", t->data[0]);\n    } else {\n        printf("[");\n        for (int64_t i = 0; i < t->size && i < 20; i++) {\n            if (i > 0) printf(", ");\n            printf("%.4f", t->data[i]);\n        }\n        if (t->size > 20) printf(", ...");\n        printf("]\\n");\n    }\n    return TRIAD_NONE_VAL;\n}\n\n'
_C_MAIN_HEADER = '\nint main(int argc, char **argv) {\n    (void)argc; (void)argv;\n'

class CCodeGen:

    def __init__(self):
        self._lines: list[str] = []
        self._indent: int = 0
        self._temp_counter: int = 0
        self._func_forward_decls: list[str] = []
        self._func_defs: list[str] = []
        self._in_function: bool = False
        self._loop_depth: int = 0
        self._scope_stack: list[set[str]] = []
        self._closures: dict[str, list[str]] = {}
        self._top_level_fns: set[str] = set()
        self._classes: dict[str, IRClassDecl] = {}

    def _fresh(self, prefix: str='_t') -> str:
        self._temp_counter += 1
        return f'{prefix}{self._temp_counter}'

    def _emit(self, line: str):
        self._lines.append('    ' * self._indent + line)

    def _emit_raw(self, line: str):
        self._lines.append(line)

    def _collect_names_expr(self, node: IRNode, names: set[str]):
        if isinstance(node, IRIdent):
            names.add(node.name)
        elif isinstance(node, IRBinOp):
            self._collect_names_expr(node.left, names)
            self._collect_names_expr(node.right, names)
        elif isinstance(node, IRUnaryOp):
            self._collect_names_expr(node.operand, names)
        elif isinstance(node, IRCall):
            if isinstance(node.func, IRIdent):
                names.add(node.func.name)
            else:
                self._collect_names_expr(node.func, names)
            for a in node.args:
                self._collect_names_expr(a, names)
        elif isinstance(node, IRMethodCall):
            self._collect_names_expr(node.obj, names)
            for a in node.args:
                self._collect_names_expr(a, names)
        elif isinstance(node, IRIndex):
            self._collect_names_expr(node.obj, names)
            self._collect_names_expr(node.index, names)
        elif isinstance(node, IRSlice):
            self._collect_names_expr(node.obj, names)
            if node.start:
                self._collect_names_expr(node.start, names)
            if node.end:
                self._collect_names_expr(node.end, names)
            if node.step:
                self._collect_names_expr(node.step, names)
        elif isinstance(node, IRField):
            self._collect_names_expr(node.obj, names)
        elif isinstance(node, IRList):
            for e in node.elements:
                self._collect_names_expr(e, names)
        elif isinstance(node, IRMap):
            for k, v in node.pairs:
                self._collect_names_expr(k, names)
                self._collect_names_expr(v, names)
        elif isinstance(node, IRFString):
            for text, expr in node.parts:
                if expr is not None:
                    self._collect_names_expr(expr, names)
        elif isinstance(node, IRListComp):
            self._collect_names_expr(node.iter, names)
            self._collect_names_expr(node.expr, names)
            if node.condition:
                self._collect_names_expr(node.condition, names)
        elif isinstance(node, IRAssignExpr):
            self._collect_names_expr(node.value, names)

    def _collect_names_stmt(self, node: IRNode, names: set[str]):
        if isinstance(node, (IRLet, IRConst)):
            if node.value is not None:
                self._collect_names_expr(node.value, names)
        elif isinstance(node, IRAssign):
            self._collect_names_expr(node.value, names)
            if isinstance(node.target, IRIdent):
                pass
            else:
                self._collect_names_expr(node.target, names)
        elif isinstance(node, IRExprStmt):
            self._collect_names_expr(node.expr, names)
        elif isinstance(node, IRReturn):
            if node.value:
                self._collect_names_expr(node.value, names)
        elif isinstance(node, IRIf):
            self._collect_names_expr(node.condition, names)
            for stmt in node.then_body:
                self._collect_names_stmt(stmt, names)
            for cond_ir, body in node.elif_clauses:
                self._collect_names_expr(cond_ir, names)
                for stmt in body:
                    self._collect_names_stmt(stmt, names)
            for stmt in node.else_body or []:
                self._collect_names_stmt(stmt, names)
        elif isinstance(node, IRFor):
            self._collect_names_expr(node.iter, names)
            for stmt in node.body:
                self._collect_names_stmt(stmt, names)
        elif isinstance(node, IRWhile):
            self._collect_names_expr(node.condition, names)
            for stmt in node.body:
                self._collect_names_stmt(stmt, names)
        elif isinstance(node, IRTryCatch):
            for stmt in node.body:
                self._collect_names_stmt(stmt, names)
            for stmt in node.catch_body or []:
                self._collect_names_stmt(stmt, names)
            for stmt in node.finally_body or []:
                self._collect_names_stmt(stmt, names)
        elif isinstance(node, IRThrow):
            if node.value:
                self._collect_names_expr(node.value, names)
        elif isinstance(node, IRDestructLet):
            self._collect_names_expr(node.value, names)

    def _free_vars_fn(self, fn_node: IRFunction, outer_bound: set[str]) -> list[str]:
        fn_bound = set(fn_node.params)
        local_names: set[str] = set()
        for stmt in fn_node.body:
            if isinstance(stmt, (IRLet, IRConst)):
                local_names.add(stmt.name)
            elif isinstance(stmt, IRDestructLet):
                local_names.update(stmt.names)
            elif isinstance(stmt, IRFunction):
                local_names.add(stmt.name)
        fn_bound = fn_bound | local_names
        referenced: set[str] = set()
        for stmt in fn_node.body:
            if isinstance(stmt, IRFunction):
                continue
            self._collect_names_stmt(stmt, referenced)
        free = referenced - fn_bound
        return sorted(free)

    def generate(self, module: IRModule) -> str:
        self._lines = []
        self._func_forward_decls = []
        self._func_defs = []
        self._file_scope_decls: list[str] = []
        self._temp_counter = 0
        self._in_function = False
        self._loop_depth = 0
        for node in module.body:
            if isinstance(node, IRFunction):
                fwd = f'static TriadValue {self._c_fn_name(node.name)}(int32_t _nargs, TriadValue *_args);'
                if fwd not in self._func_forward_decls:
                    self._func_forward_decls.append(fwd)
                self._top_level_fns.add(node.name)
        for node in module.body:
            if isinstance(node, IRFunction):
                self._emit_function(node)
            elif isinstance(node, IRClassDecl):
                self._emit_class(node)
            elif isinstance(node, IRImport):
                pass
            else:
                self._in_function = False
                self._emit_top_level_stmt(node)
        parts = [_C_PREAMBLE]
        for decl in self._file_scope_decls:
            parts.append(decl + '\n')
        for fwd in self._func_forward_decls:
            parts.append(fwd + '\n')
        parts.append('\n')
        for fdef in self._func_defs:
            parts.append(fdef)
            parts.append('\n')
        parts.append(_C_MAIN_HEADER)
        for line in self._lines:
            parts.append('    ' + line + '\n')
        parts.append('    return 0;\n}\n')
        return ''.join(parts)

    def _emit_top_level_stmt(self, node: IRNode):
        if isinstance(node, (IRLet, IRConst)):
            decl = self._gen_toplevel_let(node)
            self._emit(decl)
        elif isinstance(node, IRAssign):
            self._emit(self._gen_assign(node))
        elif isinstance(node, IRExprStmt):
            val = self._gen_expr(node.expr)
            self._emit(f'(void){val};')
        elif isinstance(node, IRIf):
            self._gen_if(node)
        elif isinstance(node, IRFor):
            self._gen_for(node)
        elif isinstance(node, IRWhile):
            self._gen_while(node)
        elif isinstance(node, IRTryCatch):
            self._gen_try_catch(node)
        elif isinstance(node, IRThrow):
            self._gen_throw(node)
        elif isinstance(node, IRDestructLet):
            self._gen_destruct_let(node)
        elif isinstance(node, IRReturn):
            val = self._gen_expr(node.value) if node.value else 'TRIAD_NONE_VAL'
            self._emit(f'return {val};')
        elif isinstance(node, (IRTypeDecl, IREntityDecl, IRWorldDecl, IRRegDecl, IRObserve, IRRun)):
            pass

    def _emit_stmt(self, node: IRNode):
        if isinstance(node, (IRLet, IRConst)):
            self._emit(self._gen_local_let(node))
        elif isinstance(node, IRAssign):
            self._emit(self._gen_assign(node))
        elif isinstance(node, IRExprStmt):
            val = self._gen_expr(node.expr)
            self._emit(f'(void){val};')
        elif isinstance(node, IRReturn):
            val = self._gen_expr(node.value) if node.value else 'TRIAD_NONE_VAL'
            self._emit(f'return {val};')
        elif isinstance(node, IRIf):
            self._gen_if(node)
        elif isinstance(node, IRFor):
            self._gen_for(node)
        elif isinstance(node, IRWhile):
            self._gen_while(node)
        elif isinstance(node, IRBreak):
            self._emit('break;')
        elif isinstance(node, IRContinue):
            self._emit('continue;')
        elif isinstance(node, IRTryCatch):
            self._gen_try_catch(node)
        elif isinstance(node, IRThrow):
            self._gen_throw(node)
        elif isinstance(node, IRDestructLet):
            self._gen_destruct_let(node)
        elif isinstance(node, IRYield):
            self._emit_yield(node)
        elif isinstance(node, (IRTypeDecl, IREntityDecl, IRWorldDecl, IRRegDecl, IRObserve, IRRun)):
            pass

    @staticmethod
    def _c_fn_name(name: str) -> str:
        return f'_triad_fn_{name}'

    def _has_yield(self, stmts: list) -> bool:
        for s in stmts:
            if isinstance(s, IRYield):
                return True
            if isinstance(s, IRFunction):
                continue
            if hasattr(s, 'body') and isinstance(s.body, list):
                if self._has_yield(s.body):
                    return True
            if hasattr(s, 'then_body') and isinstance(s.then_body, list):
                if self._has_yield(s.then_body):
                    return True
            if hasattr(s, 'else_body') and s.else_body:
                if self._has_yield(s.else_body):
                    return True
            if hasattr(s, 'elif_clauses'):
                for cond, body in s.elif_clauses or []:
                    if self._has_yield(body):
                        return True
            if hasattr(s, 'catch_body') and s.catch_body:
                if self._has_yield(s.catch_body):
                    return True
        return False

    def _emit_yield(self, node: IRYield):
        val = self._gen_expr(node.value) if node.value else 'TRIAD_NONE_VAL'
        self._emit(f'triad_list_push(_yield_list, {val});')

    def _emit_function(self, node: IRFunction, outer_bound: set[str] | None=None):
        fname = self._c_fn_name(node.name)
        saved = self._in_function
        self._in_function = True
        body_lines: list[str] = []
        saved_lines = self._lines
        saved_indent = self._indent
        self._lines = body_lines
        self._indent = 1
        if outer_bound is None:
            outer_bound = set()
        fn_scope = outer_bound | set(node.params)
        for stmt in node.body:
            if isinstance(stmt, (IRLet, IRConst)):
                fn_scope.add(stmt.name)
            elif isinstance(stmt, IRDestructLet):
                fn_scope.update(stmt.names)
        for i, pname in enumerate(node.params):
            self._emit(f'TriadValue {pname} = (_nargs > {i}) ? _args[{i}] : TRIAD_NONE_VAL;')
        is_gen = self._has_yield(node.body)
        if is_gen:
            self._emit('TriadList *_yield_list = triad_list_new();')
        nested_fns = [s for s in node.body if isinstance(s, IRFunction)]
        for nested in nested_fns:
            free = self._free_vars_fn(nested, fn_scope)
            nparams = len(nested.params)
            ncaptured = len(free)
            nested_fname = self._c_fn_name(nested.name)
            nshift = nparams
            self._closures[nested.name] = free
            saved2 = self._lines
            saved2_indent = self._indent
            nested_lines: list[str] = []
            self._lines = nested_lines
            self._indent = 1
            nested_is_gen = self._has_yield(nested.body)
            for i, pname in enumerate(nested.params):
                self._emit(f'TriadValue {pname} = (_nargs > {i}) ? _args[{i}] : TRIAD_NONE_VAL;')
            for ci, cname in enumerate(free):
                self._emit(f'TriadValue {self._sanitize(cname)} = (_nargs > {nshift + ci}) ? _args[{nshift + ci}] : TRIAD_NONE_VAL;')
            if nested_is_gen:
                self._emit('TriadList *_yield_list = triad_list_new();')
            for stmt in nested.body:
                if isinstance(stmt, IRFunction):
                    continue
                self._emit_stmt(stmt)
            if nested_is_gen:
                self._emit('return (TriadValue){.tag = TRIAD_LIST, .as = {.lval = _yield_list}};')
            else:
                self._emit('return TRIAD_NONE_VAL;')
            self._lines = saved2
            self._indent = saved2_indent
            defn = f'static TriadValue {nested_fname}(int32_t _nargs, TriadValue *_args) {{\n'
            for line in nested_lines:
                defn += line + '\n'
            defn += '}\n'
            self._func_defs.append(defn)
            self._func_forward_decls.append(f'static TriadValue {nested_fname}(int32_t _nargs, TriadValue *_args);')
        for stmt in node.body:
            if isinstance(stmt, IRFunction):
                nested = stmt
                free = self._closures.get(nested.name, [])
                nested_fname = self._c_fn_name(nested.name)
                if free:
                    cl_var = self._fresh('_cl')
                    self._emit(f'TriadClosure *{cl_var} = triad_closure_new({nested_fname}, {len(free)});')
                    for ci, vname in enumerate(free):
                        self._emit(f'{cl_var}->captured[{ci}] = {self._sanitize(vname)};')
                    self._emit(f'TriadValue {self._sanitize(nested.name)} = (TriadValue){{.tag = TRIAD_CLOSURE, .as = {{.cval = {cl_var}}}}};')
                else:
                    self._emit(f'TriadValue {self._sanitize(nested.name)} = (TriadValue){{.tag = TRIAD_CLOSURE, .as = {{.cval = triad_closure_new({nested_fname}, 0)}}}};')
            else:
                self._emit_stmt(stmt)
        if is_gen:
            self._emit('return (TriadValue){.tag = TRIAD_LIST, .as = {.lval = _yield_list}};')
        else:
            self._emit('return TRIAD_NONE_VAL;')
        self._lines = saved_lines
        self._indent = saved_indent
        self._in_function = saved
        defn = f'static TriadValue {fname}(int32_t _nargs, TriadValue *_args) {{\n'
        for line in body_lines:
            defn += line + '\n'
        defn += '}\n'
        self._func_defs.append(defn)

    def _emit_class(self, node: IRClassDecl):
        self._classes[node.name] = node
        for method in node.methods:
            mname = f'_triad_cls_{node.name}_{method.name}'
            self._top_level_fns.add(mname)
            fwd = f'static TriadValue {mname}(int32_t _nargs, TriadValue *_args);'
            if fwd not in self._func_forward_decls:
                self._func_forward_decls.append(fwd)
            saved = self._in_function
            self._in_function = True
            body_lines: list[str] = []
            saved_lines = self._lines
            saved_indent = self._indent
            self._lines = body_lines
            self._indent = 1
            self._emit(f'TriadValue self = (_nargs > 0) ? _args[0] : TRIAD_NONE_VAL;')
            for i, pname in enumerate(method.params):
                self._emit(f'TriadValue {self._sanitize(pname)} = (_nargs > {i + 1}) ? _args[{i + 1}] : TRIAD_NONE_VAL;')
            for stmt in method.body:
                self._emit_stmt(stmt)
            self._emit('return TRIAD_NONE_VAL;')
            self._lines = saved_lines
            self._indent = saved_indent
            self._in_function = saved
            defn = f'static TriadValue {mname}(int32_t _nargs, TriadValue *_args) {{\n'
            for line in body_lines:
                defn += line + '\n'
            defn += '}\n'
            self._func_defs.append(defn)

    def _gen_toplevel_let(self, node: IRNode) -> str:
        if isinstance(node, IRLet):
            name = self._sanitize(node.name)
            if node.value is not None:
                val = self._gen_expr(node.value)
                self._file_scope_decls.append(f'static TriadValue {name};')
                return f'{name} = {val};'
            self._file_scope_decls.append(f'static TriadValue {name} = TRIAD_NONE_VAL;')
            return ''
        elif isinstance(node, IRConst):
            name = self._sanitize(node.name)
            val = self._gen_expr(node.value)
            self._file_scope_decls.append(f'static TriadValue {name};')
            return f'{name} = {val};'
        return ''

    def _gen_local_let(self, node: IRNode) -> str:
        if isinstance(node, IRLet):
            name = self._sanitize(node.name)
            if node.value is not None:
                val = self._gen_expr(node.value)
                return f'TriadValue {name} = {val};'
            return f'TriadValue {name} = TRIAD_NONE_VAL;'
        elif isinstance(node, IRConst):
            name = self._sanitize(node.name)
            val = self._gen_expr(node.value)
            return f'TriadValue {name} = {val};'
        return '/* unknown let */;'

    def _gen_destruct_let(self, node: IRDestructLet) -> None:
        val_expr = self._gen_expr(node.value)
        tmp = self._fresh('_dt')
        self._emit(f'TriadValue {tmp} = {val_expr};')
        for i, name in enumerate(node.names):
            self._emit(f'TriadValue {self._sanitize(name)} = triad_list_get({tmp}.as.lval, {i});')

    def _gen_assign(self, node: IRAssign) -> str:
        if isinstance(node.target, IRIdent):
            name = self._sanitize(node.target.name)
            val = self._gen_expr(node.value)
            return f'{name} = {val};'
        elif isinstance(node.target, IRIndex):
            obj = self._gen_expr(node.target.obj)
            idx = self._gen_expr(node.target.index)
            val = self._gen_expr(node.value)
            return f'triad_list_set({obj}.as.lval, ({idx}).as.ival, {val});'
        elif isinstance(node.target, IRField):
            obj = self._gen_expr(node.target.obj)
            key = f'(TriadString[]){{.refcount=1, .len={len(node.target.field)}, .cap={len(node.target.field) + 1}, .data="{node.target.field}"}}'
            val = self._gen_expr(node.value)
            return f'triad_object_set({obj}.as.oval, triad_str_new("{node.target.field}"), {val});'
        target = self._gen_expr(node.target)
        val = self._gen_expr(node.value)
        return f'{target} = {val};'

    def _gen_if(self, node: IRIf):
        cond = self._gen_expr(node.condition)
        self._emit(f'if (triad_is_truthy({cond})) {{')
        self._indent += 1
        for stmt in node.then_body:
            self._emit_stmt(stmt)
        self._indent -= 1
        for cond_ir, body in node.elif_clauses:
            econd = self._gen_expr(cond_ir)
            self._emit(f'}} else if (triad_is_truthy({econd})) {{')
            self._indent += 1
            for stmt in body:
                self._emit_stmt(stmt)
            self._indent -= 1
        if node.else_body:
            self._emit('} else {')
            self._indent += 1
            for stmt in node.else_body:
                self._emit_stmt(stmt)
            self._indent -= 1
        self._emit('}')

    def _gen_for(self, node: IRFor):
        iter_expr = self._gen_expr(node.iter)
        var = self._sanitize(node.var)
        tmp = self._fresh('_iter')
        idx = self._fresh('_i')
        self._emit(f'{{')
        self._indent += 1
        self._emit(f'TriadValue {tmp} = {iter_expr};')
        self._emit(f'if ({tmp}.tag == TRIAD_LIST) {{')
        self._indent += 1
        self._emit(f'for (int32_t {idx} = 0; {idx} < triad_list_len({tmp}.as.lval); {idx}++) {{')
        self._indent += 1
        self._emit(f'TriadValue {var} = triad_list_get({tmp}.as.lval, {idx});')
        saved = self._loop_depth
        self._loop_depth += 1
        for stmt in node.body:
            self._emit_stmt(stmt)
        self._loop_depth = saved
        self._indent -= 1
        self._emit('}')
        self._indent -= 1
        self._emit(f'}} else if ({tmp}.tag == TRIAD_INT) {{')
        self._indent += 1
        self._emit(f'for (int64_t {idx} = 0; {idx} < {tmp}.as.ival; {idx}++) {{')
        self._indent += 1
        self._emit(f'TriadValue {var} = TRIAD_INT({idx});')
        self._loop_depth += 1
        for stmt in node.body:
            self._emit_stmt(stmt)
        self._loop_depth -= 1
        self._indent -= 1
        self._emit('}')
        self._indent -= 1
        self._emit('}')
        self._indent -= 1
        self._emit('}')

    def _gen_while(self, node: IRWhile):
        cond = self._gen_expr(node.condition)
        self._emit(f'while (triad_is_truthy({cond})) {{')
        self._indent += 1
        saved = self._loop_depth
        self._loop_depth += 1
        for stmt in node.body:
            self._emit_stmt(stmt)
        self._loop_depth = saved
        self._indent -= 1
        self._emit('}')

    def _gen_try_catch(self, node: IRTryCatch):
        self._emit('TRIAD_TRY_BEGIN {')
        self._indent += 1
        for stmt in node.body:
            self._emit_stmt(stmt)
        self._indent -= 1
        if node.catch_body:
            var = self._sanitize(node.catch_var) if node.catch_var else '_exc'
            self._emit(f'}} TRIAD_CATCH({var}) {{')
            self._indent += 1
            for stmt in node.catch_body:
                self._emit_stmt(stmt)
            self._indent -= 1
        if node.finally_body:
            self._emit('} TRIAD_FINALLY {')
            self._indent += 1
            for stmt in node.finally_body:
                self._emit_stmt(stmt)
            self._indent -= 1
        self._emit('} TRIAD_END;')

    def _gen_throw(self, node: IRThrow):
        if node.value:
            val = self._gen_expr(node.value)
            self._emit(f'triad_throw({val});')
        else:
            self._emit('triad_throw(TRIAD_NONE_VAL);')

    def _gen_expr(self, node: IRNode) -> str:
        if isinstance(node, IRInt):
            return f'TRIAD_INT({node.value}LL)'
        if isinstance(node, IRFloat):
            return f'TRIAD_FLOAT({node.value})'
        if isinstance(node, IRBool):
            return f"TRIAD_BOOL({('true' if node.value else 'false')})"
        if isinstance(node, IRString):
            escaped = node.value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\t', '\\t')
            return f'(TriadValue){{.tag = TRIAD_STRING, .as = {{.sval = triad_str_new("{escaped}")}}}}'
        if isinstance(node, IRNone):
            return 'TRIAD_NONE_VAL'
        if isinstance(node, IRIdent):
            return self._sanitize(node.name)
        if isinstance(node, IRBinOp):
            return self._gen_binop(node)
        if isinstance(node, IRUnaryOp):
            return self._gen_unaryop(node)
        if isinstance(node, IRCall):
            return self._gen_call(node)
        if isinstance(node, IRMethodCall):
            return self._gen_method_call(node)
        if isinstance(node, IRIndex):
            return self._gen_index(node)
        if isinstance(node, IRSlice):
            return self._gen_slice(node)
        if isinstance(node, IRField):
            return self._gen_field(node)
        if isinstance(node, IRList):
            return self._gen_list(node)
        if isinstance(node, IRMap):
            return self._gen_map(node)
        if isinstance(node, IRFString):
            return self._gen_fstring(node)
        if isinstance(node, IRListComp):
            return self._gen_listcomp(node)
        if isinstance(node, IRAssignExpr):
            return self._gen_assign_expr(node)
        return 'TRIAD_NONE_VAL'

    def _gen_binop(self, node: IRBinOp) -> str:
        left = self._gen_expr(node.left)
        right = self._gen_expr(node.right)
        op = node.op
        if op in ('+', '-', '*', '/', '%', '**'):
            return self._gen_arith_binop(left, right, op)
        if op in ('==', '!=', '<', '>', '<=', '>='):
            return self._gen_cmp_binop(left, right, op)
        if op in ('and', 'or'):
            l = self._gen_expr(node.left)
            r = self._gen_expr(node.right)
            if op == 'and':
                return f'(triad_is_truthy({l}) ? ({r}) : ({l}))'
            return f'(triad_is_truthy({l}) ? ({l}) : ({r}))'
        if op == 'in':
            return f'(TriadValue){{.tag = TRIAD_BOOL, .as = {{.bval = triad_list_contains({right}.as.lval, {left})}}}}'
        return f'TRIAD_NONE_VAL'

    def _gen_arith_binop(self, left: str, right: str, op: str) -> str:
        tmp = self._fresh('_bin')
        if op == '**':
            return f'(TriadValue){{.tag = TRIAD_FLOAT, .as = {{.fval = triad_math_pow(({left}).as.fval, ({right}).as.fval)}}}}'
        if op == '%':
            return f'({{ TriadValue {tmp}_l = ({left}), {tmp}_r = ({right}); ({tmp}_l.tag == TRIAD_INT && {tmp}_r.tag == TRIAD_INT) ? TRIAD_INT({tmp}_l.as.ival % {tmp}_r.as.ival) : TRIAD_FLOAT(fmod(({tmp}_l.tag==TRIAD_INT ? (double){tmp}_l.as.ival : {tmp}_l.as.fval), ({tmp}_r.tag==TRIAD_INT ? (double){tmp}_r.as.ival : {tmp}_r.as.fval))); }})'
        c_op = {'+': '+', '-': '-', '*': '*', '/': '/'}.get(op, '+')
        return f'({{ TriadValue {tmp}_l = ({left}), {tmp}_r = ({right}); ({tmp}_l.tag == TRIAD_INT && {tmp}_r.tag == TRIAD_INT) ? TRIAD_INT({tmp}_l.as.ival {c_op} {tmp}_r.as.ival) : TRIAD_FLOAT(({tmp}_l.tag==TRIAD_INT ? (double){tmp}_l.as.ival : {tmp}_l.as.fval) {c_op} ({tmp}_r.tag==TRIAD_INT ? (double){tmp}_r.as.ival : {tmp}_r.as.fval)); }})'

    def _gen_cmp_binop(self, left: str, right: str, op: str) -> str:
        tmp = self._fresh('_cmp')
        c_op = {'==': '==', '!=': '!=', '<': '<', '>': '>', '<=': '<=', '>=': '>='}.get(op, '==')
        return f'({{ TriadValue {tmp}_l = ({left}), {tmp}_r = ({right}); bool {tmp}_result; if ({tmp}_l.tag == TRIAD_INT && {tmp}_r.tag == TRIAD_INT) {tmp}_result = ({tmp}_l.as.ival {c_op} {tmp}_r.as.ival); else if ({tmp}_l.tag == TRIAD_STRING && {tmp}_r.tag == TRIAD_STRING) {tmp}_result = triad_str_eq({tmp}_l.as.sval, {tmp}_r.as.sval); else {tmp}_result = (({tmp}_l.tag==TRIAD_INT ? (double){tmp}_l.as.ival : {tmp}_l.as.fval) {c_op} ({tmp}_r.tag==TRIAD_INT ? (double){tmp}_r.as.ival : {tmp}_r.as.fval)); (TriadValue){{.tag = TRIAD_BOOL, .as = {{.bval = {tmp}_result}}}}; }})'

    def _gen_unaryop(self, node: IRUnaryOp) -> str:
        operand = self._gen_expr(node.operand)
        if node.op == '-':
            return f'(({{ TriadValue _u = ({operand}); (_u.tag == TRIAD_INT) ? TRIAD_INT(-_u.as.ival) : TRIAD_FLOAT(-_u.as.fval); }}))'
        if node.op == 'not':
            return f'TRIAD_BOOL(!triad_is_truthy({operand}))'
        return operand

    def _gen_call(self, node: IRCall) -> str:
        args = [self._gen_expr(a) for a in node.args]
        if isinstance(node.func, IRIdent):
            name = node.func.name
            if name == 'print':
                if not args:
                    return '({void)0; printf("\\n"); TRIAD_NONE_VAL;})'
                args_arr = ', '.join(args)
                return f'(triad_print({len(args)}, (TriadValue[]){{{args_arr}}}), TRIAD_NONE_VAL)'
            if name == 'len':
                return f'(TriadValue){{.tag = TRIAD_INT, .as = {{.ival = ({args[0]}).tag == TRIAD_LIST ? triad_list_len(({args[0]}).as.lval) : ({args[0]}).tag == TRIAD_STRING ? triad_str_len(({args[0]}).as.sval) : triad_dict_len(({args[0]}).as.dval)}}}}'
            if name == 'range':
                if len(args) == 1:
                    return f'(TriadValue){{.tag = TRIAD_LIST, .as = {{.lval = triad_range(0, ({args[0]}).as.ival, 1)}}}}'
                if len(args) == 2:
                    return f'(TriadValue){{.tag = TRIAD_LIST, .as = {{.lval = triad_range(({args[0]}).as.ival, ({args[1]}).as.ival, 1)}}}}'
                return f'(TriadValue){{.tag = TRIAD_LIST, .as = {{.lval = triad_range(({args[0]}).as.ival, ({args[1]}).as.ival, ({args[2]}).as.ival)}}}}'
            if name == 'abs':
                return f'(({{ TriadValue _a = ({args[0]}); (_a.tag == TRIAD_INT) ? TRIAD_INT(triad_math_abs_int(_a.as.ival)) : TRIAD_FLOAT(triad_math_abs(_a.as.fval)); }}))'
            if name == 'sqrt':
                return f'TRIAD_FLOAT(triad_math_sqrt(({args[0]}).tag == TRIAD_INT ? (double)({args[0]}).as.ival : ({args[0]}).as.fval))'
            if name == 'sin':
                return f'TRIAD_FLOAT(triad_math_sin(({args[0]}).tag == TRIAD_INT ? (double)({args[0]}).as.ival : ({args[0]}).as.fval))'
            if name == 'cos':
                return f'TRIAD_FLOAT(triad_math_cos(({args[0]}).tag == TRIAD_INT ? (double)({args[0]}).as.ival : ({args[0]}).as.fval))'
            if name == 'exp':
                return f'TRIAD_FLOAT(triad_math_exp(({args[0]}).tag == TRIAD_INT ? (double)({args[0]}).as.ival : ({args[0]}).as.fval))'
            if name == 'log':
                return f'TRIAD_FLOAT(triad_math_log(({args[0]}).tag == TRIAD_INT ? (double)({args[0]}).as.ival : ({args[0]}).as.fval))'
            if name == 'int':
                return f'(({{ TriadValue _c = ({args[0]}); (_c.tag == TRIAD_FLOAT) ? TRIAD_INT((int64_t)_c.as.fval) : (_c.tag == TRIAD_STRING) ? TRIAD_INT(atoll(_c.as.sval->data)) : _c; }}))'
            if name == 'float':
                return f'(({{ TriadValue _c = ({args[0]}); (_c.tag == TRIAD_INT) ? TRIAD_FLOAT((double)_c.as.ival) : (_c.tag == TRIAD_STRING) ? TRIAD_FLOAT(atof(_c.as.sval->data)) : _c; }}))'
            if name == 'str':
                return f'(TriadValue){{.tag = TRIAD_STRING, .as = {{.sval = triad_value_to_string({args[0]})}}}}'
            if name == 'type':
                return f'triad_type_name({args[0]})'
            if name == 'input':
                return f"triad_input({(args[0] if args else 'NULL')})"
            if name == 'push':
                return f'(triad_list_push(({args[0]}).as.lval, {args[1]}), TRIAD_NONE_VAL)'
            if name == 'pop':
                return f'(triad_list_remove_at(({args[0]}).as.lval, triad_list_len(({args[0]}).as.lval)-1), TRIAD_NONE_VAL)'
            if name == 'keys':
                return f'(TriadValue){{.tag = TRIAD_LIST, .as = {{.lval = triad_dict_keys(({args[0]}).as.dval)}}}}'
            if name == 'has':
                return f'TRIAD_BOOL(triad_dict_has(({args[0]}).as.dval, ({args[1]}).as.sval))'
            if name == 'solver_solve':
                return f'_triad_solver_solve({args[0]})'
            if name == 'ml_tensor':
                a1 = args[0] if len(args) > 0 else 'TRIAD_NONE_VAL'
                a2 = args[1] if len(args) > 1 else 'TRIAD_BOOL(0)'
                return f'_ml_tensor({a1}, {a2})'
            if name == 'ml_zeros':
                a1 = args[0]
                a2 = args[1] if len(args) > 1 else 'TRIAD_NONE_VAL'
                a3 = args[2] if len(args) > 2 else 'TRIAD_BOOL(0)'
                return f'_ml_tensor_zeros({a1}, {a2}, {a3})'
            if name == 'ml_randn':
                a1 = args[0]
                a2 = args[1] if len(args) > 1 else 'TRIAD_NONE_VAL'
                a3 = args[2] if len(args) > 2 else 'TRIAD_BOOL(0)'
                return f'_ml_tensor_randn({a1}, {a2}, {a3})'
            if name == 'ml_linear':
                return f'_ml_linear_new({args[0]}, {args[1]})'
            if name == 'ml_forward':
                return f'_ml_linear_forward({args[0]}, {args[1]})'
            if name == 'ml_relu':
                return f'_ml_relu({args[0]})'
            if name == 'ml_sigmoid':
                return f'_ml_sigmoid({args[0]})'
            if name == 'ml_tanh_act':
                return f'_ml_tanh({args[0]})'
            if name == 'ml_softmax':
                return f'_ml_softmax({args[0]})'
            if name == 'ml_mse_loss':
                return f'_ml_mse_loss({args[0]}, {args[1]})'
            if name == 'ml_cross_entropy':
                return f'_ml_cross_entropy({args[0]}, {args[1]})'
            if name == 'ml_backward':
                return f'_ml_backward({args[0]})'
            if name == 'ml_item':
                return f'_ml_tensor_item({args[0]})'
            if name == 'ml_data':
                return f'_ml_tensor_data({args[0]}, {args[1]})'
            if name == 'ml_seq_new':
                return f'_ml_sequential_new({args[0]})'
            if name == 'ml_seq_set':
                return f"_ml_sequential_set({args[0]}, {args[1]}, {args[2]}, {(args[3] if len(args) > 3 else 'TRIAD_NONE_VAL')})"
            if name == 'ml_seq_forward':
                return f'_ml_sequential_forward({args[0]}, {args[1]})'
            if name == 'ml_adam':
                return f'_ml_adam_new({args[0]}, {args[1]})'
            if name == 'ml_sgd':
                return f'_ml_sgd_new({args[0]}, {args[1]})'
            if name == 'ml_adam_step':
                return f'_ml_adam_step({args[0]})'
            if name == 'ml_adam_zero':
                return f'_ml_adam_zero_grad({args[0]})'
            if name == 'ml_sgd_step':
                return f'_ml_sgd_step({args[0]})'
            if name == 'ml_sgd_zero':
                return f'_ml_sgd_zero_grad({args[0]})'
            if name == 'ml_tensor_add':
                return f'_ml_tensor_add({args[0]}, {args[1]})'
            if name == 'ml_tensor_sub':
                return f'_ml_tensor_sub({args[0]}, {args[1]})'
            if name == 'ml_tensor_mul':
                return f'_ml_tensor_mul({args[0]}, {args[1]})'
            if name == 'ml_tensor_matmul':
                return f'_ml_tensor_matmul({args[0]}, {args[1]})'
            if name == 'ml_tensor_sum':
                return f'_ml_tensor_sum({args[0]})'
            if name == 'ml_tensor_mean':
                return f'_ml_tensor_mean({args[0]})'
            if name == 'ml_print':
                return f'_ml_tensor_print({args[0]})'
            if name in self._classes:
                cls = self._classes[name]
                tmp = self._fresh('_obj')
                obj_init = f'(TriadValue){{.tag = TRIAD_OBJECT, .as = {{.oval = triad_object_new("{name}")}}}}'
                parts_c = [f'({{ TriadValue {tmp} = {obj_init}; ']
                new_fn = f'_triad_cls_{name}_new'
                args_with_self = ', '.join([tmp] + args)
                parts_c.append(f'{new_fn}({1 + len(args)}, (TriadValue[]){{{args_with_self}}}); ')
                parts_c.append(f'{tmp}; }})')
                return ''.join(parts_c)
            if name in self._top_level_fns:
                fname = self._c_fn_name(name)
                args_arr = ', '.join(args)
                return f'{fname}({len(args)}, (TriadValue[]){{{args_arr}}})'
            fname = self._c_fn_name(name)
            args_arr = ', '.join(args)
            return f'triad_closure_call({self._sanitize(name)}.as.cval, {len(args)}, (TriadValue[]){{{args_arr}}})'
        func = self._gen_expr(node.func)
        args_arr = ', '.join(args)
        return f'triad_closure_call(({func}).as.cval, {len(args)}, (TriadValue[]){{{args_arr}}})'

    def _gen_method_call(self, node: IRMethodCall) -> str:
        obj = self._gen_expr(node.obj)
        args = [self._gen_expr(a) for a in node.args]
        m = node.method
        if m == 'push':
            return f'(triad_list_push(({obj}).as.lval, {args[0]}), TRIAD_NONE_VAL)'
        if m == 'pop':
            return f'triad_list_get(({obj}).as.lval, triad_list_len(({obj}).as.lval)-1)'
        if m == 'len':
            return f'TRIAD_INT(triad_list_len(({obj}).as.lval))'
        if m == 'upper':
            return f'(TriadValue){{.tag = TRIAD_STRING, .as = {{.sval = triad_str_upper(({obj}).as.sval)}}}}'
        if m == 'lower':
            return f'(TriadValue){{.tag = TRIAD_STRING, .as = {{.sval = triad_str_lower(({obj}).as.sval)}}}}'
        if m == 'strip':
            return f'(TriadValue){{.tag = TRIAD_STRING, .as = {{.sval = triad_str_strip(({obj}).as.sval)}}}}'
        if m == 'split':
            if args:
                return f'(TriadValue){{.tag = TRIAD_LIST, .as = {{.lval = triad_str_split(({obj}).as.sval, ({args[0]}).as.sval->data)}}}}'
            return f'(TriadValue){{.tag = TRIAD_LIST, .as = {{.lval = triad_str_split(({obj}).as.sval, " ")}}}}'
        if m == 'replace':
            return f'(TriadValue){{.tag = TRIAD_STRING, .as = {{.sval = triad_str_replace(({obj}).as.sval, ({args[0]}).as.sval->data, ({args[1]}).as.sval->data)}}}}'
        if m == 'contains':
            return f'TRIAD_BOOL(triad_str_contains(({obj}).as.sval, ({args[0]}).as.sval->data))'
        if m == 'starts_with':
            return f'TRIAD_BOOL(triad_str_starts_with(({obj}).as.sval, ({args[0]}).as.sval->data))'
        if m == 'ends_with':
            return f'TRIAD_BOOL(triad_str_ends_with(({obj}).as.sval, ({args[0]}).as.sval->data))'
        if m == 'join':
            return f'(TriadValue){{.tag = TRIAD_STRING, .as = {{.sval = _triad_str_join(({obj}).as.sval, {args[0]}.as.lval)}}}}'
        if m in ('append', 'push'):
            return f'(triad_list_push(({obj}).as.lval, {args[0]}), TRIAD_NONE_VAL)'
        if m == 'sort':
            return f'(triad_list_sort(({obj}).as.lval), {obj})'
        if m == 'reversed':
            return f'(TriadValue){{.tag = TRIAD_LIST, .as = {{.lval = triad_list_reversed(({obj}).as.lval)}}}}'
        if m == 'keys':
            return f'(TriadValue){{.tag = TRIAD_LIST, .as = {{.lval = triad_dict_keys(({obj}).as.dval)}}}}'
        if m == 'values':
            return f'(TriadValue){{.tag = TRIAD_LIST, .as = {{.lval = triad_dict_values(({obj}).as.dval)}}}}'
        if m == 'items':
            return f'(TriadValue){{.tag = TRIAD_LIST, .as = {{.lval = triad_dict_items(({obj}).as.dval)}}}}'
        if m == 'has':
            return f'TRIAD_BOOL(triad_dict_has(({obj}).as.dval, ({args[0]}).as.sval))'
        if m == 'sum':
            return f'TRIAD_FLOAT(triad_ndarray_sum(({obj}).as.aval))'
        if m == 'mean':
            return f'TRIAD_FLOAT(triad_ndarray_mean(({obj}).as.aval))'
        if m == 'reshape':
            return f'(TriadValue){{.tag = TRIAD_NDARRAY, .as = {{.aval = triad_ndarray_reshape(({obj}).as.aval, {args[0]})}}}}'
        for cls_name, cls_node in self._classes.items():
            for method in cls_node.methods:
                if method.name == m:
                    cls_fn = f'_triad_cls_{cls_name}_{m}'
                    if args:
                        args_arr = ', '.join(args)
                        return f'{cls_fn}({1 + len(args)}, (TriadValue[]){{{obj}, {args_arr}}})'
                    return f'{cls_fn}(1, (TriadValue[]){{{obj}}})'
        fname = self._c_fn_name(m)
        args_arr = ', '.join(args)
        return f'{fname}({len(args) + 1}, (TriadValue[]){{{obj}, {args_arr}}})'

    def _gen_index(self, node: IRIndex) -> str:
        obj = self._gen_expr(node.obj)
        idx = self._gen_expr(node.index)
        return f'({{ TriadValue _obj = ({obj}), _idx = ({idx}); (_obj.tag == TRIAD_LIST) ? triad_list_get(_obj.as.lval, _idx.as.ival) : (_obj.tag == TRIAD_DICT) ? triad_dict_get(_obj.as.dval, _idx.as.sval) : TRIAD_NONE_VAL; }})'

    def _gen_slice(self, node: IRSlice) -> str:
        obj = self._gen_expr(node.obj)
        s = self._gen_expr(node.start) if node.start else 'TRIAD_INT(0)'
        e = self._gen_expr(node.end) if node.end else 'TRIAD_INT(0)'
        step = self._gen_expr(node.step) if node.step else 'TRIAD_INT(1)'
        return f'({{ TriadValue _so = ({obj}); (_so.tag == TRIAD_LIST) ? (TriadValue){{.tag = TRIAD_LIST, .as = {{.lval = triad_list_slice(_so.as.lval, ({s}).as.ival, ({e}).as.ival, ({step}).as.ival)}}}} : (TriadValue){{.tag = TRIAD_STRING, .as = {{.sval = triad_str_slice(_so.as.sval, ({s}).as.ival, ({e}).as.ival, ({step}).as.ival)}}}}; }})'

    def _gen_field(self, node: IRField) -> str:
        obj = self._gen_expr(node.obj)
        if node.field == 'length' or node.field == 'len':
            return f'({{ TriadValue _f = ({obj}); (_f.tag == TRIAD_LIST) ? TRIAD_INT(triad_list_len(_f.as.lval)) : (_f.tag == TRIAD_STRING) ? TRIAD_INT(triad_str_len(_f.as.sval)) : TRIAD_INT(triad_dict_len(_f.as.dval)); }})'
        return f'triad_object_get(({obj}).as.oval, triad_str_new("{node.field}"))'

    def _gen_list(self, node: IRList) -> str:
        elems = [self._gen_expr(e) for e in node.elements]
        if not elems:
            return '(TriadValue){.tag = TRIAD_LIST, .as = {.lval = triad_list_new()}}'
        elems_str = ', '.join(elems)
        return f'({{ TriadList *_tl = triad_list_new_cap({len(elems)}); TriadValue _ta[] = {{{elems_str}}}; for (int32_t _ti = 0; _ti < {len(elems)}; _ti++) triad_list_push(_tl, _ta[_ti]); (TriadValue){{.tag = TRIAD_LIST, .as = {{.lval = _tl}}}}; }})'

    def _gen_map(self, node: IRMap) -> str:
        if not node.pairs:
            return '(TriadValue){.tag = TRIAD_DICT, .as = {.dval = triad_dict_new()}}'
        parts = []
        for k, v in node.pairs:
            key_expr = self._gen_expr(k)
            val_expr = self._gen_expr(v)
            parts.append(f'triad_dict_set(_td, ({key_expr}).as.sval, {val_expr});')
        body = ' '.join(parts)
        return f'({{ TriadDict *_td = triad_dict_new(); {body} (TriadValue){{.tag = TRIAD_DICT, .as = {{.dval = _td}}}}; }})'

    def _gen_fstring(self, node: IRFString) -> str:
        parts = []
        for text, expr in node.parts:
            if expr is not None:
                raw = self._gen_expr(expr)
                parts.append(f'(TriadValue){{.tag = TRIAD_STRING, .as = {{.sval = triad_value_to_string({raw})}}}}')
            elif text:
                escaped = text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                parts.append(f'(TriadValue){{.tag = TRIAD_STRING, .as = {{.sval = triad_str_new("{escaped}")}}}}')
        if not parts:
            return '(TriadValue){.tag = TRIAD_STRING, .as = {.sval = triad_str_new("")}}'
        if len(parts) == 1:
            return parts[0]
        result = parts[0]
        for p in parts[1:]:
            result = f'(TriadValue){{.tag = TRIAD_STRING, .as = {{.sval = triad_str_concat(({result}).as.sval, ({p}).as.sval)}}}}'
        return result

    def _gen_listcomp(self, node: IRListComp) -> str:
        tmp_list = self._fresh('_lc')
        iter_expr = self._gen_expr(node.iter)
        var = self._sanitize(node.var)
        idx = self._fresh('_lci')
        result_expr = self._gen_expr(node.expr)
        cond_expr = self._gen_expr(node.condition) if node.condition else None
        code = f'({{ TriadList *{tmp_list} = triad_list_new(); TriadValue _lc_iter = {iter_expr}; if (_lc_iter.tag == TRIAD_LIST) {{ for (int32_t {idx} = 0; {idx} < triad_list_len(_lc_iter.as.lval); {idx}++) {{ TriadValue {var} = triad_list_get(_lc_iter.as.lval, {idx}); '
        if cond_expr:
            code += f'if (!triad_is_truthy({cond_expr})) continue; '
        tail = '(TriadValue){.tag = TRIAD_LIST, .as = {.lval = ' + tmp_list + '}};})'
        code += 'triad_list_push(' + tmp_list + ', ' + result_expr + '); }} }} ' + tail
        return code

    def _gen_assign_expr(self, node: IRAssignExpr) -> str:
        val = self._gen_expr(node.value)
        if isinstance(node.target, IRIdent):
            name = self._sanitize(node.target.name)
            return f'({name} = {val})'
        return val

    @staticmethod
    def _sanitize(name: str) -> str:
        if name in ('int', 'float', 'char', 'long', 'short', 'double', 'void', 'static', 'const', 'unsigned', 'signed', 'for', 'while', 'if', 'else', 'switch', 'case', 'break', 'continue', 'return', 'struct', 'typedef', 'enum', 'union', 'sizeof', 'NULL', 'true', 'false', 'auto', 'register', 'extern', 'volatile', 'inline', 'default', 'do', 'goto'):
            return f'_triad_{name}'
        return name

def compile_to_c(source: str, filename: str='<triad>') -> str:
    from frontend.lexer_universal import tokenize
    from frontend.parser_universal import Parser
    from compiler.lower import lower_module
    tokens = tokenize(source, filename)
    parser = Parser(tokens, filename)
    mod = parser.parse_module()
    ir = lower_module(mod)
    gen = CCodeGen()
    return gen.generate(ir)