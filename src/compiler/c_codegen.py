from __future__ import annotations

from compiler.ir import *


class CCompileError(Exception):
    pass

_C_BASE = r'''#include "triad_rt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>

#define TRIAD_ENSURE(ptr, msg) do { if ((ptr) == NULL) { fprintf(stderr, "triad: allocation failed: %s\n", (msg)); abort(); } } while(0)
'''

_C_SOLVER = r'''
static double _solver_dget(TriadValue cfg, const char *key, double def) {
    TriadValue v = triad_dict_get(cfg.as.dval, triad_str_new(key));
    if (v.tag == TRIAD_NONE) return def;
    return (v.tag == TRIAD_INT) ? (double)v.as.ival : v.as.fval;
}

static int32_t _solver_iget(TriadValue cfg, const char *key, int32_t def) {
    TriadValue v = triad_dict_get(cfg.as.dval, triad_str_new(key));
    if (v.tag == TRIAD_NONE) return def;
    return (v.tag == TRIAD_INT) ? (int32_t)v.as.ival : (int32_t)v.as.fval;
}

static int64_t _solver_lget(TriadValue cfg, const char *key, int64_t def) {
    TriadValue v = triad_dict_get(cfg.as.dval, triad_str_new(key));
    if (v.tag == TRIAD_NONE) return def;
    return (v.tag == TRIAD_INT) ? v.as.ival : (int64_t)v.as.fval;
}

static double _solver_vget(TriadValue v, double def) {
    if (v.tag == TRIAD_INT) return (double)v.as.ival;
    if (v.tag == TRIAD_FLOAT) return v.as.fval;
    return def;
}

static TriadValue _triad_solver_solve(TriadValue cfg) {
    static double _nu_def[] = {2.0, 0.5, 0.1};
    static double _lam_def[] = {-0.3, -0.2, -0.1};
    static double _nu_cfg[16];
    static double _lam_cfg[16];

    TriadSolverC p;
    memset(&p, 0, sizeof(p));

    p.N      = _solver_iget(cfg, "N", 128);
    p.L      = _solver_dget(cfg, "L", 32.0);
    p.dt     = _solver_dget(cfg, "dt", 0.005);
    p.T      = _solver_dget(cfg, "T", 2.0);
    p.hbar   = _solver_dget(cfg, "hbar", 1.0);
    p.m      = _solver_dget(cfg, "m", 1.0);
    p.omega  = _solver_dget(cfg, "omega", 0.05);
    p.Lambda = _solver_dget(cfg, "Lambda", -0.5);
    p.alpha  = _solver_dget(cfg, "alpha", 0.15);
    p.sigma  = _solver_dget(cfg, "sigma", 1.5);
    p.Gamma  = _solver_dget(cfg, "Gamma", 0.05);
    p.f_FDT  = _solver_dget(cfg, "f_FDT", 0.002);
    p.M      = 3;
    p.nu     = _nu_def;
    p.lam    = _lam_def;
    {
        TriadValue nu_v = triad_dict_get(cfg.as.dval, triad_str_new("nu"));
        TriadValue lam_v = triad_dict_get(cfg.as.dval, triad_str_new("lam"));
        if (nu_v.tag == TRIAD_LIST && lam_v.tag == TRIAD_LIST) {
            int32_t nu_n = triad_list_len(nu_v.as.lval);
            int32_t lam_n = triad_list_len(lam_v.as.lval);
            if (nu_n == lam_n && nu_n >= 1 && nu_n <= 16) {
                for (int32_t _i = 0; _i < nu_n; _i++) {
                    _nu_cfg[_i] = _solver_vget(triad_list_get(nu_v.as.lval, _i), _nu_def[_i < 3 ? _i : 2]);
                    _lam_cfg[_i] = _solver_vget(triad_list_get(lam_v.as.lval, _i), _lam_def[_i < 3 ? _i : 2]);
                }
                p.M = nu_n;
                p.nu = _nu_cfg;
                p.lam = _lam_cfg;
            }
        }
    }
    p.mode   = 2;
    p.seed   = (uint64_t)_solver_lget(cfg, "seed", 42);
    p.D      = _solver_iget(cfg, "D", 3);
    p.fdt_couple = _solver_iget(cfg, "fdt_couple", 1);
    p.kT     = _solver_dget(cfg, "kT", 1.0);
    p.init_sigma = _solver_dget(cfg, "init_sigma", 0.0);

    {
        TriadValue iv = triad_dict_get(cfg.as.dval, triad_str_new("init"));
        if (iv.tag == TRIAD_STRING && strcmp(iv.as.sval->data, "chaos") == 0) {
            p.init_mode = 1;
        }
    }

    {
        TriadValue vv = triad_dict_get(cfg.as.dval, triad_str_new("V_ext"));
        if (vv.tag == TRIAD_STRING) {
            p.V_ext = vv.as.sval->data;
        } else {
            p.V_ext = "harmonic";
        }
    }

    int64_t _total = 1;
    for (int32_t _d = 0; _d < p.D; _d++) {
        _total *= p.N;
    }

    TriadCplx *_psi = NULL;
    double *_dens = NULL;
    double *_xax = NULL;
    double _dx = 0.0;

    TriadSolverResult r;
    TriadSolverResult2D r2;
    TriadSolverResult3D r3;
    TriadSolverResultND rN;

    memset(&r, 0, sizeof(r));
    memset(&r2, 0, sizeof(r2));
    memset(&r3, 0, sizeof(r3));
    memset(&rN, 0, sizeof(rN));

    if (p.D >= 4) {
        rN = triad_solve_nd(&p);
        _psi = rN.psi_final;
        _dens = rN.density_final;
        _xax = rN.x;
        _dx = rN.dx;
    } else if (p.D == 3) {
        r3 = triad_solve_3d(&p);
        _psi = r3.psi_final;
        _dens = r3.density_final;
        _xax = r3.x;
        _dx = r3.dx;
    } else if (p.D == 2) {
        r2 = triad_solve_2d(&p);
        _psi = r2.psi_final;
        _dens = r2.density_final;
        _xax = r2.x;
        _dx = r2.dx;
    } else {
        r = triad_solve_1d(&p);
        _psi = r.psi_final;
        _dens = r.density_final;
        _xax = r.x;
        _dx = r.dx;
    }

    double _dV = 1.0;
    for (int32_t _d = 0; _d < p.D; _d++) {
        _dV *= _dx;
    }

    double norm = 0.0;
    double peak = 0.0;

    for (int64_t i = 0; i < _total; i++) {
        norm += _dens[i] * _dV;
        if (_dens[i] > peak) {
            peak = _dens[i];
        }
    }

    TriadDict *rd = triad_dict_new();
    triad_dict_set(rd, triad_str_new("norm"), TRIAD_FLOAT(norm));
    triad_dict_set(rd, triad_str_new("peak"), TRIAD_FLOAT(peak));
    triad_dict_set(rd, triad_str_new("N"), TRIAD_INT(p.N));
    triad_dict_set(rd, triad_str_new("dx"), TRIAD_FLOAT(_dx));
    triad_dict_set(rd, triad_str_new("D"), TRIAD_INT(p.D));

    {
        TriadCplx *psi_hat = (TriadCplx*)malloc((size_t)p.N * sizeof(TriadCplx));
        if (psi_hat != NULL) {
            triad_fft_fn(p.N, _psi, psi_hat);

            double total_P = 0.0;
            double structured_P = 0.0;
            double k_cutoff = 1.0;

            for (int32_t i = 0; i < p.N; i++) {
                double re = psi_hat[i].re;
                double im = psi_hat[i].im;
                double Pi = re * re + im * im;

                total_P += Pi;

                double fi = (double)i;
                if (i > p.N / 2) {
                    fi -= (double)p.N;
                }

                double ki = 2.0 * 3.14159265358979323846 * fi / ((double)p.N * _dx);
                if (ki < 0.0) {
                    ki = -ki;
                }

                if (ki > k_cutoff) {
                    structured_P += Pi;
                }
            }

            double cryst = (total_P > 0.0) ? (structured_P / total_P) : 0.0;
            triad_dict_set(rd, triad_str_new("crystallinity"), TRIAD_FLOAT(cryst));
            free(psi_hat);
        } else {
            triad_dict_set(rd, triad_str_new("crystallinity"), TRIAD_FLOAT(0.0));
        }
    }

    {
        TriadList *dlist = triad_list_new_cap((int32_t)_total);
        for (int64_t i = 0; i < _total; i++) {
            triad_list_push(dlist, TRIAD_FLOAT(_dens[i]));
        }
        triad_dict_set(
            rd,
            triad_str_new("density"),
            (TriadValue){.tag = TRIAD_LIST, .as = {.lval = dlist}}
        );
    }

    {
        TriadList *xlist = triad_list_new_cap(p.N);
        for (int32_t i = 0; i < p.N; i++) {
            triad_list_push(xlist, TRIAD_FLOAT(_xax[i]));
        }
        triad_dict_set(
            rd,
            triad_str_new("x"),
            (TriadValue){.tag = TRIAD_LIST, .as = {.lval = xlist}}
        );
    }

    if (p.D >= 4) {
        triad_solver_result_nd_free(&rN);
    } else if (p.D == 3) {
        triad_solver_result_3d_free(&r3);
    } else if (p.D == 2) {
        triad_solver_result_2d_free(&r2);
    } else {
        triad_solver_result_free(&r);
    }

    return (TriadValue){.tag = TRIAD_DICT, .as = {.dval = rd}};
}

'''

_C_ML = r'''
#include "triad_ml.h"

static double _ml_vf(TriadValue v) {
    return (v.tag == TRIAD_INT) ? (double)v.as.ival : v.as.fval;
}

static TriadValue _ml_tensor(TriadValue data_v, TriadValue rg_v) {
    int rg = (rg_v.tag == TRIAD_BOOL) ? rg_v.as.bval : 0;

    if (data_v.tag == TRIAD_LIST) {
        TriadList *l = data_v.as.lval;

        if (l->len > 0 && l->items[0].tag == TRIAD_LIST) {
            int32_t rows = l->len;
            int32_t cols = triad_list_len(l->items[0].as.lval);
            int32_t shape[] = {rows, cols};
            TriadTensor *t = triad_tensor_new(2, shape, rg);

            for (int32_t i = 0; i < rows; i++) {
                TriadList *row = l->items[i].as.lval;
                for (int32_t j = 0; j < cols; j++) {
                    t->data[i * cols + j] = _ml_vf(row->items[j]);
                }
            }

            return TRIAD_PTR_VAL(t);
        }

        int32_t shape[] = {l->len};
        TriadTensor *t = triad_tensor_new(1, shape, rg);

        for (int32_t i = 0; i < l->len; i++) {
            t->data[i] = _ml_vf(l->items[i]);
        }

        return TRIAD_PTR_VAL(t);
    }

    TriadTensor *t = triad_tensor_scalar(_ml_vf(data_v), rg);
    return TRIAD_PTR_VAL(t);
}

static TriadValue _ml_tensor_zeros(TriadValue rows_v, TriadValue cols_v, TriadValue rg_v) {
    int rg = (rg_v.tag == TRIAD_BOOL) ? rg_v.as.bval : 0;

    if (cols_v.tag == TRIAD_NONE) {
        int32_t s[] = {(int32_t)rows_v.as.ival};
        return TRIAD_PTR_VAL(triad_tensor_zeros(1, s, rg));
    }

    int32_t s[] = {(int32_t)rows_v.as.ival, (int32_t)cols_v.as.ival};
    return TRIAD_PTR_VAL(triad_tensor_zeros(2, s, rg));
}

static TriadValue _ml_tensor_randn(TriadValue rows_v, TriadValue cols_v, TriadValue rg_v) {
    int rg = (rg_v.tag == TRIAD_BOOL) ? rg_v.as.bval : 0;

    if (cols_v.tag == TRIAD_NONE) {
        int32_t s[] = {(int32_t)rows_v.as.ival};
        return TRIAD_PTR_VAL(triad_tensor_randn(1, s, rg));
    }

    int32_t s[] = {(int32_t)rows_v.as.ival, (int32_t)cols_v.as.ival};
    return TRIAD_PTR_VAL(triad_tensor_randn(2, s, rg));
}

static TriadValue _ml_triad_new(TriadValue in_v, TriadValue out_v) {
    return TRIAD_PTR_VAL(triad_triad_new((int32_t)in_v.as.ival, (int32_t)out_v.as.ival, 1));
}

static TriadValue _ml_triad_forward(TriadValue layer_v, TriadValue x_v) {
    return TRIAD_PTR_VAL(triad_triad_forward((Triadtriad*)layer_v.as.ptr, (TriadTensor*)x_v.as.ptr));
}

static TriadValue _ml_relu(TriadValue x_v) {
    return TRIAD_PTR_VAL(triad_tensor_relu((TriadTensor*)x_v.as.ptr));
}

static TriadValue _ml_sigmoid(TriadValue x_v) {
    return TRIAD_PTR_VAL(triad_tensor_sigmoid((TriadTensor*)x_v.as.ptr));
}

static TriadValue _ml_tanh(TriadValue x_v) {
    return TRIAD_PTR_VAL(triad_tensor_tanh((TriadTensor*)x_v.as.ptr));
}

static TriadValue _ml_softmax(TriadValue x_v) {
    return TRIAD_PTR_VAL(triad_tensor_softmax((TriadTensor*)x_v.as.ptr));
}

static TriadValue _ml_mse_loss(TriadValue pred_v, TriadValue tgt_v) {
    return TRIAD_PTR_VAL(triad_tensor_mse_loss((TriadTensor*)pred_v.as.ptr, (TriadTensor*)tgt_v.as.ptr));
}

static TriadValue _ml_cross_entropy(TriadValue logits_v, TriadValue tgt_v) {
    return TRIAD_PTR_VAL(triad_tensor_cross_entropy((TriadTensor*)logits_v.as.ptr, (TriadTensor*)tgt_v.as.ptr));
}

static TriadValue _ml_backward(TriadValue t_v) {
    triad_tensor_backward((TriadTensor*)t_v.as.ptr, NULL);
    return TRIAD_NONE_VAL;
}

static TriadValue _ml_tensor_data(TriadValue t_v, TriadValue idx_v) {
    TriadTensor *t = (TriadTensor*)t_v.as.ptr;
    return TRIAD_FLOAT(t->data[(int32_t)idx_v.as.ival]);
}

static TriadValue _ml_tensor_item(TriadValue t_v) {
    TriadTensor *t = (TriadTensor*)t_v.as.ptr;
    return TRIAD_FLOAT(t->data[0]);
}

static TriadValue _ml_sequential_new(TriadValue n_v) {
    return TRIAD_PTR_VAL(triad_sequential_new((int32_t)n_v.as.ival));
}

static TriadValue _ml_sequential_set(
    TriadValue seq_v,
    TriadValue idx_v,
    TriadValue type_v,
    TriadValue layer_v
) {
    triad_sequential_set(
        (TriadSequential*)seq_v.as.ptr,
        (int32_t)idx_v.as.ival,
        (TriadLayerType)(int32_t)type_v.as.ival,
        (type_v.as.ival == TRIAD_LAYER_triad) ? layer_v.as.ptr : NULL
    );

    return TRIAD_NONE_VAL;
}

static TriadValue _ml_sequential_forward(TriadValue seq_v, TriadValue x_v) {
    return TRIAD_PTR_VAL(triad_sequential_forward((TriadSequential*)seq_v.as.ptr, (TriadTensor*)x_v.as.ptr));
}

static TriadValue _ml_adam_new(TriadValue seq_v, TriadValue lr_v) {
    TriadTensor *params[256];
    int32_t np = triad_sequential_params((TriadSequential*)seq_v.as.ptr, params, 256);
    return TRIAD_PTR_VAL(triad_adam_new(params, np, _ml_vf(lr_v), 0.9, 0.999, 1e-8));
}

static TriadValue _ml_sgd_new(TriadValue seq_v, TriadValue lr_v) {
    TriadTensor *params[256];
    int32_t np = triad_sequential_params((TriadSequential*)seq_v.as.ptr, params, 256);
    return TRIAD_PTR_VAL(triad_sgd_new(params, np, _ml_vf(lr_v), 0.0));
}

static TriadValue _ml_adam_step(TriadValue opt_v) {
    triad_adam_step((TriadAdam*)opt_v.as.ptr);
    return TRIAD_NONE_VAL;
}

static TriadValue _ml_adam_zero_grad(TriadValue opt_v) {
    triad_adam_zero_grad((TriadAdam*)opt_v.as.ptr);
    return TRIAD_NONE_VAL;
}

static TriadValue _ml_sgd_step(TriadValue opt_v) {
    triad_sgd_step((TriadSGD*)opt_v.as.ptr);
    return TRIAD_NONE_VAL;
}

static TriadValue _ml_sgd_zero_grad(TriadValue opt_v) {
    triad_sgd_zero_grad((TriadSGD*)opt_v.as.ptr);
    return TRIAD_NONE_VAL;
}

static TriadValue _ml_tensor_add(TriadValue a_v, TriadValue b_v) {
    return TRIAD_PTR_VAL(triad_tensor_add((TriadTensor*)a_v.as.ptr, (TriadTensor*)b_v.as.ptr));
}

static TriadValue _ml_tensor_sub(TriadValue a_v, TriadValue b_v) {
    return TRIAD_PTR_VAL(triad_tensor_sub((TriadTensor*)a_v.as.ptr, (TriadTensor*)b_v.as.ptr));
}

static TriadValue _ml_tensor_mul(TriadValue a_v, TriadValue b_v) {
    return TRIAD_PTR_VAL(triad_tensor_mul((TriadTensor*)a_v.as.ptr, (TriadTensor*)b_v.as.ptr));
}

static TriadValue _ml_tensor_matmul(TriadValue a_v, TriadValue b_v) {
    return TRIAD_PTR_VAL(triad_tensor_matmul((TriadTensor*)a_v.as.ptr, (TriadTensor*)b_v.as.ptr));
}

static TriadValue _ml_tensor_sum(TriadValue a_v) {
    return TRIAD_PTR_VAL(triad_tensor_sum((TriadTensor*)a_v.as.ptr));
}

static TriadValue _ml_tensor_mean(TriadValue a_v) {
    return TRIAD_PTR_VAL(triad_tensor_mean((TriadTensor*)a_v.as.ptr));
}

static TriadValue _ml_embedding_new(TriadValue vocab_v, TriadValue dim_v) {
    return TRIAD_PTR_VAL(triad_embedding_new((int32_t)vocab_v.as.ival, (int32_t)dim_v.as.ival));
}

static TriadValue _ml_embedding_forward(TriadValue e_v, TriadValue idx_v) {
    return TRIAD_PTR_VAL(triad_embedding_forward((TriadEmbedding*)e_v.as.ptr, (TriadTensor*)idx_v.as.ptr));
}

static TriadValue _ml_layernorm_new(TriadValue dim_v) {
    return TRIAD_PTR_VAL(triad_layer_norm_new((int32_t)dim_v.as.ival, 1e-5));
}

static TriadValue _ml_layernorm_forward(TriadValue ln_v, TriadValue x_v) {
    return TRIAD_PTR_VAL(triad_layer_norm_forward((TriadLayerNorm*)ln_v.as.ptr, (TriadTensor*)x_v.as.ptr));
}

static TriadValue _ml_mha_new(TriadValue dm_v, TriadValue nh_v) {
    return TRIAD_PTR_VAL(triad_mha_new((int32_t)dm_v.as.ival, (int32_t)nh_v.as.ival));
}

static TriadValue _ml_mha_forward(TriadValue m_v, TriadValue x_v) {
    return TRIAD_PTR_VAL(triad_mha_forward((TriadMultiHeadAttention*)m_v.as.ptr, (TriadTensor*)x_v.as.ptr));
}

static TriadValue _ml_transformer_new(
    TriadValue vocab_v,
    TriadValue dm_v,
    TriadValue nb_v,
    TriadValue nh_v,
    TriadValue dff_v
) {
    return TRIAD_PTR_VAL(triad_transformer_new(
        (int32_t)vocab_v.as.ival,
        (int32_t)dm_v.as.ival,
        (int32_t)nb_v.as.ival,
        (int32_t)nh_v.as.ival,
        (int32_t)dff_v.as.ival
    ));
}

static TriadValue _ml_transformer_forward(TriadValue t_v, TriadValue idx_v) {
    return TRIAD_PTR_VAL(triad_transformer_forward((TriadTransformer*)t_v.as.ptr, (TriadTensor*)idx_v.as.ptr));
}

static TriadValue _ml_transformer_adam(TriadValue t_v, TriadValue lr_v) {
    TriadTensor *params[1024];
    int32_t np = triad_transformer_params((TriadTransformer*)t_v.as.ptr, params, 1024);
    return TRIAD_PTR_VAL(triad_adam_new(params, np, _ml_vf(lr_v), 0.9, 0.999, 1e-8));
}

static TriadValue _ml_wave_new(TriadValue in_v, TriadValue out_v) {
    return TRIAD_PTR_VAL(triad_wave_triad_new((int32_t)in_v.as.ival, (int32_t)out_v.as.ival));
}

static TriadValue _ml_wave_forward(TriadValue w_v, TriadValue x_v) {
    return TRIAD_PTR_VAL(triad_wave_triad_forward((TriadWavetriad*)w_v.as.ptr, (TriadTensor*)x_v.as.ptr));
}

static TriadValue _ml_wave_adam(TriadValue w_v, TriadValue lin_v, TriadValue lr_v) {
    TriadTensor *params[64];
    int32_t np = triad_wave_triad_params((TriadWavetriad*)w_v.as.ptr, params, 64);

    if (lin_v.tag != TRIAD_NONE) {
        Triadtriad *l = (Triadtriad*)lin_v.as.ptr;
        params[np++] = l->weight;
        if (l->bias) {
            params[np++] = l->bias;
        }
    }

    return TRIAD_PTR_VAL(triad_adam_new(params, np, _ml_vf(lr_v), 0.9, 0.999, 1e-8));
}

static TriadValue _ml_tensor_print(TriadValue t_v) {
    TriadTensor *t = (TriadTensor*)t_v.as.ptr;

    if (t->ndim == 0 || t->size == 1) {
        printf("%.6f\n", t->data[0]);
    } else {
        printf("[");
        for (int64_t i = 0; i < t->size && i < 20; i++) {
            if (i > 0) {
                printf(", ");
            }
            printf("%.4f", t->data[i]);
        }
        if (t->size > 20) {
            printf(", ...");
        }
        printf("]\n");
    }

    return TRIAD_NONE_VAL;
}
'''

_C_CCALL = r'''
#ifdef _WIN32
#include <windows.h>

static void *_ccall_dlopen(const char *lib_name) {
    void *h = (void*)LoadLibraryA(lib_name);

    if (!h) {
        fprintf(stderr, "ccall: cannot open library %s\n", lib_name);
        return NULL;
    }

    return h;
}

static void *_ccall_sym(void *handle, const char *name) {
    return (void*)GetProcAddress((HMODULE)handle, name);
}

static const char *_ccall_sym_error(void *sym) {
    return sym ? NULL : "unknown symbol";
}
#else
#include <dlfcn.h>

static void *_ccall_dlopen(const char *lib_name) {
    void *h = dlopen(lib_name, RTLD_LAZY);

    if (!h) {
        fprintf(stderr, "ccall: cannot open library %s: %s\n", lib_name, dlerror());
        return NULL;
    }

    return h;
}

static void *_ccall_sym(void *handle, const char *name) {
    dlerror();
    return dlsym(handle, name);
}

static const char *_ccall_sym_error(void *sym) {
    (void)sym;
    return dlerror();
}
#endif

static TriadValue _tri_ccall(TriadValue lib_v, TriadValue fn_v, int32_t nargs, TriadValue *args) {
    if (lib_v.tag != TRIAD_STRING || fn_v.tag != TRIAD_STRING) {
        fprintf(stderr, "ccall: lib and func must be strings\n");
        return TRIAD_NONE_VAL;
    }

    void *handle = _ccall_dlopen(lib_v.as.sval->data);
    if (!handle) {
        return TRIAD_NONE_VAL;
    }

    void *sym = _ccall_sym(handle, fn_v.as.sval->data);
    const char *err = _ccall_sym_error(sym);

    if (err) {
        fprintf(stderr, "ccall: cannot find symbol %s: %s\n", fn_v.as.sval->data, err);
        return TRIAD_NONE_VAL;
    }

    if (nargs == 1) {
        TriadValue a = args[0];

        if (a.tag == TRIAD_FLOAT) {
            typedef double (*fn1d)(double);
            double r = ((fn1d)sym)(a.as.fval);
            return TRIAD_FLOAT(r);
        }

        if (a.tag == TRIAD_INT) {
            typedef int (*fn1i)(int);
            int r = ((fn1i)sym)((int)a.as.ival);
            return TRIAD_INT(r);
        }

        if (a.tag == TRIAD_STRING) {
            typedef int (*fn1s)(const char*);
            int r = ((fn1s)sym)(a.as.sval->data);
            return TRIAD_INT(r);
        }
    } else if (nargs == 2) {
        TriadValue a0 = args[0];
        TriadValue a1 = args[1];

        double d0 = (a0.tag == TRIAD_INT) ? (double)a0.as.ival : a0.as.fval;
        double d1 = (a1.tag == TRIAD_INT) ? (double)a1.as.ival : a1.as.fval;

        typedef double (*fn2d)(double, double);
        double r = ((fn2d)sym)(d0, d1);
        return TRIAD_FLOAT(r);
    } else if (nargs == 0) {
        typedef double (*fn0)(void);
        double r = ((fn0)sym)();
        return TRIAD_FLOAT(r);
    }

    fprintf(stderr, "ccall: unsupported arity %d\n", nargs);
    return TRIAD_NONE_VAL;
}
'''

_CCALL_NAMES = frozenset({"ccall"})

_C_PYTHON = '#include "triad_python.h"\n'
_PYTHON_NAMES = frozenset({"py_call", "py_eval", "py_exec"})

_NATIVE_MODULES = frozenset({"math", "random"})

_MATH_FNS = {
    "sqrt": "triad_math_sqrt", "sin": "triad_math_sin", "cos": "triad_math_cos",
    "tan": "triad_math_tan", "exp": "triad_math_exp", "log": "triad_math_log",
    "log10": "triad_math_log10", "floor": "triad_math_floor",
    "ceil": "triad_math_ceil", "pow": "triad_math_pow",
}
_MATH_CONSTS = {"pi": "3.14159265358979323846", "e": "2.71828182845904523536"}

_RANDOM_FNS = {
    "random": ("triad_random_double", 0),
    "uniform": ("triad_random_uniform", 2),
    "randint": ("triad_random_int", 2),
    "seed": ("triad_random_seed", 1),
}

_C_DISPATCH_COMMON = r'''
static TriadString *_triad_str_join(TriadString *sep, TriadList *l) {
    TriadString *out = triad_str_new("");
    for (int32_t i = 0; i < l->len; i++) {
        if (i > 0) {
            TriadString *t = triad_str_concat(out, sep);
            out = t;
        }
        TriadString *e = triad_value_to_string(l->items[i]);
        TriadString *t2 = triad_str_concat(out, e);
        out = t2;
    }
    return out;
}

static TriadString *_triad_format_spec(TriadValue v, const char *spec) {
    char buf[256];
    int width = -1, prec = -1;
    char conv = 0;
    const char *p = spec;
    if (*p >= '0' && *p <= '9') { width = atoi(p); while (*p >= '0' && *p <= '9') p++; }
    if (*p == '.') { p++; prec = atoi(p); while (*p >= '0' && *p <= '9') p++; }
    if (*p) conv = *p;
    double d = (v.tag == TRIAD_INT) ? (double)v.as.ival : v.as.fval;
    switch (conv) {
    case 'f':
        snprintf(buf, sizeof(buf), "%*.*f", width < 0 ? 0 : width, prec < 0 ? 6 : prec, d);
        return triad_str_new(buf);
    case 'e':
        snprintf(buf, sizeof(buf), "%*.*e", width < 0 ? 0 : width, prec < 0 ? 6 : prec, d);
        return triad_str_new(buf);
    case 'g':
        snprintf(buf, sizeof(buf), "%*.*g", width < 0 ? 0 : width, prec < 0 ? 6 : prec, d);
        return triad_str_new(buf);
    case '%':
        snprintf(buf, sizeof(buf), "%*.*f%%", width < 0 ? 0 : width, prec < 0 ? 6 : prec, d * 100.0);
        return triad_str_new(buf);
    case 'd':
        snprintf(buf, sizeof(buf), "%*lld", width < 0 ? 0 : width,
                 (long long)(v.tag == TRIAD_INT ? v.as.ival : (int64_t)d));
        return triad_str_new(buf);
    default:
        return triad_value_to_string(v);
    }
}

static TriadValue _triad_dispatch_err(const char *what, const char *name) {
    fprintf(stderr, "runtime error: %s '%s' not supported on this value\n", what, name);
    exit(1);
    return TRIAD_NONE_VAL;
}

static TriadValue _triad_dispatch_err_tag(const char *what, const char *name, int tag) {
    fprintf(stderr, "runtime error: %s '%s' not supported (value tag=%d)\n", what, name, tag);
    exit(1);
    return TRIAD_NONE_VAL;
}

static double _triad_as_dbl(TriadValue v) {
    return v.tag == TRIAD_INT ? (double)v.as.ival : v.as.fval;
}

static TriadValue _triad_reshape_v(TriadValue a, TriadValue shape) {
    int32_t dims[8];
    int32_t nd = 0;
    if (shape.tag == TRIAD_LIST) {
        for (int32_t i = 0; i < triad_list_len(shape.as.lval) && nd < 8; i++)
            dims[nd++] = (int32_t)triad_list_get(shape.as.lval, i).as.ival;
    } else {
        dims[nd++] = (int32_t)shape.as.ival;
    }
    return (TriadValue){.tag = TRIAD_NDARRAY,
                        .as = {.aval = triad_ndarray_reshape(a.as.aval, nd, dims)}};
}

static TriadValue _triad_num_binop(char op, TriadValue l, TriadValue r) {
    if (op == '+' && (l.tag == TRIAD_STRING || r.tag == TRIAD_STRING)) {
        return (TriadValue){.tag = TRIAD_STRING, .as = {.sval =
            triad_str_concat(triad_value_to_string(l), triad_value_to_string(r))}};
    }
    if (op == '+' && l.tag == TRIAD_LIST && r.tag == TRIAD_LIST) {
        TriadList *out = triad_list_new_cap(l.as.lval->len + r.as.lval->len);
        for (int32_t i = 0; i < l.as.lval->len; i++) triad_list_push(out, l.as.lval->items[i]);
        for (int32_t i = 0; i < r.as.lval->len; i++) triad_list_push(out, r.as.lval->items[i]);
        return (TriadValue){.tag = TRIAD_LIST, .as = {.lval = out}};
    }
    if (op == 'm')
        return _triad_dispatch_err("operator", "@");
    if (l.tag == TRIAD_INT && r.tag == TRIAD_INT) {
        int64_t a = l.as.ival, b = r.as.ival;
        switch (op) {
        case '+': return TRIAD_INT(a + b);
        case '-': return TRIAD_INT(a - b);
        case '*': return TRIAD_INT(a * b);
        case '/': return TRIAD_FLOAT((double)a / (double)b);
        case 'f': { int64_t q = a / b; if ((a % b != 0) && ((a < 0) != (b < 0))) q--; return TRIAD_INT(q); }
        case '%': return TRIAD_INT(((a % b) + b) % b);
        case 'p': {
            if (b >= 0) {
                int64_t acc = 1, base = a, e = b;
                while (e > 0) { if (e & 1) acc *= base; base *= base; e >>= 1; }
                return TRIAD_INT(acc);
            }
            return TRIAD_FLOAT(triad_math_pow((double)a, (double)b));
        }
        }
    }
    double a = _triad_as_dbl(l), b = _triad_as_dbl(r);
    switch (op) {
    case '+': return TRIAD_FLOAT(a + b);
    case '-': return TRIAD_FLOAT(a - b);
    case '*': return TRIAD_FLOAT(a * b);
    case '/': return TRIAD_FLOAT(a / b);
    case 'f': return TRIAD_FLOAT(floor(a / b));
    case '%': return TRIAD_FLOAT(fmod(fmod(a, b) + b, b));
    case 'p': return TRIAD_FLOAT(triad_math_pow(a, b));
    }
    return TRIAD_NONE_VAL;
}

static TriadValue _triad_num_cmp(char op, TriadValue l, TriadValue r) {
    if (l.tag == TRIAD_STRING && r.tag == TRIAD_STRING) {
        int c = strcmp(l.as.sval->data, r.as.sval->data);
        switch (op) {
        case '=': return TRIAD_BOOL(c == 0);
        case '!': return TRIAD_BOOL(c != 0);
        case '<': return TRIAD_BOOL(c < 0);
        case '>': return TRIAD_BOOL(c > 0);
        case 'l': return TRIAD_BOOL(c <= 0);
        case 'g': return TRIAD_BOOL(c >= 0);
        }
    }
    if (op == '=' || op == '!') {
        bool eq;
        if ((l.tag == TRIAD_INT || l.tag == TRIAD_FLOAT) &&
            (r.tag == TRIAD_INT || r.tag == TRIAD_FLOAT))
            eq = _triad_as_dbl(l) == _triad_as_dbl(r);
        else
            eq = triad_value_eq(l, r);
        return TRIAD_BOOL(op == '=' ? eq : !eq);
    }
    double a = _triad_as_dbl(l), b = _triad_as_dbl(r);
    switch (op) {
    case '<': return TRIAD_BOOL(a < b);
    case '>': return TRIAD_BOOL(a > b);
    case 'l': return TRIAD_BOOL(a <= b);
    case 'g': return TRIAD_BOOL(a >= b);
    }
    return TRIAD_BOOL(false);
}

static TriadValue _triad_identity_eq(TriadValue l, TriadValue r) {
    if (l.tag == TRIAD_INT && r.tag == TRIAD_INT)
        return TRIAD_BOOL(l.as.ival == r.as.ival);
    if (l.tag == TRIAD_FLOAT && r.tag == TRIAD_FLOAT)
        return TRIAD_BOOL(l.as.fval == r.as.fval);
    if (l.tag == TRIAD_BOOL && r.tag == TRIAD_BOOL)
        return TRIAD_BOOL(l.as.bval == r.as.bval);
    if (l.tag == TRIAD_NONE && r.tag == TRIAD_NONE)
        return TRIAD_BOOL(true);
    return TRIAD_BOOL(l.as.ival == r.as.ival);
}

static TriadValue _triad_bitwise(char op, TriadValue l, TriadValue r) {
    int64_t a, b;
    if (l.tag == TRIAD_INT) a = l.as.ival;
    else if (l.tag == TRIAD_FLOAT) a = (int64_t)l.as.fval;
    else if (l.tag == TRIAD_BOOL) a = l.as.bval ? 1 : 0;
    else return _triad_dispatch_err("operator", "bitwise");
    if (r.tag == TRIAD_INT) b = r.as.ival;
    else if (r.tag == TRIAD_FLOAT) b = (int64_t)r.as.fval;
    else if (r.tag == TRIAD_BOOL) b = r.as.bval ? 1 : 0;
    else return _triad_dispatch_err("operator", "bitwise");
    switch (op) {
    case '&': return TRIAD_INT(a & b);
    case '|': return TRIAD_INT(a | b);
    case '^': return TRIAD_INT(a ^ b);
    case '<': return TRIAD_INT(a << b);
    case '>': return TRIAD_INT(a >> b);
    }
    return TRIAD_NONE_VAL;
}

static TriadValue _triad_bitwise_not(TriadValue v) {
    int64_t a;
    if (v.tag == TRIAD_INT) a = v.as.ival;
    else if (v.tag == TRIAD_FLOAT) a = (int64_t)v.as.fval;
    else if (v.tag == TRIAD_BOOL) a = v.as.bval ? 1 : 0;
    else return _triad_dispatch_err("operator", "~");
    return TRIAD_INT(~a);
}
'''

_C_DISPATCH_PY = _C_DISPATCH_COMMON + r'''
#ifndef TRIAD_SCRIPT_DIR
#define TRIAD_SCRIPT_DIR NULL
#endif

static TriadValue _triad_binop(char op, TriadValue l, TriadValue r) {
    if (l.tag == TRIAD_PYOBJ || r.tag == TRIAD_PYOBJ)
        return triad_py_binop(op, l, r);
    return _triad_num_binop(op, l, r);
}

static TriadValue _triad_to_float(TriadValue v) {
    if (v.tag == TRIAD_FLOAT) return v;
    if (v.tag == TRIAD_INT) return TRIAD_FLOAT((double)v.as.ival);
    if (v.tag == TRIAD_STRING) return TRIAD_FLOAT(atof(v.as.sval->data));
    if (v.tag == TRIAD_BOOL) return TRIAD_FLOAT(v.as.bval ? 1.0 : 0.0);
    if (v.tag == TRIAD_PYOBJ) return TRIAD_FLOAT(triad_py_to_double(v));
    return v;
}

static TriadValue _triad_to_int(TriadValue v) {
    if (v.tag == TRIAD_INT) return v;
    if (v.tag == TRIAD_FLOAT) return TRIAD_INT((int64_t)v.as.fval);
    if (v.tag == TRIAD_STRING) return TRIAD_INT(atoll(v.as.sval->data));
    if (v.tag == TRIAD_BOOL) return TRIAD_INT(v.as.bval ? 1 : 0);
    if (v.tag == TRIAD_PYOBJ) return TRIAD_INT(triad_py_to_int(v));
    return v;
}

static TriadValue _triad_cmp(char op, TriadValue l, TriadValue r) {
    if (l.tag == TRIAD_PYOBJ || r.tag == TRIAD_PYOBJ)
        return triad_py_cmp(op, l, r);
    return _triad_num_cmp(op, l, r);
}

static TriadValue _triad_contains(TriadValue item, TriadValue container) {
    if (container.tag == TRIAD_PYOBJ)
        return triad_py_contains(item, container);
    if (container.tag == TRIAD_LIST)
        return TRIAD_BOOL(triad_list_contains(container.as.lval, item));
    if (container.tag == TRIAD_DICT)
        return TRIAD_BOOL(triad_dict_has(container.as.dval, item.as.sval));
    if (container.tag == TRIAD_STRING)
        return TRIAD_BOOL(triad_str_contains(container.as.sval, item.as.sval->data));
    return TRIAD_BOOL(false);
}

static TriadValue _triad_dict_get_default(TriadValue obj, TriadValue key, TriadValue def) {
    if (obj.tag == TRIAD_DICT) {
        TriadString *k = (key.tag == TRIAD_STRING) ? key.as.sval : triad_value_to_string(key);
        TriadValue v = triad_dict_get(obj.as.dval, k);
        return v.tag == TRIAD_NONE ? def : v;
    }
    return _triad_dispatch_err_tag("method", "get", (int)obj.tag);
}

static TriadValue _triad_call_value(TriadValue fn, int32_t n, TriadValue *args,
                                    int32_t nkw, const char **kwn, TriadValue *kwv) {
    if (fn.tag == TRIAD_PYOBJ)
        return triad_py_call_value(fn, n, args, nkw, kwn, kwv);
    if (fn.tag == TRIAD_NATIVE_FN)
        return fn.as.nfn(n, args);
    if (fn.tag == TRIAD_CLOSURE)
        return triad_closure_call(fn.as.cval, n, args);
    return _triad_dispatch_err("call", "value");
}

static TriadValue _triad_method_fallback(TriadValue obj, const char *m,
                                         int32_t n, TriadValue *args,
                                         int32_t nkw, const char **kwn, TriadValue *kwv) {
    if (obj.tag == TRIAD_PYOBJ)
        return triad_py_method_kw(obj, m, n, args, nkw, kwn, kwv);
    if (obj.tag == TRIAD_REGEX) {
        const char *s = (n > 0 && args[0].tag == TRIAD_STRING && args[0].as.sval) ? args[0].as.sval->data : "";
        if (!strcmp(m, "test")) return TRIAD_BOOL(triad_regex_test(obj.as.rxval, s));
        if (!strcmp(m, "search")) return triad_regex_search(obj.as.rxval, s);
        if (!strcmp(m, "match")) return triad_regex_match(obj.as.rxval, s);
        if (!strcmp(m, "findall")) return triad_regex_findall(obj.as.rxval, s);
    }
    if (obj.tag == TRIAD_OBJECT) {
        TriadValue f = triad_object_get(obj.as.oval, triad_str_new(m));
        if (f.tag != TRIAD_NONE) {
            return _triad_call_value(f, n, args, nkw, kwn, kwv);
        }
        if (obj.as.oval->class_meta) {
            TriadNativeFn fn = triad_class_meta_resolve_method(obj.as.oval->class_meta, m);
            if (fn) return fn(n, args);
        }
    }
    fprintf(stderr, "runtime error: method '%s' not supported (value tag=%d)\n", m, (int)obj.tag);
    exit(1);
    return TRIAD_NONE_VAL;
}

static TriadValue _triad_field_any(TriadValue obj, const char *name) {
    if (obj.tag == TRIAD_PYOBJ)
        return triad_py_getattr(obj, name);
    if (obj.tag == TRIAD_REGEX) {
        if (!strcmp(name, "pattern") || !strcmp(name, "source"))
            return (TriadValue){.tag = TRIAD_STRING, .as = {.sval = triad_str_new(triad_regex_pattern(obj.as.rxval))}};
        if (!strcmp(name, "flags"))
            return (TriadValue){.tag = TRIAD_STRING, .as = {.sval = triad_str_new(triad_regex_flags(obj.as.rxval))}};
    }
    if (obj.tag == TRIAD_OBJECT)
        return triad_object_get(obj.as.oval, triad_str_new(name));
    if (obj.tag == TRIAD_DICT)
        return triad_dict_get(obj.as.dval, triad_str_new(name));
    return _triad_dispatch_err_tag("field", name, (int)obj.tag);
}

static TriadValue _triad_setattr_any(TriadValue obj, const char *name, TriadValue val) {
    if (obj.tag == TRIAD_PYOBJ)
        return triad_py_setattr(obj, name, val);
    if (obj.tag == TRIAD_OBJECT) {
        triad_object_set(obj.as.oval, triad_str_new(name), val);
        return TRIAD_NONE_VAL;
    }
    return _triad_dispatch_err("field-assign", name);
}

static TriadValue _triad_delattr_any(TriadValue obj, const char *name) {
    if (obj.tag == TRIAD_OBJECT) {
        triad_object_set(obj.as.oval, triad_str_new(name), TRIAD_NONE_VAL);
        return TRIAD_NONE_VAL;
    }
    if (obj.tag == TRIAD_DICT) {
        triad_dict_del(obj.as.dval, triad_str_new(name));
        return TRIAD_NONE_VAL;
    }
    if (obj.tag == TRIAD_PYOBJ)
        return triad_py_setattr(obj, name, TRIAD_NONE_VAL);
    return _triad_dispatch_err("field-del", name);
}

static TriadValue _triad_index_any(TriadValue obj, TriadValue idx) {
    if (obj.tag == TRIAD_LIST) {
        int64_t i = idx.as.ival;
        if (i < 0) i += triad_list_len(obj.as.lval);
        return triad_list_get(obj.as.lval, (int32_t)i);
    }
    if (obj.tag == TRIAD_DICT) {
        TriadString *k = (idx.tag == TRIAD_STRING) ? idx.as.sval : triad_value_to_string(idx);
        return triad_dict_get(obj.as.dval, k);
    }
    if (obj.tag == TRIAD_STRING) {
        int64_t i = idx.as.ival;
        if (i < 0) i += obj.as.sval->len;
        return (TriadValue){.tag = TRIAD_STRING,
                            .as = {.sval = triad_str_slice(obj.as.sval, (int32_t)i, (int32_t)i + 1, 1)}};
    }
    if (obj.tag == TRIAD_PYOBJ)
        return triad_py_getitem(obj, idx);
    return _triad_dispatch_err_tag("index", "[]", (int)obj.tag);
}

static TriadValue _triad_setindex_any(TriadValue obj, TriadValue idx, TriadValue val) {
    if (obj.tag == TRIAD_LIST) {
        int64_t i = idx.as.ival;
        if (i < 0) i += triad_list_len(obj.as.lval);
        triad_list_set(obj.as.lval, (int32_t)i, val);
        return TRIAD_NONE_VAL;
    }
    if (obj.tag == TRIAD_DICT) {
        TriadString *k = (idx.tag == TRIAD_STRING) ? idx.as.sval : triad_value_to_string(idx);
        triad_dict_set(obj.as.dval, k, val);
        return TRIAD_NONE_VAL;
    }
    if (obj.tag == TRIAD_PYOBJ)
        return triad_py_setitem(obj, idx, val);
    return _triad_dispatch_err_tag("index-assign", "[]=", (int)obj.tag);
}

static TriadValue _triad_delindex_any(TriadValue obj, TriadValue idx) {
    if (obj.tag == TRIAD_LIST) {
        int64_t i = idx.as.ival;
        if (i < 0) i += triad_list_len(obj.as.lval);
        triad_list_remove_at(obj.as.lval, (int32_t)i);
        return TRIAD_NONE_VAL;
    }
    if (obj.tag == TRIAD_DICT) {
        TriadString *k = (idx.tag == TRIAD_STRING) ? idx.as.sval : triad_value_to_string(idx);
        triad_dict_del(obj.as.dval, k);
        return TRIAD_NONE_VAL;
    }
    if (obj.tag == TRIAD_PYOBJ)
        return triad_py_setitem(obj, idx, TRIAD_NONE_VAL);
    return _triad_dispatch_err_tag("index-del", "[]", (int)obj.tag);
}

static TriadValue _triad_iter_prep(TriadValue v) {
    if (v.tag == TRIAD_PYOBJ)
        return triad_py_to_list(v);
    if (v.tag == TRIAD_DICT) {
        return (TriadValue){.tag = TRIAD_LIST, .as = {.lval = triad_dict_keys(v.as.dval)}};
    }
    if (v.tag == TRIAD_STRING) {
        TriadList *cl = triad_list_new_cap(v.as.sval->len);
        for (int32_t i = 0; i < v.as.sval->len; i++) {
            triad_list_push(cl, (TriadValue){.tag = TRIAD_STRING,
                .as = {.sval = triad_str_new_len(v.as.sval->data + i, 1)}});
        }
        return (TriadValue){.tag = TRIAD_LIST, .as = {.lval = cl}};
    }
    if (v.tag == TRIAD_TUPLE) {
        TriadList *cl = triad_list_new_cap(v.as.tval->len);
        for (int32_t i = 0; i < v.as.tval->len; i++) {
            triad_list_push(cl, triad_tuple_get(v.as.tval, i));
        }
        return (TriadValue){.tag = TRIAD_LIST, .as = {.lval = cl}};
    }
    if (v.tag == TRIAD_SET) {
        TriadList *cl = triad_set_to_list(v.as.sset);
        return (TriadValue){.tag = TRIAD_LIST, .as = {.lval = cl}};
    }
    return v;
}

static TriadValue _triad_len_any(TriadValue v) {
    switch (v.tag) {
    case TRIAD_LIST:      return TRIAD_INT(triad_list_len(v.as.lval));
    case TRIAD_STRING:    return TRIAD_INT(triad_str_len(v.as.sval));
    case TRIAD_DICT:      return TRIAD_INT(triad_dict_len(v.as.dval));
    case TRIAD_TUPLE:    return TRIAD_INT(triad_tuple_len(v.as.tval));
    case TRIAD_BYTES:    return TRIAD_INT(triad_bytes_len(v.as.bvalp));
    case TRIAD_SET:      return TRIAD_INT(triad_set_len(v.as.sset));
    case TRIAD_PYOBJ:    return triad_py_len(v);
    default:              return TRIAD_INT(0);
    }
}

static TriadValue _triad_slice_any(TriadValue obj, int64_t s, int64_t e,
                                   int64_t step, int has_s, int has_e) {
    if (obj.tag == TRIAD_PYOBJ)
        return triad_py_getslice(obj, s, e, step, has_s, has_e);
    if (obj.tag == TRIAD_LIST) {
        int32_t n = triad_list_len(obj.as.lval);
        int64_t st = has_s ? s : 0, en = has_e ? e : n;
        if (st < 0) st += n;
        if (en < 0) en += n;
        if (st < 0) st = 0;
        if (en < 0) en = 0;
        if (st > n) st = n;
        if (en > n) en = n;
        return (TriadValue){.tag = TRIAD_LIST, .as = {.lval =
            triad_list_slice(obj.as.lval, (int32_t)st, (int32_t)en, (int32_t)step)}};
    }
    if (obj.tag == TRIAD_STRING) {
        int32_t n = obj.as.sval->len;
        int64_t st = has_s ? s : 0, en = has_e ? e : n;
        if (st < 0) st += n;
        if (en < 0) en += n;
        if (st < 0) st = 0;
        if (en < 0) en = 0;
        if (st > n) st = n;
        if (en > n) en = n;
        return (TriadValue){.tag = TRIAD_STRING, .as = {.sval =
            triad_str_slice(obj.as.sval, (int32_t)st, (int32_t)en, (int32_t)step)}};
    }
    return _triad_dispatch_err_tag("slice", "[:]", (int)obj.tag);
}
'''

_C_DISPATCH_NOPY = _C_DISPATCH_COMMON + r'''
static TriadValue _triad_binop(char op, TriadValue l, TriadValue r) {
    return _triad_num_binop(op, l, r);
}

static TriadValue _triad_to_float(TriadValue v) {
    if (v.tag == TRIAD_FLOAT) return v;
    if (v.tag == TRIAD_INT) return TRIAD_FLOAT((double)v.as.ival);
    if (v.tag == TRIAD_STRING) return TRIAD_FLOAT(atof(v.as.sval->data));
    if (v.tag == TRIAD_BOOL) return TRIAD_FLOAT(v.as.bval ? 1.0 : 0.0);
    return v;
}

static TriadValue _triad_to_int(TriadValue v) {
    if (v.tag == TRIAD_INT) return v;
    if (v.tag == TRIAD_FLOAT) return TRIAD_INT((int64_t)v.as.fval);
    if (v.tag == TRIAD_STRING) return TRIAD_INT(atoll(v.as.sval->data));
    if (v.tag == TRIAD_BOOL) return TRIAD_INT(v.as.bval ? 1 : 0);
    return v;
}

static TriadValue _triad_cmp(char op, TriadValue l, TriadValue r) {
    return _triad_num_cmp(op, l, r);
}

static TriadValue _triad_contains(TriadValue item, TriadValue container) {
    if (container.tag == TRIAD_LIST)
        return TRIAD_BOOL(triad_list_contains(container.as.lval, item));
    if (container.tag == TRIAD_DICT)
        return TRIAD_BOOL(triad_dict_has(container.as.dval, item.as.sval));
    if (container.tag == TRIAD_STRING)
        return TRIAD_BOOL(triad_str_contains(container.as.sval, item.as.sval->data));
    return TRIAD_BOOL(false);
}

static TriadValue _triad_dict_get_default(TriadValue obj, TriadValue key, TriadValue def) {
    if (obj.tag == TRIAD_DICT) {
        TriadString *k = (key.tag == TRIAD_STRING) ? key.as.sval : triad_value_to_string(key);
        TriadValue v = triad_dict_get(obj.as.dval, k);
        return v.tag == TRIAD_NONE ? def : v;
    }
    return _triad_dispatch_err_tag("method", "get", (int)obj.tag);
}

static TriadValue _triad_call_value(TriadValue fn, int32_t n, TriadValue *args,
                                    int32_t nkw, const char **kwn, TriadValue *kwv) {
    (void)nkw; (void)kwn; (void)kwv;
    if (fn.tag == TRIAD_NATIVE_FN)
        return fn.as.nfn(n, args);
    if (fn.tag == TRIAD_CLOSURE)
        return triad_closure_call(fn.as.cval, n, args);
    return _triad_dispatch_err("call", "value");
}

static TriadValue _triad_method_fallback(TriadValue obj, const char *m,
                                         int32_t n, TriadValue *args,
                                         int32_t nkw, const char **kwn, TriadValue *kwv) {
    if (obj.tag == TRIAD_REGEX) {
        const char *s = (n > 0 && args[0].tag == TRIAD_STRING && args[0].as.sval) ? args[0].as.sval->data : "";
        if (!strcmp(m, "test")) return TRIAD_BOOL(triad_regex_test(obj.as.rxval, s));
        if (!strcmp(m, "search")) return triad_regex_search(obj.as.rxval, s);
        if (!strcmp(m, "match")) return triad_regex_match(obj.as.rxval, s);
        if (!strcmp(m, "findall")) return triad_regex_findall(obj.as.rxval, s);
    }
    if (obj.tag == TRIAD_OBJECT) {
        TriadValue f = triad_object_get(obj.as.oval, triad_str_new(m));
        if (f.tag != TRIAD_NONE)
            return _triad_call_value(f, n, args, nkw, kwn, kwv);
        if (obj.as.oval->class_meta) {
            TriadNativeFn fn = triad_class_meta_resolve_method(obj.as.oval->class_meta, m);
            if (fn) return fn(n, args);
        }
    }
    fprintf(stderr, "runtime error: method '%s' not supported (value tag=%d)\n", m, (int)obj.tag);
    exit(1);
    return TRIAD_NONE_VAL;
}

static TriadValue _triad_field_any(TriadValue obj, const char *name) {
    if (obj.tag == TRIAD_REGEX) {
        if (!strcmp(name, "pattern") || !strcmp(name, "source"))
            return (TriadValue){.tag = TRIAD_STRING, .as = {.sval = triad_str_new(triad_regex_pattern(obj.as.rxval))}};
        if (!strcmp(name, "flags"))
            return (TriadValue){.tag = TRIAD_STRING, .as = {.sval = triad_str_new(triad_regex_flags(obj.as.rxval))}};
    }
    if (obj.tag == TRIAD_OBJECT)
        return triad_object_get(obj.as.oval, triad_str_new(name));
    if (obj.tag == TRIAD_DICT)
        return triad_dict_get(obj.as.dval, triad_str_new(name));
    return _triad_dispatch_err_tag("field", name, (int)obj.tag);
}

static TriadValue _triad_setattr_any(TriadValue obj, const char *name, TriadValue val) {
    if (obj.tag == TRIAD_OBJECT) {
        triad_object_set(obj.as.oval, triad_str_new(name), val);
        return TRIAD_NONE_VAL;
    }
    return _triad_dispatch_err("field-assign", name);
}

static TriadValue _triad_delattr_any(TriadValue obj, const char *name) {
    if (obj.tag == TRIAD_OBJECT) {
        triad_object_set(obj.as.oval, triad_str_new(name), TRIAD_NONE_VAL);
        return TRIAD_NONE_VAL;
    }
    if (obj.tag == TRIAD_DICT) {
        triad_dict_del(obj.as.dval, triad_str_new(name));
        return TRIAD_NONE_VAL;
    }
    return _triad_dispatch_err("field-del", name);
}

static TriadValue _triad_index_any(TriadValue obj, TriadValue idx) {
    if (obj.tag == TRIAD_LIST) {
        int64_t i = idx.as.ival;
        if (i < 0) i += triad_list_len(obj.as.lval);
        return triad_list_get(obj.as.lval, (int32_t)i);
    }
    if (obj.tag == TRIAD_DICT) {
        TriadString *k = (idx.tag == TRIAD_STRING) ? idx.as.sval : triad_value_to_string(idx);
        return triad_dict_get(obj.as.dval, k);
    }
    if (obj.tag == TRIAD_STRING) {
        int64_t i = idx.as.ival;
        if (i < 0) i += obj.as.sval->len;
        return (TriadValue){.tag = TRIAD_STRING,
                            .as = {.sval = triad_str_slice(obj.as.sval, (int32_t)i, (int32_t)i + 1, 1)}};
    }
    return _triad_dispatch_err_tag("index", "[]", (int)obj.tag);
}

static TriadValue _triad_setindex_any(TriadValue obj, TriadValue idx, TriadValue val) {
    if (obj.tag == TRIAD_LIST) {
        int64_t i = idx.as.ival;
        if (i < 0) i += triad_list_len(obj.as.lval);
        triad_list_set(obj.as.lval, (int32_t)i, val);
        return TRIAD_NONE_VAL;
    }
    if (obj.tag == TRIAD_DICT) {
        TriadString *k = (idx.tag == TRIAD_STRING) ? idx.as.sval : triad_value_to_string(idx);
        triad_dict_set(obj.as.dval, k, val);
        return TRIAD_NONE_VAL;
    }
    return _triad_dispatch_err_tag("index-assign", "[]=", (int)obj.tag);
}

static TriadValue _triad_delindex_any(TriadValue obj, TriadValue idx) {
    if (obj.tag == TRIAD_LIST) {
        int64_t i = idx.as.ival;
        if (i < 0) i += triad_list_len(obj.as.lval);
        triad_list_remove_at(obj.as.lval, (int32_t)i);
        return TRIAD_NONE_VAL;
    }
    if (obj.tag == TRIAD_DICT) {
        TriadString *k = (idx.tag == TRIAD_STRING) ? idx.as.sval : triad_value_to_string(idx);
        triad_dict_del(obj.as.dval, k);
        return TRIAD_NONE_VAL;
    }
    return _triad_dispatch_err_tag("index-del", "[]", (int)obj.tag);
}

static TriadValue _triad_iter_prep(TriadValue v) {
    if (v.tag == TRIAD_DICT)
        return (TriadValue){.tag = TRIAD_LIST, .as = {.lval = triad_dict_keys(v.as.dval)}};
    if (v.tag == TRIAD_STRING) {
        TriadList *cl = triad_list_new_cap(v.as.sval->len);
        for (int32_t i = 0; i < v.as.sval->len; i++) {
            triad_list_push(cl, (TriadValue){.tag = TRIAD_STRING,
                .as = {.sval = triad_str_new_len(v.as.sval->data + i, 1)}});
        }
        return (TriadValue){.tag = TRIAD_LIST, .as = {.lval = cl}};
    }
    return v;
}

static TriadValue _triad_len_any(TriadValue v) {
    switch (v.tag) {
    case TRIAD_LIST:   return TRIAD_INT(triad_list_len(v.as.lval));
    case TRIAD_STRING: return TRIAD_INT(triad_str_len(v.as.sval));
    case TRIAD_DICT:   return TRIAD_INT(triad_dict_len(v.as.dval));
    default:           return TRIAD_INT(0);
    }
}

static TriadValue _triad_slice_any(TriadValue obj, int64_t s, int64_t e,
                                   int64_t step, int has_s, int has_e) {
    if (obj.tag == TRIAD_LIST) {
        int32_t n = triad_list_len(obj.as.lval);
        int64_t st = has_s ? s : 0, en = has_e ? e : n;
        if (st < 0) st += n;
        if (en < 0) en += n;
        if (st < 0) st = 0;
        if (en < 0) en = 0;
        if (st > n) st = n;
        if (en > n) en = n;
        return (TriadValue){.tag = TRIAD_LIST, .as = {.lval =
            triad_list_slice(obj.as.lval, (int32_t)st, (int32_t)en, (int32_t)step)}};
    }
    if (obj.tag == TRIAD_STRING) {
        int32_t n = obj.as.sval->len;
        int64_t st = has_s ? s : 0, en = has_e ? e : n;
        if (st < 0) st += n;
        if (en < 0) en += n;
        if (st < 0) st = 0;
        if (en < 0) en = 0;
        if (st > n) st = n;
        if (en > n) en = n;
        return (TriadValue){.tag = TRIAD_STRING, .as = {.sval =
            triad_str_slice(obj.as.sval, (int32_t)st, (int32_t)en, (int32_t)step)}};
    }
    return _triad_dispatch_err_tag("slice", "[:]", (int)obj.tag);
}
'''

_SOLVER_NAMES = frozenset({"solver_solve", "solve"})
_ML_NAMES = frozenset({
    "ml_tensor", "ml_zeros", "ml_randn", "ml_triad", "ml_forward",
    "ml_relu", "ml_sigmoid", "ml_tanh_act", "ml_softmax",
    "ml_mse_loss", "ml_cross_entropy", "ml_backward",
    "ml_item", "ml_data", "ml_seq_new", "ml_seq_set", "ml_seq_forward",
    "ml_adam", "ml_sgd", "ml_adam_step", "ml_adam_zero",
    "ml_sgd_step", "ml_sgd_zero", "ml_tensor_add", "ml_tensor_sub",
    "ml_tensor_mul", "ml_tensor_matmul", "ml_tensor_sum",
    "ml_tensor_mean", "ml_print",
    "ml_embedding", "ml_embedding_forward", "ml_layernorm",
    "ml_layernorm_forward", "ml_mha", "ml_mha_forward",
    "ml_transformer", "ml_transformer_forward", "ml_transformer_adam",
    "ml_wave", "ml_wave_forward", "ml_wave_adam",
})

def _detect_preamble_needs(module):

    needs_solver = False
    needs_ml = False
    needs_ccall = False
    needs_python = False
    for imp in list(getattr(module, 'imports', []) or []) + [n for n in module.body if isinstance(n, IRImport)]:
        root = imp.path[0] if imp.path else ''
        if root not in _NATIVE_MODULES:
            needs_python = True
    def _walk(nodes):
        nonlocal needs_solver, needs_ml, needs_ccall, needs_python
        for node in nodes:

            if isinstance(node, IRCall):
                if isinstance(node.func, IRIdent):
                    fname = node.func.name
                    if fname in _SOLVER_NAMES:
                        needs_solver = True
                    if fname in _ML_NAMES:
                        needs_ml = True
                    if fname in _CCALL_NAMES:
                        needs_ccall = True
                    if fname in _PYTHON_NAMES:
                        needs_python = True

            if hasattr(node, "name") and isinstance(node.name, str):
                if node.name in _SOLVER_NAMES:
                    needs_solver = True
                if node.name in _ML_NAMES:
                    needs_ml = True
                if node.name in _CCALL_NAMES:
                    needs_ccall = True
                if node.name in _PYTHON_NAMES:
                    needs_python = True

            if hasattr(node, "args") and isinstance(node.args, list):
                _walk(node.args)
            if hasattr(node, "func") and node.func is not None and hasattr(node.func, "__dict__"):
                _walk([node.func])
            if hasattr(node, "body") and isinstance(node.body, list):
                _walk(node.body)
            if hasattr(node, "else_body") and node.else_body:
                _walk(node.else_body if isinstance(node.else_body, list) else [node.else_body])
            if hasattr(node, "elif_clauses") and node.elif_clauses:
                for cond, body in node.elif_clauses:
                    _walk([cond])
                    _walk(body)
            if hasattr(node, "elements") and isinstance(node.elements, list):
                _walk(node.elements)
            for attr in ("target", "iter", "value", "condition", "expr", "left", "right", "operand"):
                child = getattr(node, attr, None)
                if child is not None and hasattr(child, "__dict__"):
                    _walk([child])
    _walk(module.body)
    return needs_solver, needs_ml, needs_ccall, needs_python

def _build_preamble(module):

    needs_solver, needs_ml, needs_ccall, needs_python = _detect_preamble_needs(module)
    parts = [_C_BASE]
    if needs_solver:
        parts.append("\n" + _C_SOLVER)
    if needs_ml:
        parts.append("\n" + _C_ML)
    if needs_ccall:
        parts.append("\n" + _C_CCALL)
    if needs_python:
        parts.append("\n" + _C_PYTHON)
        parts.append("\n" + _C_DISPATCH_PY)
    else:
        parts.append("\n" + _C_DISPATCH_NOPY)
    return "".join(parts)

_C_MAIN_HEADER = '\nTriadMain()\n'

class CCodeGen:

    def __init__(self):
        self._lines: list[str] = []
        self._indent: int = 0
        self._temp_counter: int = 0
        self._func_forward_decls: list[str] = []
        self._func_defs: list[str] = []
        self._in_function: bool = False
        self._loop_depth: int = 0
        self._closures: dict[str, list[str]] = {}
        self._top_level_fns: set[str] = set()
        self._classes: dict[str, IRClassDecl] = {}
        self._current_class: str = ""
        self._top_level_fn_params: dict[str, list[str]] = {}
        self._top_level_fn_meta: dict[str, tuple[int, int]] = {}
        self._native_mod_aliases: dict[str, str] = {}
        self._py_bound_names: set[str] = set()
        self._class_meta_registrations: list[IRClassDecl] = []

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
            for part in node.parts:
                if part[1] is not None:
                    self._collect_names_expr(part[1], names)
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
            if not isinstance(node.target, IRIdent):
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
                self._in_function = False
                self._emit_import(node)
            elif isinstance(node, IRTypeDecl):
                self._emit_typedecl(node)
            else:
                self._in_function = False
                self._emit_top_level_stmt(node)
        parts = [_build_preamble(module)]
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
        parts.append('    TriadMainEnd()\n')
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
        elif isinstance(node, IRWith):
            self._gen_with(node)
        elif isinstance(node, IRAssert):
            self._gen_assert(node)
        elif isinstance(node, IRPass):
            self._emit('(void)0;')
        elif isinstance(node, IRDel):
            self._gen_del(node)
        elif isinstance(node, IRAsyncFor):
            self._gen_for(IRFor(node.var, node.iter, node.body))
        elif isinstance(node, IRAsyncWith):
            self._gen_with(IRWith(node.expr, node.var, node.body))
        elif isinstance(node, IRSequence):
            self._gen_sequence(node)
        elif isinstance(node, IRTryCatch):
            self._gen_try_catch(node)
        elif isinstance(node, IRThrow):
            self._gen_throw(node)
        elif isinstance(node, IRDestructLet):
            self._gen_destruct_let(node)
        elif isinstance(node, IRReturn):
            val = self._gen_expr(node.value) if node.value else 'TRIAD_NONE_VAL'
            self._emit(f'return {val};')
        elif isinstance(node, IRMapDestruct):
            self._gen_map_destruct(node)
        elif isinstance(node, IRCouple):
            pass
        elif isinstance(node, IRPair):
            pass
        elif isinstance(node, IRRing):
            pass
        elif isinstance(node, IRAnnotation):
            pass
        elif isinstance(node, IRMatch):
            self._gen_match(node)
        elif isinstance(node, (IRTypeDecl, IREntityDecl, IRWorldDecl, IRRegDecl, IRSubstrateDecl, IRObserve, IRRun)):
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
        elif isinstance(node, IRWith):
            self._gen_with(node)
        elif isinstance(node, IRAssert):
            self._gen_assert(node)
        elif isinstance(node, IRPass):
            self._emit('(void)0;')
        elif isinstance(node, IRDel):
            self._gen_del(node)
        elif isinstance(node, IRAsyncFor):
            self._gen_for(IRFor(node.var, node.iter, node.body))
        elif isinstance(node, IRAsyncWith):
            self._gen_with(IRWith(node.expr, node.var, node.body))
        elif isinstance(node, IRSequence):
            self._gen_sequence(node)
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
        elif isinstance(node, IRMapDestruct):
            self._gen_map_destruct(node)
        elif isinstance(node, IRCouple):
            pass
        elif isinstance(node, IRPair):
            pass
        elif isinstance(node, IRRing):
            pass
        elif isinstance(node, IRAnnotation):
            pass
        elif isinstance(node, IRMatch):
            self._gen_match(node)
        elif isinstance(node, (IRTypeDecl, IREntityDecl, IRWorldDecl, IRRegDecl, IRSubstrateDecl, IRObserve, IRRun)):
            pass

    @staticmethod
    def _c_fn_name(name: str) -> str:
        return f'_triad_fn_{name}'

    def _has_yield(self, stmts: list[object]) -> bool:
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

    def _emit_param_bindings(self, node: IRFunction):
        star, kw = getattr(node, 'star_idx', -1), getattr(node, 'kw_idx', -1)
        for i, pname in enumerate(node.params):
            name = self._sanitize(pname)
            if i == star:
                self._emit(f'TriadValue {name} = ({{ TriadList *_va = triad_list_new(); '
                           f'for (int32_t _vi = {i}; _vi < _nargs; _vi++) '
                           f'{{ if (_args[_vi].tag != TRIAD_DICT || {kw} < 0) triad_list_push(_va, _args[_vi]); }} '
                           f'(TriadValue){{.tag = TRIAD_LIST, .as = {{.lval = _va}}}}; }});')
            elif i == kw:
                self._emit(f'TriadValue {name} = (_nargs > {i} && _args[_nargs - 1].tag == TRIAD_DICT) '
                           f'? _args[_nargs - 1] '
                           f': (TriadValue){{.tag = TRIAD_DICT, .as = {{.dval = triad_dict_new()}}}};')
            else:
                self._emit(f'TriadValue {name} = (_nargs > {i}) ? _args[{i}] : TRIAD_NONE_VAL;')

    def _emit_function(self, node: IRFunction, outer_bound: set[str] | None=None):
        fname = self._c_fn_name(node.name)
        self._top_level_fn_params[node.name] = list(node.params)
        self._top_level_fn_meta = getattr(self, '_top_level_fn_meta', {})
        self._top_level_fn_meta[node.name] = (getattr(node, 'star_idx', -1), getattr(node, 'kw_idx', -1))
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
        self._emit_param_bindings(node)
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
        self._current_class = node.name
        self._class_meta_registrations.append(node)
        if not any(m.name in ('new', '__init__') for m in node.methods):

            ctor = f'_triad_cls_{node.name}_new'
            self._top_level_fns.add(ctor)
            fwd = f'static TriadValue {ctor}(int32_t _nargs, TriadValue *_args);'
            if fwd not in self._func_forward_decls:
                self._func_forward_decls.append(fwd)
            lines = [f'static TriadValue {ctor}(int32_t _nargs, TriadValue *_args) {{',
                     '    TriadValue self = (_nargs > 0) ? _args[0] : TRIAD_NONE_VAL;']
            for i, fname_ in enumerate(node.fields):
                lines.append(f'    if (_nargs > {i + 1}) triad_object_set(self.as.oval, triad_str_new("{fname_}"), _args[{i + 1}]);')
            lines.append('    return self;')
            lines.append('}')
            self._func_defs.append('\n'.join(lines) + '\n')
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
            self._emit('TriadValue _triad_self = (_nargs > 0) ? _args[0] : TRIAD_NONE_VAL;')
            mparams = [p for p in method.params if p != 'self']
            for i, pname in enumerate(mparams):
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

    def _emit_class_meta_registrations(self):
        for cls_node in self._class_meta_registrations:
            name = cls_node.name
            parent = cls_node.parent
            fields_strs = [f'"{f}"' for f in cls_node.fields]
            fields_c = ', '.join(fields_strs)
            fields_arr = f'(const char*[]){{{fields_c}}}' if fields_strs else 'NULL'
            method_entries = []
            for method in cls_node.methods:
                mname = f'_triad_cls_{name}_{method.name}'
                method_entries.append(f'("{method.name}", (TriadNativeFn){mname})')
            if method_entries:
                methods_var = f'_triad_cls_{name}_methods'
                methods_init = ', '.join(method_entries)
                self._file_scope_decls.append(
                    f'static TriadClassMethod {methods_var}[] = {{{methods_init}, {{NULL, NULL}}}};'
                )
                methods_len = len(method_entries)
            else:
                methods_var = 'NULL'
                methods_len = 0
            parent_lookup = f'triad_class_meta_lookup("{parent}")' if parent else 'NULL'
            meta_var = f'_triad_cls_{name}_meta'
            self._file_scope_decls.append(f'static TriadClassMeta *{meta_var};')
            self._lines.append(
                f'{meta_var} = triad_class_meta_new("{name}", {parent_lookup}, '
                f'{fields_arr}, {len(fields_strs)});'
            )
            self._lines.append(
                f'{meta_var}->methods = {methods_var};'
            )
            self._lines.append(
                f'{meta_var}->methods_len = {methods_len};'
            )
            self._lines.append(
                f'triad_class_meta_register({meta_var});'
            )

    def _emit_import(self, node: IRImport):
        dotted = '.'.join(node.path)
        root = node.path[0] if node.path else ''
        if node.names:

            mod_tmp = self._fresh('_mod')
            self._file_scope_decls.append(f'static TriadValue {mod_tmp};')
            self._emit(f'{mod_tmp} = triad_py_import("{dotted}", TRIAD_SCRIPT_DIR);')
            aliases = list(node.aliases or [])
            while len(aliases) < len(node.names):
                aliases.append(None)
            for n, a in zip(node.names, aliases):
                bound = self._sanitize(a or n)
                self._file_scope_decls.append(f'static TriadValue {bound};')
                self._emit(f'{bound} = triad_py_getattr({mod_tmp}, "{n}");')
                self._py_bound_names.add(bound)
            return
        if root in _NATIVE_MODULES and node.alias is None:

            self._native_mod_aliases[root] = root
            return

        if node.alias:
            bound = self._sanitize(node.alias)
            target = dotted
        else:
            bound = self._sanitize(root)
            target = root if len(node.path) > 1 else dotted

            if len(node.path) > 1:
                self._emit(f'(void)triad_py_import("{dotted}", TRIAD_SCRIPT_DIR);')
        self._file_scope_decls.append(f'static TriadValue {bound};')
        self._emit(f'{bound} = triad_py_import("{target}", TRIAD_SCRIPT_DIR);')
        self._py_bound_names.add(bound)

    def _emit_typedecl(self, node: IRTypeDecl):

        fields = [f[0] for f in node.fields]
        self._classes[node.name] = IRClassDecl(name=node.name, fields=fields, methods=[])
        ctor = f'_triad_cls_{node.name}_new'
        self._top_level_fns.add(ctor)
        fwd = f'static TriadValue {ctor}(int32_t _nargs, TriadValue *_args);'
        if fwd not in self._func_forward_decls:
            self._func_forward_decls.append(fwd)
        lines = [f'static TriadValue {ctor}(int32_t _nargs, TriadValue *_args) {{',
                 '    TriadValue self = (_nargs > 0) ? _args[0] : TRIAD_NONE_VAL;']
        for i, fname in enumerate(fields):
            lines.append(f'    if (_nargs > {i + 1}) triad_object_set(self.as.oval, triad_str_new("{fname}"), _args[{i + 1}]);')
        lines.append('    return self;')
        lines.append('}')
        self._func_defs.append('\n'.join(lines) + '\n')

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
        return '(void)0;'

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
            return f'(void)_triad_setindex_any(({obj}), ({idx}), {val});'
        elif isinstance(node.target, IRField):
            obj = self._gen_expr(node.target.obj)
            val = self._gen_expr(node.value)
            return f'(void)_triad_setattr_any(({obj}), "{node.target.field}", {val});'
        target = self._gen_expr(node.target)
        val = self._gen_expr(node.value)
        return f'{target} = {val};'

    def _gen_if(self, node: IRIf):
        cond = self._gen_expr(node.condition)
        self._emit(f'if (triad_is_y({cond})) {{')
        self._indent += 1
        for stmt in node.then_body:
            self._emit_stmt(stmt)
        self._indent -= 1
        for cond_ir, body in node.elif_clauses:
            econd = self._gen_expr(cond_ir)
            self._emit(f'}} else if (triad_is_y({econd})) {{')
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
        self._emit('{')
        self._indent += 1
        self._emit(f'TriadValue {tmp} = _triad_iter_prep({iter_expr});')
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
        self._emit(f'while (triad_is_y({cond})) {{')
        self._indent += 1
        saved = self._loop_depth
        self._loop_depth += 1
        for stmt in node.body:
            self._emit_stmt(stmt)
        self._loop_depth = saved
        self._indent -= 1
        self._emit('}')

    def _gen_with(self, node: IRWith):
        expr = self._gen_expr(node.expr) if node.expr else 'TRIAD_NONE_VAL'
        tmp = self._fresh('_with')
        self._emit('{')
        self._indent += 1
        self._emit(f'TriadValue {tmp} = {expr};')
        if node.var:
            self._emit(f'TriadValue {self._sanitize(node.var)} = {tmp};')
        for stmt in node.body:
            self._emit_stmt(stmt)
        self._indent -= 1
        self._emit('}')

    def _gen_assert(self, node: IRAssert):
        cond = self._gen_expr(node.condition)
        if node.message:
            msg = self._gen_expr(node.message)
            self._emit(f'if (!triad_is_y({cond})) {{ TriadString *_am = triad_value_to_string({msg}); fprintf(stderr, "assertion failed: %s\\n", _am->data); exit(1); }}')
        else:
            self._emit(f'if (!triad_is_y({cond})) {{ fprintf(stderr, "assertion failed\\n"); exit(1); }}')

    def _gen_del(self, node: IRDel):
        if isinstance(node.target, IRIdent):
            self._emit(f'{self._sanitize(node.target.name)} = TRIAD_NONE_VAL;')
        elif isinstance(node.target, IRIndex):
            obj = self._gen_expr(node.target.obj)
            idx = self._gen_expr(node.target.index)
            self._emit(f'(void)_triad_delindex_any(({obj}), ({idx}));')
        elif isinstance(node.target, IRField):
            obj = self._gen_expr(node.target.obj)
            self._emit(f'(void)_triad_delattr_any(({obj}), "{node.target.field}");')

    def _gen_sequence(self, node: IRSequence):
        inputs = self._gen_expr(node.inputs) if node.inputs else 'TRIAD_NONE_VAL'
        each = self._gen_expr(node.each_for) if node.each_for else 'TRIAD_NONE_VAL'
        self._emit(f'(void){inputs};')
        self._emit(f'(void){each};')
        self._emit('(void)0;')

    def _gen_try_catch(self, node: IRTryCatch):
        self._emit('TRIAD_TRY_BEGIN {')
        self._indent += 1
        for stmt in node.body:
            self._emit_stmt(stmt)
        self._indent -= 1
        if node.catches:
            for cb in node.catches:
                if cb.exceptions:
                    exc_str = ', '.join(cb.exceptions) if len(cb.exceptions) > 1 else cb.exceptions[0]
                else:
                    exc_str = 'Exception'
                var = self._sanitize(cb.var) if cb.var else '_exc'
                self._emit(f'}} TRIAD_CATCH_NAMED({var}, {exc_str}) {{')
                self._indent += 1
                for stmt in cb.body:
                    self._emit_stmt(stmt)
                self._indent -= 1
        elif node.catch_body:
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
            if node.name in self._top_level_fns and node.name not in self._classes:

                return (f'(TriadValue){{.tag = TRIAD_CLOSURE, .as = {{.cval = '
                        f'triad_closure_new({self._c_fn_name(node.name)}, 0)}}}}')
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
        if isinstance(node, IRComplex):
            return self._gen_complex(node)
        if isinstance(node, IRBytes):
            return self._gen_bytes(node)
        if isinstance(node, IRTuple):
            return self._gen_tuple(node)
        if isinstance(node, IRSet):
            return self._gen_set(node)
        if isinstance(node, IRTernary):
            return self._gen_ternary(node)
        if isinstance(node, IRChainCmp):
            return self._gen_chain_cmp(node)
        if isinstance(node, IRSuper):
            return self._gen_super(node)
        if isinstance(node, IRDictComp):
            return self._gen_dictcomp(node)
        if isinstance(node, IRSetComp):
            return self._gen_setcomp(node)
        if isinstance(node, IRGenComp):
            return self._gen_gencomp(node)
        if isinstance(node, IRYieldExpr):
            return self._gen_yield_expr(node)
        if isinstance(node, IRAwait):
            return self._gen_await(node)
        if isinstance(node, IRNullish):
            return self._gen_nullish(node)
        if isinstance(node, IRElvis):
            return self._gen_elvis(node)
        if isinstance(node, IROptChain):
            return self._gen_optchain(node)
        if isinstance(node, IRRegex):
            return self._gen_regex(node)
        if isinstance(node, IRCompoundAssign):
            return self._gen_compound_assign(node)
        if isinstance(node, IRLambda):
            return self._gen_lambda(node)
        return 'TRIAD_NONE_VAL'

    def _gen_binop(self, node: IRBinOp) -> str:
        left = self._gen_expr(node.left)
        right = self._gen_expr(node.right)
        op = node.op
        if op in ('+', '-', '*', '/', '%', '**', '@', '//'):
            return self._gen_arith_binop(left, right, op)
        if op in ('==', '!=', '<', '>', '<=', '>='):
            return self._gen_cmp_binop(left, right, op)
        if op in ('and', 'or'):
            l = self._gen_expr(node.left)
            r = self._gen_expr(node.right)
            if op == 'and':
                return f'(triad_is_y({l}) ? ({r}) : ({l}))'
            return f'(triad_is_y({l}) ? ({l}) : ({r}))'
        if op == 'in':
            return f'_triad_contains(({left}), ({right}))'
        if op == 'not_in':
            return f'TRIAD_BOOL(!triad_is_y(_triad_contains(({left}), ({right}))))'
        if op == 'is':
            return f'_triad_identity_eq(({left}), ({right}))'
        if op == 'is_not':
            return f'TRIAD_BOOL(!triad_is_y(_triad_identity_eq(({left}), ({right}))))'
        if op in ('&', '|', '^', '<<', '>>'):
            bop = {'&': '&', '|': '|', '^': '^', '<<': '<', '>>': '>'}[op]
            return f"_triad_bitwise('{bop}', ({left}), ({right}))"
        return 'TRIAD_NONE_VAL'

    def _gen_arith_binop(self, left: str, right: str, op: str) -> str:
        opc = {'+': '+', '-': '-', '*': '*', '/': '/', '%': '%', '**': 'p',
               '@': 'm', '//': 'f'}[op]
        return f"_triad_binop('{opc}', ({left}), ({right}))"

    def _gen_cmp_binop(self, left: str, right: str, op: str) -> str:
        opc = {'==': '=', '!=': '!', '<': '<', '>': '>', '<=': 'l', '>=': 'g'}[op]
        return f"_triad_cmp('{opc}', ({left}), ({right}))"

    def _gen_unaryop(self, node: IRUnaryOp) -> str:
        operand = self._gen_expr(node.operand)
        if node.op == '-':
            return f'(({{ TriadValue _u = ({operand}); (_u.tag == TRIAD_INT) ? TRIAD_INT(-_u.as.ival) : TRIAD_FLOAT(-_u.as.fval); }}))'
        if node.op == 'not':
            return f'TRIAD_BOOL(!triad_is_y({operand}))'
        if node.op == '~':
            return f'_triad_bitwise_not(({operand}))'
        return operand

    def _gen_nullish(self, node) -> str:
        l = self._gen_expr(node.left)
        r = self._gen_expr(node.right)
        return f'({{ TriadValue _t = ({l}); _t.tag == TRIAD_NONE ? ({r}) : _t; }})'

    def _gen_elvis(self, node) -> str:
        c = self._gen_expr(node.cond)
        e = self._gen_expr(node.else_val)
        return f'({{ TriadValue _c = ({c}); triad_is_y(_c) ? _c : ({e}); }})'

    def _gen_optchain(self, node) -> str:
        o = self._gen_expr(node.obj)
        if node.kind == 'attr':
            access = f'_triad_field_any(_o, "{node.name}")'
        elif node.kind == 'index':
            access = f'_triad_index_any(_o, ({self._gen_expr(node.index)}))'
        else:
            if node.kwargs:
                names = ', '.join(f'"{k}"' for k in node.kwargs)
                vals = ', '.join(self._gen_expr(v) for v in node.kwargs.values())
                kw_c = f'{len(node.kwargs)}, (const char*[]){{{names}}}, (TriadValue[]){{{vals}}}'
            else:
                kw_c = '0, NULL, NULL'
            if not any(isinstance(a, IRUnaryOp) and a.op == '*' for a in node.args):
                args = [self._gen_expr(a) for a in node.args]
                n = len(args)
                args_arr = ', '.join(args) if args else ''
                args_c = f'(TriadValue[]){{{args_arr}}}' if args else 'NULL'
                access = (f'({{ TriadValue _mo = _triad_field_any(_o, "{node.name}"); '
                          f'_triad_method_fallback(_mo, "{node.name}", {n}, {args_c}, {kw_c}); }})')
            else:
                av = self._fresh('_argv')
                an = self._fresh('_argc')
                ai = self._fresh('_ai')
                parts = []
                bounds = []
                for a in node.args:
                    if isinstance(a, IRUnaryOp) and a.op == '*':
                        sv = self._fresh('_sp')
                        parts.append(f'TriadValue {sv} = ({self._gen_expr(a.operand)});')
                        bounds.append(('sp', sv))
                    else:
                        bounds.append(('fx', self._gen_expr(a)))
                parts.append(f'int32_t {an} = 0;')
                for kind, ex in bounds:
                    if kind == 'sp':
                        parts.append(f'{an} += ({ex}.tag == TRIAD_LIST) ? triad_list_len({ex}.as.lval) : 1;')
                    else:
                        parts.append(f'{an}++;')
                parts.append(f'TriadValue *{av} = malloc((size_t){an} * sizeof(TriadValue));')
                parts.append(f'int32_t {ai} = 0;')
                for kind, ex in bounds:
                    if kind == 'sp':
                        parts.append(f'if ({ex}.tag == TRIAD_LIST) {{ for (int32_t _si = 0; _si < triad_list_len({ex}.as.lval); _si++) {av}[{ai}++] = triad_list_get({ex}.as.lval, _si); }} else {{ {av}[{ai}++] = {ex}; }}')
                    else:
                        parts.append(f'{av}[{ai}++] = ({ex});')
                parts.append(f'TriadValue _r = _triad_method_fallback(_triad_field_any(_o, "{node.name}"), "{node.name}", {an}, {av}, {kw_c});')
                parts.append(f'free({av});')
                parts.append('_r;')
                access = '({ ' + ' '.join(parts) + ' })'
        return f'({{ TriadValue _o = ({o}); _o.tag == TRIAD_NONE ? TRIAD_NONE_VAL : ({access}); }})'

    def _gen_compound_assign(self, node) -> str:
        t = node.target
        v = self._gen_expr(node.value)
        if node.op == '??=':
            new = f'({{ TriadValue _n = ({v}); _old.tag == TRIAD_NONE ? _n : _old; }})'
        elif node.op == '||=':
            new = f'(triad_is_y(_old) ? _old : ({v}))'
        else:
            new = f'(triad_is_y(_old) ? ({v}) : _old)'
        if isinstance(t, IRIndex):
            o = self._gen_expr(t.obj)
            k = self._gen_expr(t.index)
            return (f'({{ TriadValue _o = ({o}); TriadValue _k = ({k}); '
                    f'TriadValue _old = _triad_index_any(_o, _k); TriadValue _new = {new}; '
                    f'(void)_triad_setindex_any(_o, _k, _new); _new; }})')
        if isinstance(t, IRField):
            o = self._gen_expr(t.obj)
            return (f'({{ TriadValue _o = ({o}); '
                    f'TriadValue _old = _triad_field_any(_o, "{t.field}"); TriadValue _new = {new}; '
                    f'(void)_triad_setattr_any(_o, "{t.field}", _new); _new; }})')
        return self._gen_expr(node.value)

    @staticmethod
    def _c_escape(s: str) -> str:
        return s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')

    def _gen_regex(self, node) -> str:
        pat = self._c_escape(node.pattern)
        fl = self._c_escape(node.flags or '')
        return f'((TriadValue){{.tag = TRIAD_REGEX, .as = {{.rxval = triad_regex_compile("{pat}", "{fl}")}}}})'

    def _gen_call(self, node: IRCall) -> str:
        if any(isinstance(a, IRUnaryOp) and a.op == '*' for a in node.args):
            return self._gen_spread_call(node)
        args = [self._gen_expr(a) for a in node.args]
        if isinstance(node.func, IRIdent):
            name = node.func.name
            if name == 'print':
                if not args:
                    return '({ printf("\\n"); TRIAD_NONE_VAL; })'
                args_arr = ', '.join(args)
                return f'(triad_print({len(args)}, (TriadValue[]){{{args_arr}}}), TRIAD_NONE_VAL)'
            if name == 'len':
                return f'_triad_len_any(({args[0]}))'
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
                return f'_triad_to_int(({args[0]}))'
            if name == 'float':
                return f'_triad_to_float(({args[0]}))'
            if name == 'str':
                return f'(TriadValue){{.tag = TRIAD_STRING, .as = {{.sval = triad_value_to_string({args[0]})}}}}'
            if name == 'type':
                return f'triad_type_name({args[0]})'
            if name == 'input':
                return f"triad_input({(args[0] if args else 'NULL')})"
            if name == 'exit':
                code = f'(int)(({args[0]}).tag == TRIAD_INT ? ({args[0]}).as.ival : 0)' if args else '0'
                return f'({{ exit({code}); TRIAD_NONE_VAL; }})'
            if name == 'cuda_available':
                return 'TRIAD_BOOL(false)'
            if name in ('max', 'min') and len(args) == 2:
                cmp = '>' if name == 'max' else '<'
                return (f'({{ TriadValue _ma = ({args[0]}), _mb = ({args[1]}); '
                        f'(_triad_as_dbl(_ma) {cmp} _triad_as_dbl(_mb)) ? _ma : _mb; }})')
            if name in ('max', 'min') and len(args) == 1:
                cmp = '>' if name == 'max' else '<'
                return (f'({{ TriadValue _ml_ = ({args[0]}); TriadValue _best = triad_list_get(_ml_.as.lval, 0); '
                        f'for (int32_t _mi = 1; _mi < triad_list_len(_ml_.as.lval); _mi++) {{ '
                        f'TriadValue _cur = triad_list_get(_ml_.as.lval, _mi); '
                        f'if (_triad_as_dbl(_cur) {cmp} _triad_as_dbl(_best)) _best = _cur; }} _best; }})')
            if name == 'set' and len(args) == 1:
                return (f'({{ TriadValue _su = ({args[0]}); TriadList *_sd = triad_list_new(); '
                        f'TriadValue _sit = _triad_iter_prep(_su); '
                        f'for (int32_t _si = 0; _si < triad_list_len(_sit.as.lval); _si++) {{ '
                        f'TriadValue _sv = triad_list_get(_sit.as.lval, _si); '
                        f'if (!triad_list_contains(_sd, _sv)) triad_list_push(_sd, _sv); }} '
                        f'(TriadValue){{.tag = TRIAD_LIST, .as = {{.lval = _sd}}}}; }})')
            if name == 'sorted' and len(args) == 1:
                return (f'({{ TriadValue _so = ({args[0]}); TriadList *_sc = triad_list_new(); '
                        f'TriadValue _sot = _triad_iter_prep(_so); '
                        f'for (int32_t _si = 0; _si < triad_list_len(_sot.as.lval); _si++) '
                        f'triad_list_push(_sc, triad_list_get(_sot.as.lval, _si)); '
                        f'triad_list_sort(_sc); '
                        f'(TriadValue){{.tag = TRIAD_LIST, .as = {{.lval = _sc}}}}; }})')
            if name == 'sum' and len(args) == 1:
                return (f'({{ TriadValue _sl = ({args[0]}); double _acc = 0.0; bool _allint = true; '
                        f'for (int32_t _si = 0; _si < triad_list_len(_sl.as.lval); _si++) {{ '
                        f'TriadValue _sv = triad_list_get(_sl.as.lval, _si); '
                        f'if (_sv.tag != TRIAD_INT) _allint = false; _acc += _triad_as_dbl(_sv); }} '
                        f'_allint ? TRIAD_INT((int64_t)_acc) : TRIAD_FLOAT(_acc); }})')
            if name == 'push':
                return f'(triad_list_push(({args[0]}).as.lval, {args[1]}), TRIAD_NONE_VAL)'
            if name == 'pop':
                return f'(triad_list_remove_at(({args[0]}).as.lval, triad_list_len(({args[0]}).as.lval)-1), TRIAD_NONE_VAL)'
            if name == 'keys':
                return f'(TriadValue){{.tag = TRIAD_LIST, .as = {{.lval = triad_dict_keys(({args[0]}).as.dval)}}}}'
            if name == 'has':
                return f'TRIAD_BOOL(triad_dict_has(({args[0]}).as.dval, ({args[1]}).as.sval))'
            if name in ('solver_solve', 'solve'):
                return f'_triad_solver_solve({args[0]})'
            if name == 'ccall':

                lib_expr = args[0] if len(args) > 0 else 'TRIAD_NONE_VAL'
                fn_expr = args[1] if len(args) > 1 else 'TRIAD_NONE_VAL'
                ccall_args = args[2:]
                if ccall_args:
                    args_arr = ', '.join(ccall_args)
                    return f'_tri_ccall({lib_expr}, {fn_expr}, {len(ccall_args)}, (TriadValue[]){{{args_arr}}})'
                return f'_tri_ccall({lib_expr}, {fn_expr}, 0, NULL)'
            if name == 'py_call':
                mod_expr = args[0] if len(args) > 0 else 'TRIAD_NONE_VAL'
                fn_expr = args[1] if len(args) > 1 else 'TRIAD_NONE_VAL'
                py_args = args[2:]
                if py_args:
                    args_arr = ', '.join(py_args)
                    return f'triad_py_call({mod_expr}, {fn_expr}, {len(py_args)}, (TriadValue[]){{{args_arr}}})'
                return f'triad_py_call({mod_expr}, {fn_expr}, 0, NULL)'
            if name == 'py_eval':
                return f'triad_py_eval({args[0]})' if args else 'TRIAD_NONE_VAL'
            if name == 'py_exec':
                return f'triad_py_exec({args[0]})' if args else 'TRIAD_NONE_VAL'
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
            if name == 'ml_triad':
                return f'_ml_triad_new({args[0]}, {args[1]})'
            if name == 'ml_embedding':
                return f'_ml_embedding_new({args[0]}, {args[1]})'
            if name == 'ml_embedding_forward':
                return f'_ml_embedding_forward({args[0]}, {args[1]})'
            if name == 'ml_layernorm':
                return f'_ml_layernorm_new({args[0]})'
            if name == 'ml_layernorm_forward':
                return f'_ml_layernorm_forward({args[0]}, {args[1]})'
            if name == 'ml_mha':
                return f'_ml_mha_new({args[0]}, {args[1]})'
            if name == 'ml_mha_forward':
                return f'_ml_mha_forward({args[0]}, {args[1]})'
            if name == 'ml_transformer':
                return f'_ml_transformer_new({args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]})'
            if name == 'ml_transformer_forward':
                return f'_ml_transformer_forward({args[0]}, {args[1]})'
            if name == 'ml_transformer_adam':
                return f'_ml_transformer_adam({args[0]}, {args[1]})'
            if name == 'ml_wave':
                return f'_ml_wave_new({args[0]}, {args[1]})'
            if name == 'ml_wave_forward':
                return f'_ml_wave_forward({args[0]}, {args[1]})'
            if name == 'ml_wave_adam':
                return f'_ml_wave_adam({args[0]}, {args[1]}, {args[2]})'
            if name == 'ml_forward':
                return f'_ml_triad_forward({args[0]}, {args[1]})'
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
                ctor_name, param_names = 'new', list(cls.fields)
                for method in cls.methods:
                    if method.name in ('new', '__init__'):
                        ctor_name = method.name
                        param_names = [p for p in method.params if p != 'self']
                        break
                merged = self._merge_kwargs_positional(args, node.kwargs, param_names)
                tmp = self._fresh('_obj')
                obj_init = f'triad_object_new_typed("{name}", triad_class_meta_lookup("{name}"))'
                parts_c = [f'({{ TriadValue {tmp} = {obj_init}; ']
                new_fn = f'_triad_cls_{name}_{ctor_name}'
                args_with_self = ', '.join([tmp] + merged)
                parts_c.append(f'{new_fn}({1 + len(merged)}, (TriadValue[]){{{args_with_self}}}); ')
                parts_c.append(f'{tmp}; }})')
                return ''.join(parts_c)
            if name in self._top_level_fns:
                fname = self._c_fn_name(name)
                star_i, kw_i = getattr(self, '_top_level_fn_meta', {}).get(name, (-1, -1))
                merged = self._merge_kwargs_positional(
                    args, node.kwargs, self._top_level_fn_params.get(name, []), kw_i)
                args_arr = ', '.join(merged)
                return f'{fname}({len(merged)}, (TriadValue[]){{{args_arr}}})'
            return self._gen_dynamic_call(self._sanitize(name), args, node.kwargs)
        func = self._gen_expr(node.func)
        return self._gen_dynamic_call(f'({func})', args, node.kwargs)

    def _gen_spread_call(self, node: IRCall) -> str:
        sp = self._fresh('_sp')
        stmts = [f'TriadList *{sp} = triad_list_new();']
        dict_spreads = []
        for a in node.args:
            if isinstance(a, IRUnaryOp) and a.op == '**':
                dict_spreads.append(a.operand)
            elif isinstance(a, IRUnaryOp) and a.op == '*':
                sv = self._fresh('_sv')
                stmts.append(f'TriadValue {sv} = {self._gen_expr(a.operand)};')
                stmts.append(f'if ({sv}.tag == TRIAD_LIST) {{ for (int32_t _si = 0; _si < triad_list_len({sv}.as.lval); _si++) triad_list_push({sp}, triad_list_get({sv}.as.lval, _si)); }} else {{ triad_list_push({sp}, {sv}); }}')
            else:
                stmts.append(f'triad_list_push({sp}, {self._gen_expr(a)});')
        if node.kwargs or dict_spreads:
            kd = self._fresh('_kd')
            stmts.append(f'TriadDict *{kd} = triad_dict_new();')
            for operand in dict_spreads:
                dv = self._fresh('_dv')
                stmts.append(f'TriadValue {dv} = {self._gen_expr(operand)};')
                stmts.append(f'if ({dv}.tag == TRIAD_DICT) {{ TriadList *_dl = triad_dict_keys({dv}.as.dval); for (int32_t _di = 0; _di < triad_list_len(_dl); _di++) {{ TriadString *_dk = triad_list_get(_dl, _di).as.sval; triad_dict_set({kd}, _dk, triad_dict_get({dv}.as.dval, _dk)); }} }}')
            for k, v in (node.kwargs or {}).items():
                stmts.append(f'triad_dict_set({kd}, triad_str_new("{k}"), {self._gen_expr(v)});')
            stmts.append(f'triad_list_push({sp}, (TriadValue){{.tag = TRIAD_DICT, .as = {{.dval = {kd}}}}});')
        body = ' '.join(stmts)
        if isinstance(node.func, IRIdent):
            name = node.func.name
            if name == 'print':
                return f'({{ {body} triad_print(triad_list_len({sp}), {sp}->items); TRIAD_NONE_VAL; }})'
            if name in self._top_level_fns:
                fname = self._c_fn_name(name)
                return f'({{ {body} {fname}(triad_list_len({sp}), {sp}->items); }})'
            return f'({{ {body} _triad_call_value({self._sanitize(name)}, triad_list_len({sp}), {sp}->items, 0, NULL, NULL); }})'
        func = self._gen_expr(node.func)
        return f'({{ {body} _triad_call_value(({func}), triad_list_len({sp}), {sp}->items, 0, NULL, NULL); }})'

    def _merge_kwargs_positional(self, args: list[str], kwargs: dict,
                                 param_names: list[str], kw_idx: int = -1) -> list[str]:
        if not kwargs:
            return args
        slots = list(args)
        kw_compiled = {k: self._gen_expr(v) for k, v in kwargs.items()}
        for pname in param_names[len(args):]:
            if pname in kw_compiled:
                slots.append(kw_compiled.pop(pname))
            elif kw_compiled:
                slots.append('TRIAD_NONE_VAL')
            else:
                break
        if kw_compiled and kw_idx >= 0:

            sets = ' '.join(f'triad_dict_set(_kd, triad_str_new("{k}"), {v});'
                            for k, v in kw_compiled.items())
            slots.append(f'({{ TriadDict *_kd = triad_dict_new(); {sets} '
                         f'(TriadValue){{.tag = TRIAD_DICT, .as = {{.dval = _kd}}}}; }})')
        else:
            for v in kw_compiled.values():
                slots.append(v)
        return slots

    def _gen_dynamic_call(self, fn_expr: str, args: list[str], kwargs: dict) -> str:
        n = len(args)
        args_arr = ', '.join(args) if args else ''
        args_c = f'(TriadValue[]){{{args_arr}}}' if args else 'NULL'
        if kwargs:
            names = ', '.join(f'"{k}"' for k in kwargs)
            vals = ', '.join(self._gen_expr(v) for v in kwargs.values())
            return (f'_triad_call_value({fn_expr}, {n}, {args_c}, '
                    f'{len(kwargs)}, (const char*[]){{{names}}}, (TriadValue[]){{{vals}}})')
        return f'_triad_call_value({fn_expr}, {n}, {args_c}, 0, NULL, NULL)'

    def _gen_method_call(self, node: IRMethodCall) -> str:
        m = node.method
        args = [self._gen_expr(a) for a in node.args]

        if isinstance(node.obj, IRIdent) and node.obj.name in self._native_mod_aliases:
            mod = self._native_mod_aliases[node.obj.name]
            def _dbl(a):
                return f'(({a}).tag == TRIAD_INT ? (double)({a}).as.ival : ({a}).as.fval)'
            if mod == 'math':
                if m in _MATH_FNS and len(args) == 1:
                    return f'TRIAD_FLOAT({_MATH_FNS[m]}({_dbl(args[0])}))'
                if m == 'pow' and len(args) == 2:
                    return f'TRIAD_FLOAT(triad_math_pow({_dbl(args[0])}, {_dbl(args[1])}))'
                if m == 'abs' and len(args) == 1:
                    return f'TRIAD_FLOAT(triad_math_abs({_dbl(args[0])}))'
                if m == 'min' and len(args) == 2:
                    return f'TRIAD_FLOAT(triad_math_min({_dbl(args[0])}, {_dbl(args[1])}))'
                if m == 'max' and len(args) == 2:
                    return f'TRIAD_FLOAT(triad_math_max({_dbl(args[0])}, {_dbl(args[1])}))'
                if m == 'clamp' and len(args) == 3:
                    return f'TRIAD_FLOAT(triad_math_clamp({_dbl(args[0])}, {_dbl(args[1])}, {_dbl(args[2])}))'
            if mod == 'random':
                if m == 'random' and not args:
                    return 'TRIAD_FLOAT(triad_random_double())'
                if m == 'uniform' and len(args) == 2:
                    return f'TRIAD_FLOAT(triad_random_uniform({_dbl(args[0])}, {_dbl(args[1])}))'
                if m == 'randint' and len(args) == 2:
                    return f'TRIAD_INT(triad_random_int(({args[0]}).as.ival, ({args[1]}).as.ival))'
                if m == 'seed' and len(args) == 1:
                    return f'(triad_random_seed(({args[0]}).as.ival), TRIAD_NONE_VAL)'
            raise CCompileError(f"native module '{mod}' has no member '{m}'")

        obj = self._gen_expr(node.obj)
        n = len(args)
        args_arr = ', '.join(args) if args else ''
        args_c = f'(TriadValue[]){{{args_arr}}}' if args else 'NULL'
        if node.kwargs:
            names = ', '.join(f'"{k}"' for k in node.kwargs)
            vals = ', '.join(self._gen_expr(v) for v in node.kwargs.values())
            kw_c = f'{len(node.kwargs)}, (const char*[]){{{names}}}, (TriadValue[]){{{vals}}}'
        else:
            kw_c = '0, NULL, NULL'
        fallback = f'_triad_method_fallback(_mo, "{m}", {n}, {args_c}, {kw_c})'
        if node.kwargs:

            return f'({{ TriadValue _mo = ({obj}); {fallback}; }})'
        static = self._gen_method_static('_mo', args, m)
        if static is None:
            return f'({{ TriadValue _mo = ({obj}); {fallback}; }})'
        return (f'({{ TriadValue _mo = ({obj}); '
                f'(_mo.tag == TRIAD_PYOBJ) ? {fallback} : ({static}); }})')

    def _gen_method_static(self, obj: str, args: list[str], m: str) -> str | None:
        if m == 'push':
            return f'(triad_list_push(({obj}).as.lval, {args[0]}), TRIAD_NONE_VAL)'
        if m == 'pop':
            return f'triad_list_remove_at(({obj}).as.lval, triad_list_len(({obj}).as.lval)-1)'
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
        if m == 'get':
            if len(args) == 1:
                return f'_triad_dict_get_default(({obj}), {args[0]}, TRIAD_NONE_VAL)'
            if len(args) == 2:
                return f'_triad_dict_get_default(({obj}), {args[0]}, {args[1]})'
        if m == 'sum':
            return f'TRIAD_FLOAT(triad_ndarray_sum(({obj}).as.aval))'
        if m == 'mean':
            return f'TRIAD_FLOAT(triad_ndarray_mean(({obj}).as.aval))'
        if m == 'reshape':
            return f'_triad_reshape_v(({obj}), {args[0]})'
        for cls_name, cls_node in self._classes.items():
            for method in cls_node.methods:
                if method.name == m:
                    cls_fn = f'_triad_cls_{cls_name}_{m}'
                    if args:
                        args_arr = ', '.join(args)
                        return f'{cls_fn}({1 + len(args)}, (TriadValue[]){{{obj}, {args_arr}}})'
                    return f'{cls_fn}(1, (TriadValue[]){{{obj}}})'
        return None

    def _gen_index(self, node: IRIndex) -> str:
        obj = self._gen_expr(node.obj)
        idx = self._gen_expr(node.index)
        return f'_triad_index_any(({obj}), ({idx}))'

    def _gen_slice(self, node: IRSlice) -> str:
        obj = self._gen_expr(node.obj)
        s = f'({self._gen_expr(node.start)}).as.ival' if node.start else '0'
        e = f'({self._gen_expr(node.end)}).as.ival' if node.end else '0'
        step = f'({self._gen_expr(node.step)}).as.ival' if node.step else '1'
        has_s = '1' if node.start else '0'
        has_e = '1' if node.end else '0'
        return f'_triad_slice_any(({obj}), {s}, {e}, {step}, {has_s}, {has_e})'

    def _gen_field(self, node: IRField) -> str:
        if isinstance(node.obj, IRIdent) and node.obj.name in self._native_mod_aliases:
            mod = self._native_mod_aliases[node.obj.name]
            if mod == 'math' and node.field in _MATH_CONSTS:
                return f'TRIAD_FLOAT({_MATH_CONSTS[node.field]})'
            raise CCompileError(f"native module '{mod}' has no constant '{node.field}'")
        obj = self._gen_expr(node.obj)
        if node.field == 'length' or node.field == 'len':
            return f'_triad_len_any(({obj}))'
        return f'_triad_field_any(({obj}), "{node.field}")'

    def _gen_list(self, node: IRList) -> str:
        if not any(isinstance(e, IRUnaryOp) and e.op == '*' for e in node.elements):
            elems = [self._gen_expr(e) for e in node.elements]
            if not elems:
                return '(TriadValue){.tag = TRIAD_LIST, .as = {.lval = triad_list_new()}}'
            elems_str = ', '.join(elems)
            return f'({{ TriadList *_tl = triad_list_new_cap({len(elems)}); TriadValue _ta[] = {{{elems_str}}}; for (int32_t _ti = 0; _ti < {len(elems)}; _ti++) triad_list_push(_tl, _ta[_ti]); (TriadValue){{.tag = TRIAD_LIST, .as = {{.lval = _tl}}}}; }})'
        parts = ['TriadList *_tl = triad_list_new();']
        for e in node.elements:
            if isinstance(e, IRUnaryOp) and e.op == '*':
                sv = self._fresh('_sp')
                parts.append(f'TriadValue {sv} = ({self._gen_expr(e.operand)});')
                parts.append(f'if ({sv}.tag == TRIAD_LIST) {{ for (int32_t _si = 0; _si < triad_list_len({sv}.as.lval); _si++) triad_list_push(_tl, triad_list_get({sv}.as.lval, _si)); }} else {{ triad_list_push(_tl, {sv}); }}')
            else:
                parts.append(f'triad_list_push(_tl, ({self._gen_expr(e)}));')
        parts.append('(TriadValue){.tag = TRIAD_LIST, .as = {.lval = _tl}};')
        return '({ ' + ' '.join(parts) + ' })'

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
        for part in node.parts:
            text, expr = part[0], part[1]
            spec = part[2] if len(part) > 2 else None
            if expr is not None:
                raw = self._gen_expr(expr)
                if spec:
                    sp = spec.replace('\\', '').replace('"', '')
                    parts.append(f'(TriadValue){{.tag = TRIAD_STRING, .as = {{.sval = _triad_format_spec({raw}, "{sp}")}}}}')
                else:
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
        if node.clauses and len(node.clauses) > 1:
            tmp_list = self._fresh('_lc')
            result = f'({{ TriadList *{tmp_list} = triad_list_new(); '
            loop_vars = []
            for clause in node.clauses:
                var = self._sanitize(clause.var)
                iter_expr = self._gen_expr(clause.iter)
                idx = self._fresh('_lci')
                loop_vars.append((var, iter_expr, idx, clause.conditions))
            for var, iter_expr, idx, conditions in loop_vars:
                result += f'TriadValue _lc_iter_{var} = _triad_iter_prep({iter_expr}); '
                result += f'if (_lc_iter_{var}.tag == TRIAD_LIST) {{ '
                result += f'for (int32_t {idx} = 0; {idx} < triad_list_len(_lc_iter_{var}.as.lval); {idx}++) {{ '
                result += f'TriadValue {var} = triad_list_get(_lc_iter_{var}.as.lval, {idx}); '
                for cond in conditions:
                    cond_expr = self._gen_expr(cond)
                    result += f'if (!triad_is_y({cond_expr})) continue; '
            result_expr = self._gen_expr(node.expr)
            result += f'triad_list_push({tmp_list}, {result_expr}); '
            for _ in loop_vars:
                result += '}} '
            result += f'(TriadValue){{.tag = TRIAD_LIST, .as = {{.lval = {tmp_list}}}}}; }})'
            return result
        tmp_list = self._fresh('_lc')
        iter_expr = self._gen_expr(node.iter)
        var = self._sanitize(node.var)
        idx = self._fresh('_lci')
        result_expr = self._gen_expr(node.expr)
        cond_expr = self._gen_expr(node.condition) if node.condition else None
        code = (f'({{ TriadList *{tmp_list} = triad_list_new(); '
                f'TriadValue _lc_iter = _triad_iter_prep({iter_expr}); '
                f'if (_lc_iter.tag == TRIAD_LIST) {{ '
                f'for (int32_t {idx} = 0; {idx} < triad_list_len(_lc_iter.as.lval); {idx}++) {{ '
                f'TriadValue {var} = triad_list_get(_lc_iter.as.lval, {idx}); ')
        if cond_expr:
            code += f'if (!triad_is_y({cond_expr})) continue; '
        code += f'triad_list_push({tmp_list}, {result_expr}); }} }} '
        code += f'(TriadValue){{.tag = TRIAD_LIST, .as = {{.lval = {tmp_list}}}}}; }})'
        return code

    def _gen_assign_expr(self, node: IRAssignExpr) -> str:
        val = self._gen_expr(node.value)
        if isinstance(node.target, IRIdent):
            name = self._sanitize(node.target.name)
            return f'({name} = {val})'
        return val

    def _gen_complex(self, node: IRComplex) -> str:
        r = node.real
        i = node.imag
        return f'((TriadValue){{.tag = TRIAD_COMPLEX, .as = {{.cvalp = triad_complex_new({r}, {i})}}}})'

    def _gen_bytes(self, node: IRBytes) -> str:
        raw = node.value
        hex_bytes = ', '.join(f'0x{b:02x}' for b in raw.encode('utf-8') if isinstance(raw, str))
        if not hex_bytes:
            hex_bytes = '0x00'
        n = len(raw.encode('utf-8')) if isinstance(raw, str) else 0
        return f'((TriadValue){{.tag = TRIAD_BYTES, .as = {{.bval = triad_bytes_new((uint8_t[]){{{hex_bytes}}}, {n})}}}})'

    def _gen_tuple(self, node: IRTuple) -> str:
        elems = [self._gen_expr(e) for e in node.elements]
        if not elems:
            return '(TriadValue){.tag = TRIAD_TUPLE, .as = {.tval = triad_tuple_new(0)}}'
        elems_str = ', '.join(elems)
        return f'({{ TriadTuple *_tt = triad_tuple_new({len(elems)}); TriadValue _ta[] = {{{elems_str}}}; for (int32_t _ti = 0; _ti < {len(elems)}; _ti++) triad_tuple_set(_tt, _ti, _ta[_ti]); (TriadValue){{.tag = TRIAD_TUPLE, .as = {{.tval = _tt}}}}; }})'

    def _gen_set(self, node: IRSet) -> str:
        elems = [self._gen_expr(e) for e in node.elements]
        if not elems:
            return '(TriadValue){.tag = TRIAD_SET, .as = {.sset = triad_set_new()}}'
        parts = []
        for e in elems:
            parts.append(f'triad_set_add(_ss, {e});')
        body = ' '.join(parts)
        return f'({{ TriadSet *_ss = triad_set_new(); {body} (TriadValue){{.tag = TRIAD_SET, .as = {{.sset = _ss}}}}; }})'

    def _gen_ternary(self, node: IRTernary) -> str:
        cond = self._gen_expr(node.condition)
        then_v = self._gen_expr(node.then_val)
        else_v = self._gen_expr(node.else_val)
        return f'(triad_is_y({cond}) ? ({then_v}) : ({else_v}))'

    def _gen_chain_cmp(self, node: IRChainCmp) -> str:
        _CMP_OPC = {'==': '=', '!=': '!', '<': '<', '>': '>', '<=': 'l', '>=': 'g'}
        if len(node.ops) == 1:
            return self._gen_expr(IRBinOp(node.ops[0], node.operands[0], node.operands[1]))
        tmps = []
        tmp_decls_list = []
        for i in range(len(node.operands)):
            t = self._fresh('_ch')
            tmps.append(t)
            tmp_decls_list.append(f'TriadValue {t} = {self._gen_expr(node.operands[i])};')
        tmp_decls = ' '.join(tmp_decls_list)
        cmp_parts = []
        for i, op in enumerate(node.ops):
            opc = _CMP_OPC[op]
            cmp_parts.append(f"triad_is_y(_triad_cmp('{opc}', {tmps[i]}, {tmps[i+1]}))")
        all_parts = ' && '.join(cmp_parts)
        return f'({{ {tmp_decls} TRIAD_BOOL({all_parts}); }})'

    def _gen_super(self, node: IRSuper) -> str:
        args = [self._gen_expr(a) for a in node.args]
        if args:
            args_arr = ', '.join(args)
            return f'_triad_super({len(args)}, (TriadValue[]){{{args_arr}}})'
        return '_triad_super(0, NULL)'

    def _gen_dictcomp(self, node: IRDictComp) -> str:
        tmp_dict = self._fresh('_dc')
        result = f'({{ TriadDict *{tmp_dict} = triad_dict_new(); '
        if node.clauses:
            loop_vars = []
            for clause in node.clauses:
                var = self._sanitize(clause.var)
                iter_expr = self._gen_expr(clause.iter)
                idx = self._fresh('_dci')
                loop_vars.append((var, iter_expr, idx, clause.conditions))
            depth = len(loop_vars)
            open_braces = []
            for var, iter_expr, idx, conditions in loop_vars:
                result += f'TriadValue _dc_iter_{var} = _triad_iter_prep({iter_expr}); '
                result += f'if (_dc_iter_{var}.tag == TRIAD_LIST) {{ '
                result += f'for (int32_t {idx} = 0; {idx} < triad_list_len(_dc_iter_{var}.as.lval); {idx}++) {{ '
                result += f'TriadValue {var} = triad_list_get(_dc_iter_{var}.as.lval, {idx}); '
                for cond in conditions:
                    result += f'if (!triad_is_y({self._gen_expr(cond)})) continue; '
                open_braces.append(2)
            key_expr = self._gen_expr(node.key_expr)
            val_expr = self._gen_expr(node.value_expr)
            result += f'triad_dict_set({tmp_dict}, ({key_expr}).as.sval, {val_expr}); '
            for _ in loop_vars:
                result += '}} '
        else:
            key_expr = self._gen_expr(node.key_expr)
            val_expr = self._gen_expr(node.value_expr)
            result += f'triad_dict_set({tmp_dict}, ({key_expr}).as.sval, {val_expr}); '
        result += f'(TriadValue){{.tag = TRIAD_DICT, .as = {{.dval = {tmp_dict}}}}}; }})'
        return result

    def _gen_setcomp(self, node: IRSetComp) -> str:
        tmp_set = self._fresh('_sc')
        result = f'({{ TriadSet *{tmp_set} = triad_set_new(); '
        if node.clauses:
            loop_vars = []
            for clause in node.clauses:
                var = self._sanitize(clause.var)
                iter_expr = self._gen_expr(clause.iter)
                idx = self._fresh('_sci')
                loop_vars.append((var, iter_expr, idx, clause.conditions))
            for var, iter_expr, idx, conditions in loop_vars:
                result += f'TriadValue _sc_iter_{var} = _triad_iter_prep({iter_expr}); '
                result += f'if (_sc_iter_{var}.tag == TRIAD_LIST) {{ '
                result += f'for (int32_t {idx} = 0; {idx} < triad_list_len(_sc_iter_{var}.as.lval); {idx}++) {{ '
                result += f'TriadValue {var} = triad_list_get(_sc_iter_{var}.as.lval, {idx}); '
                for cond in conditions:
                    result += f'if (!triad_is_y({self._gen_expr(cond)})) continue; '
            expr = self._gen_expr(node.expr)
            result += f'triad_set_add({tmp_set}, {expr}); '
            for _ in loop_vars:
                result += '}} '
        else:
            expr = self._gen_expr(node.expr)
            result += f'triad_set_add({tmp_set}, {expr}); '
        result += f'(TriadValue){{.tag = TRIAD_SET, .as = {{.sset = {tmp_set}}}}}; }})'
        return result

    def _gen_gencomp(self, node: IRGenComp) -> str:
        tmp_list = self._fresh('_gc')
        result = f'({{ TriadList *{tmp_list} = triad_list_new(); '
        if node.clauses:
            loop_vars = []
            for clause in node.clauses:
                var = self._sanitize(clause.var)
                iter_expr = self._gen_expr(clause.iter)
                idx = self._fresh('_gci')
                loop_vars.append((var, iter_expr, idx, clause.conditions))
            for var, iter_expr, idx, conditions in loop_vars:
                result += f'TriadValue _gc_iter_{var} = _triad_iter_prep({iter_expr}); '
                result += f'if (_gc_iter_{var}.tag == TRIAD_LIST) {{ '
                result += f'for (int32_t {idx} = 0; {idx} < triad_list_len(_gc_iter_{var}.as.lval); {idx}++) {{ '
                result += f'TriadValue {var} = triad_list_get(_gc_iter_{var}.as.lval, {idx}); '
                for cond in conditions:
                    result += f'if (!triad_is_y({self._gen_expr(cond)})) continue; '
            expr = self._gen_expr(node.expr)
            result += f'triad_list_push({tmp_list}, {expr}); '
            for _ in loop_vars:
                result += '}} '
        else:
            expr = self._gen_expr(node.expr)
            result += f'triad_list_push({tmp_list}, {expr}); '
        result += f'(TriadValue){{.tag = TRIAD_LIST, .as = {{.lval = {tmp_list}}}}}; }})'
        return result

    def _gen_yield_expr(self, node: IRYieldExpr) -> str:
        val = self._gen_expr(node.value) if node.value else 'TRIAD_NONE_VAL'
        return f'_triad_yield_value({val})'

    def _gen_await(self, node: IRAwait) -> str:
        val = self._gen_expr(node.value)
        return f'_triad_await_value({val})'

    @staticmethod
    def _sanitize(name: str) -> str:
        if name in ('int', 'float', 'char', 'long', 'short', 'double', 'void',
                    'static', 'const', 'unsigned', 'signed', 'for', 'while', 'if',
                    'else', 'switch', 'case', 'break', 'continue', 'return',
                    'struct', 'typedef', 'enum', 'union', 'sizeof', 'NULL', 'true',
                    'false', 'auto', 'register', 'extern', 'volatile', 'inline',
                    'default', 'do', 'goto',
                    'class', 'new', 'delete', 'this', 'template', 'virtual',
                    'friend', 'private', 'protected', 'public', 'namespace',
                    'using', 'try', 'catch', 'throw', 'operator', 'explicit',
                    'mutable', 'typename', 'bool', 'wchar_t',
                    'div', 'exit', 'time', 'abs', 'pow', 'sin', 'cos', 'tan',
                    'exp', 'log', 'log10', 'sqrt', 'floor', 'ceil', 'random',
                    'index', 'remove', 'free', 'main', 'signal', 'raise',
                    'clock', 'system', 'rename', 'clone', 'stdin', 'stdout',
                    'stderr', 'sleep', 'pause', 'kill', 'link', 'unlink',
                    'read', 'write', 'open', 'close', 'send', 'recv', 'gamma',
                    'y0', 'y1', 'j0', 'j1', 'round', 'trunc', 'matherr',
                    'self', 'None', 'async', 'await', 'yield', 'lambda',
                    'def', 'import', 'from', 'as', 'pass', 'assert', 'with'):
            return f'_triad_{name}'
        return name

    def _gen_map_destruct(self, node: IRMapDestruct) -> None:
        val_var = self._fresh('_md')
        val_expr = self._gen_expr(node.value)
        self._emit(f'TriadValue {val_var} = {val_expr};')
        for name in node.names:
            san = self._sanitize(name)
            self._emit(f'TriadValue {san} = triad_dict_get_str({val_var}, "{name}");')

    def _gen_match(self, node: IRMatch) -> None:
        subject_var = self._fresh('_m')
        subject_expr = self._gen_expr(node.subject)
        self._emit(f'TriadValue {subject_var} = {subject_expr};')
        first = True
        for case in node.cases:
            cond = self._gen_match_condition(subject_var, case.pattern)
            keyword = 'if' if first else 'else if'
            first = False
            self._emit(f'{keyword} ({cond}) {{')
            self._indent += 1
            self._gen_match_bindings(subject_var, case.pattern)
            for stmt in case.body:
                self._emit_stmt(stmt)
            self._indent -= 1
            self._emit('}')
        if node.else_body:
            self._emit('else {')
            self._indent += 1
            for stmt in node.else_body:
                self._emit_stmt(stmt)
            self._indent -= 1
            self._emit('}')

    def _gen_match_condition(self, subject_var: str, pattern: IRNode) -> str:
        if isinstance(pattern, IRInt):
            return f'triad_eq({subject_var}, TRIAD_INT({pattern.value}LL))'
        if isinstance(pattern, IRFloat):
            return f'triad_eq({subject_var}, TRIAD_FLOAT({pattern.value}))'
        if isinstance(pattern, IRBool):
            return f'triad_eq({subject_var}, TRIAD_BOOL({str(pattern.value).lower()}))'
        if isinstance(pattern, IRString):
            escaped = pattern.value.replace('\\', '\\\\').replace('"', '\\"')
            return f'triad_eq({subject_var}, (TriadValue){{.tag = TRIAD_STRING, .as = {{.sval = triad_str_new("{escaped}")}}}})'
        if isinstance(pattern, IRNone):
            return f'triad_is_none({subject_var})'
        if isinstance(pattern, IRIdent):
            if pattern.name == '_':
                return '1'
            return '1'
        if isinstance(pattern, IRComplex):
            return f'triad_eq({subject_var}, (TriadValue){{.tag = TRIAD_COMPLEX, .as = {{.cplx = {{ {pattern.real}, {pattern.imag} }}}}}})'
        if isinstance(pattern, IRBytes):
            return f'triad_eq({subject_var}, (TriadValue){{.tag = TRIAD_BYTES, .as = {{.sval = triad_str_new("{pattern.value}")}}}})'
        if isinstance(pattern, IRList):
            checks = [f'triad_is_list({subject_var})']
            for i, el in enumerate(pattern.elements):
                elem_var = f'triad_list_get({subject_var}, {i})'
                checks.append(self._gen_match_condition(elem_var, el))
            return ' && '.join(f'({c})' for c in checks)
        if isinstance(pattern, IRTuple):
            checks = [f'triad_is_tuple({subject_var})']
            for i, el in enumerate(pattern.elements):
                elem_var = f'triad_tuple_get({subject_var}, {i})'
                checks.append(self._gen_match_condition(elem_var, el))
            return ' && '.join(f'({c})' for c in checks)
        if isinstance(pattern, IRMap):
            checks = [f'triad_is_dict({subject_var})']
            for k, v in pattern.pairs:
                k_expr = self._gen_expr(k)
                has_key = f'triad_dict_has({subject_var}, {k_expr})'
                val_check = self._gen_match_condition(f'triad_dict_get({subject_var}, {k_expr})', v)
                checks.append(f'({has_key} && {val_check})')
            return ' && '.join(f'({c})' for c in checks)
        if isinstance(pattern, IRSet):
            checks = [f'triad_is_set({subject_var})']
            for el in pattern.elements:
                el_expr = self._gen_expr(el)
                checks.append(f'triad_set_contains({subject_var}, {el_expr})')
            return ' && '.join(f'({c})' for c in checks)
        return '1'

    def _gen_match_bindings(self, subject_var: str, pattern: IRNode) -> None:
        if isinstance(pattern, IRIdent):
            if pattern.name != '_':
                san = self._sanitize(pattern.name)
                self._emit(f'TriadValue {san} = {subject_var};')
        elif isinstance(pattern, IRList):
            for i, el in enumerate(pattern.elements):
                elem_var = f'triad_list_get({subject_var}, {i})'
                self._gen_match_bindings(elem_var, el)
        elif isinstance(pattern, IRTuple):
            for i, el in enumerate(pattern.elements):
                elem_var = f'triad_tuple_get({subject_var}, {i})'
                self._gen_match_bindings(elem_var, el)
        elif isinstance(pattern, IRMap):
            for k, v in pattern.pairs:
                k_expr = self._gen_expr(k)
                val_var = f'triad_dict_get({subject_var}, {k_expr})'
                self._gen_match_bindings(val_var, v)

    def _gen_lambda(self, node: IRLambda) -> str:
        fn_id = self._fresh('_lambda')
        fn_name = f'triad_lambda_{fn_id}'
        self._in_function = True
        old_lines = self._lines
        old_indent = self._indent
        self._lines = []
        self._indent = 0
        params_str = ', '.join(f'TriadValue {self._sanitize(p)}' for p in node.params)
        self._emit(f'static TriadValue {fn_name}(int32_t _nargs, TriadValue *_args) {{')
        self._indent += 1
        for i, p in enumerate(node.params):
            san = self._sanitize(p)
            self._emit(f'TriadValue {san} = _args[{i}];')
        for stmt in node.body:
            self._emit_stmt(stmt)
        self._emit('return TRIAD_NONE_VAL;')
        self._indent -= 1
        self._emit('}')
        lambda_def = '\n'.join(self._lines)
        self._lines = old_lines
        self._indent = old_indent
        self._in_function = False
        if not hasattr(self, '_lambda_defs'):
            self._lambda_defs = []
        self._lambda_defs.append(lambda_def)
        return f'(TriadValue){{.tag = TRIAD_CLOSURE, .as = {{.cval = triad_closure_new({fn_name}, 0)}}}}'

def compile_to_c(source: str, filename: str='<triad>') -> str:
    from compiler.lower import lower_module
    from frontend.lexer_universal import tokenize
    from frontend.parser_universal import Parser
    tokens = tokenize(source, filename)
    parser = Parser(tokens, filename)
    mod = parser.parse_module()
    ir = lower_module(mod)
    gen = CCodeGen()
    return gen.generate(ir)

