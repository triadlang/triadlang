#ifndef TRIAD_PYTHON_H
#define TRIAD_PYTHON_H

#include "triad_rt.h"
#include <Python.h>

static PyObject *_tri_to_py(TriadValue v);
static TriadValue _py_to_tri(PyObject *obj);

static PyObject *_tri_closure_trampoline(PyObject *self, PyObject *args) {
    TriadClosure *c = (TriadClosure *)PyCapsule_GetPointer(self, "triad.closure");
    if (c == NULL) Py_RETURN_NONE;
    Py_ssize_t n = PyTuple_GET_SIZE(args);
    TriadValue *tv = (TriadValue *)malloc(sizeof(TriadValue) * (n > 0 ? n : 1));
    for (Py_ssize_t i = 0; i < n; i++)
        tv[i] = _py_to_tri(PyTuple_GET_ITEM(args, i));
    TriadValue r = triad_closure_call(c, (int32_t)n, tv);
    free(tv);
    return _tri_to_py(r);
}

static PyMethodDef _tri_closure_def = {
    "triad_fn", _tri_closure_trampoline, METH_VARARGS,
    "TriadLang function exposed to Python"
};

static PyObject *_tri_closure_to_pyfn(TriadClosure *c) {
    PyObject *cap = PyCapsule_New(c, "triad.closure", NULL);
    PyObject *fn = PyCFunction_New(&_tri_closure_def, cap);
    Py_DECREF(cap);
    return fn;
}

static PyObject *_tri_to_py(TriadValue v) {
    switch (v.tag) {
    case TRIAD_PYOBJ: {
        PyObject *o = (PyObject *)v.as.ptr;
        Py_INCREF(o);
        return o;
    }
    case TRIAD_CLOSURE:
        return _tri_closure_to_pyfn(v.as.cval);
    case TRIAD_NONE:
        Py_RETURN_NONE;
    case TRIAD_BOOL:
        return PyBool_FromLong(v.as.bval);
    case TRIAD_INT:
        return PyLong_FromLongLong((long long)v.as.ival);
    case TRIAD_FLOAT:
        return PyFloat_FromDouble(v.as.fval);
    case TRIAD_STRING:
        return PyUnicode_FromStringAndSize(v.as.sval->data, v.as.sval->len);
    case TRIAD_LIST: {
        TriadList *l = v.as.lval;
        PyObject *list = PyList_New(l->len);
        for (int32_t i = 0; i < l->len; i++) {
            PyList_SET_ITEM(list, i, _tri_to_py(l->items[i]));
        }
        return list;
    }
    case TRIAD_DICT: {

        TriadDict *d = v.as.dval;
        PyObject *dict = PyDict_New();
        for (int32_t i = 0; i < d->cap; i++) {
            TriadString *k = d->entries[i].key;
            if (k == NULL || k == (TriadString *)(intptr_t)-1)
                continue;
            PyObject *key = PyUnicode_FromStringAndSize(k->data, k->len);
            PyObject *val = _tri_to_py(d->entries[i].value);
            PyDict_SetItem(dict, key, val);
            Py_DECREF(key);
            Py_DECREF(val);
        }
        return dict;
    }
    default:
        Py_RETURN_NONE;
    }
}

static TriadValue _py_to_tri(PyObject *obj) {
    if (obj == NULL || obj == Py_None) {
        return TRIAD_NONE_VAL;
    }
    if (PyBool_Check(obj)) {
        return TRIAD_BOOL(obj == Py_True);
    }
    if (PyLong_Check(obj)) {
        long long val = PyLong_AsLongLong(obj);
        if (val == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            return TRIAD_FLOAT(PyLong_AsDouble(obj));
        }
        return TRIAD_INT((int64_t)val);
    }
    if (PyFloat_Check(obj)) {
        return TRIAD_FLOAT(PyFloat_AsDouble(obj));
    }
    if (PyUnicode_Check(obj)) {
        Py_ssize_t len;
        const char *data = PyUnicode_AsUTF8AndSize(obj, &len);
        if (!data) {
            PyErr_Clear();
            return TRIAD_NONE_VAL;
        }
        return (TriadValue){.tag = TRIAD_STRING, .as = {.sval = triad_str_new_len(data, (int32_t)len)}};
    }
    if (PyList_Check(obj)) {
        Py_ssize_t n = PyList_GET_SIZE(obj);
        TriadList *l = triad_list_new_cap((int32_t)n);
        for (Py_ssize_t i = 0; i < n; i++) {
            triad_list_push(l, _py_to_tri(PyList_GET_ITEM(obj, i)));
        }
        return (TriadValue){.tag = TRIAD_LIST, .as = {.lval = l}};
    }
    if (PyDict_Check(obj)) {
        TriadDict *d = triad_dict_new();
        PyObject *key, *val;
        Py_ssize_t pos = 0;
        while (PyDict_Next(obj, &pos, &key, &val)) {
            TriadValue tk = _py_to_tri(key);
            if (tk.tag == TRIAD_STRING) {
                triad_dict_set(d, tk.as.sval, _py_to_tri(val));
            }
        }
        return (TriadValue){.tag = TRIAD_DICT, .as = {.dval = d}};
    }

    Py_INCREF(obj);
    return TRIAD_PYOBJ_VAL(obj);
}

static int _tri_py_initialized = 0;

static TriadString *_tri_pyobj_str(void *pyobj) {
    PyGILState_STATE g = PyGILState_Ensure();
    PyObject *s = PyObject_Str((PyObject *)pyobj);
    TriadString *out;
    if (s && PyUnicode_Check(s)) {
        Py_ssize_t len;
        const char *data = PyUnicode_AsUTF8AndSize(s, &len);
        out = triad_str_new_len(data, (int32_t)len);
    } else {
        out = triad_str_new("<pyobj>");
    }
    Py_XDECREF(s);
    PyGILState_Release(g);
    return out;
}

static bool _tri_pyobj_y(void *pyobj) {
    PyGILState_STATE g = PyGILState_Ensure();
    int t = PyObject_IsTrue((PyObject *)pyobj);
    PyGILState_Release(g);
    return t > 0;
}

static void _tri_py_ensure_init(void) {
    if (!_tri_py_initialized) {
        Py_Initialize();

        {
            PyGILState_STATE g = PyGILState_Ensure();
            PyObject *sys_path = PySys_GetObject("path");
            const char *src = getenv("TRIAD_SRC");
#ifdef TRIAD_SRC_DIR
            if (src == NULL) src = TRIAD_SRC_DIR;
#endif
            if (sys_path != NULL && src != NULL) {
                PyObject *p = PyUnicode_FromString(src);
                if (p) { PyList_Insert(sys_path, 0, p); Py_DECREF(p); }
            }
#ifdef TRIAD_SCRIPT_DIR
            if (sys_path != NULL) {
                PyObject *p = PyUnicode_FromString(TRIAD_SCRIPT_DIR);
                if (p) { PyList_Insert(sys_path, 0, p); Py_DECREF(p); }
            }
#endif
#ifdef TRIAD_SITE_DIR

            if (sys_path != NULL) {
                PyObject *p = PyUnicode_FromString(TRIAD_SITE_DIR);
                if (p) { PyList_Insert(sys_path, 0, p); Py_DECREF(p); }
            }
#endif
            PyGILState_Release(g);
        }
        triad_pyobj_str_hook = _tri_pyobj_str;
        triad_pyobj_y_hook = _tri_pyobj_y;
        _tri_py_initialized = 1;
    }
}

static TriadValue triad_py_import(const char *dotted, const char *script_dir) {
    _tri_py_ensure_init();
    PyGILState_STATE g = PyGILState_Ensure();
    PyObject *mod = NULL;

    PyObject *bridge = PyImport_ImportModule("runtime.compiler_runtime");
    if (bridge != NULL) {
        PyObject *fn = PyObject_GetAttrString(bridge, "_tri_import");
        Py_DECREF(bridge);
        if (fn != NULL) {
            PyObject *parts = PyUnicode_FromString(dotted);
            PyObject *dot = PyUnicode_FromString(".");
            PyObject *plist = PyUnicode_Split(parts, dot, -1);
            Py_DECREF(dot);
            Py_DECREF(parts);
            PyObject *paths = PyList_New(0);
            if (script_dir != NULL) {
                PyObject *sd = PyUnicode_FromString(script_dir);
                PyList_Append(paths, sd);
                Py_DECREF(sd);
            }
            PyObject *cwd = PyUnicode_FromString(".");
            PyList_Append(paths, cwd);
            Py_DECREF(cwd);
            mod = PyObject_CallFunctionObjArgs(fn, plist, paths, NULL);
            Py_DECREF(fn);
            Py_XDECREF(plist);
            Py_DECREF(paths);
            if (mod == NULL) PyErr_Clear();
        } else {
            PyErr_Clear();
        }
    } else {
        PyErr_Clear();
    }

    if (mod == NULL) {
        mod = PyImport_ImportModule(dotted);
    }

    int _is_leaf = 0;
    if (mod != NULL && strchr(dotted, '.') != NULL) {
        PyObject *nm = PyObject_GetAttrString(mod, "__name__");
        if (nm == NULL) {
            PyErr_Clear();
            nm = PyObject_GetAttrString(mod, "_name");
        }
        if (nm != NULL && PyUnicode_Check(nm)) {
            const char *ns = PyUnicode_AsUTF8(nm);
            if (ns != NULL && strcmp(ns, dotted) == 0) _is_leaf = 1;
        }
        Py_XDECREF(nm);
        PyErr_Clear();
    }
    if (mod != NULL && !_is_leaf && strchr(dotted, '.') != NULL) {
        const char *p = strchr(dotted, '.');
        while (p != NULL && mod != NULL) {
            const char *q = strchr(p + 1, '.');
            char part[128];
            size_t n = q ? (size_t)(q - p - 1) : strlen(p + 1);
            if (n >= sizeof(part)) break;
            memcpy(part, p + 1, n);
            part[n] = '\0';
            if (PyObject_HasAttrString(mod, part)) {
                PyObject *next = PyObject_GetAttrString(mod, part);
                Py_DECREF(mod);
                mod = next;
            } else {
                break;
            }
            p = q;
        }
        if (PyErr_Occurred()) PyErr_Clear();
    }
    TriadValue out;
    if (mod == NULL) {
        fprintf(stderr, "import: cannot resolve '%s'\n", dotted);
        if (PyErr_Occurred()) PyErr_Print();
        out = TRIAD_NONE_VAL;
    } else {
        out = TRIAD_PYOBJ_VAL(mod);
    }
    PyGILState_Release(g);
    return out;
}

static TriadValue triad_py_getattr(TriadValue obj, const char *name) {
    _tri_py_ensure_init();
    PyGILState_STATE g = PyGILState_Ensure();
    PyObject *o = _tri_to_py(obj);
    PyObject *attr = PyObject_GetAttrString(o, name);
    Py_DECREF(o);
    TriadValue out;
    if (attr == NULL) {
        fprintf(stderr, "getattr: no attribute '%s'\n", name);
        if (PyErr_Occurred()) PyErr_Print();
        out = TRIAD_NONE_VAL;
    } else {
        out = _py_to_tri(attr);
        Py_DECREF(attr);
    }
    PyGILState_Release(g);
    return out;
}

static TriadValue triad_py_setattr(TriadValue obj, const char *name, TriadValue val) {
    _tri_py_ensure_init();
    PyGILState_STATE g = PyGILState_Ensure();
    PyObject *o = _tri_to_py(obj);
    PyObject *v = _tri_to_py(val);
    if (PyObject_SetAttrString(o, name, v) < 0 && PyErr_Occurred()) PyErr_Print();
    Py_DECREF(o);
    Py_DECREF(v);
    PyGILState_Release(g);
    return TRIAD_NONE_VAL;
}

static TriadValue _triad_py_call_core(PyObject *callable,
                                      int32_t nargs, TriadValue *args,
                                      int32_t nkw, const char **kwnames,
                                      TriadValue *kwvals) {
    PyObject *py_args = PyTuple_New(nargs);
    for (int32_t i = 0; i < nargs; i++)
        PyTuple_SET_ITEM(py_args, i, _tri_to_py(args[i]));
    PyObject *py_kw = NULL;
    if (nkw > 0) {
        py_kw = PyDict_New();
        for (int32_t i = 0; i < nkw; i++) {
            PyObject *v = _tri_to_py(kwvals[i]);
            PyDict_SetItemString(py_kw, kwnames[i], v);
            Py_DECREF(v);
        }
    }
    PyObject *result = PyObject_Call(callable, py_args, py_kw);
    Py_DECREF(py_args);
    Py_XDECREF(py_kw);
    TriadValue out;
    if (result == NULL) {
        if (PyErr_Occurred()) PyErr_Print();
        out = TRIAD_NONE_VAL;
    } else {
        out = _py_to_tri(result);
        Py_DECREF(result);
    }
    return out;
}

static TriadValue triad_py_method_kw(TriadValue obj, const char *name,
                                     int32_t nargs, TriadValue *args,
                                     int32_t nkw, const char **kwnames,
                                     TriadValue *kwvals) {
    _tri_py_ensure_init();
    PyGILState_STATE g = PyGILState_Ensure();
    PyObject *o = _tri_to_py(obj);
    PyObject *meth = PyObject_GetAttrString(o, name);
    Py_DECREF(o);
    TriadValue out;
    if (meth == NULL) {
        fprintf(stderr, "method: no attribute '%s'\n", name);
        if (PyErr_Occurred()) PyErr_Print();
        out = TRIAD_NONE_VAL;
    } else {
        out = _triad_py_call_core(meth, nargs, args, nkw, kwnames, kwvals);
        Py_DECREF(meth);
    }
    PyGILState_Release(g);
    return out;
}

static TriadValue triad_py_call_value(TriadValue fn,
                                      int32_t nargs, TriadValue *args,
                                      int32_t nkw, const char **kwnames,
                                      TriadValue *kwvals) {
    _tri_py_ensure_init();
    PyGILState_STATE g = PyGILState_Ensure();
    PyObject *o = _tri_to_py(fn);
    TriadValue out = _triad_py_call_core(o, nargs, args, nkw, kwnames, kwvals);
    Py_DECREF(o);
    PyGILState_Release(g);
    return out;
}

static TriadValue triad_py_getslice(TriadValue obj, int64_t start, int64_t end,
                                    int64_t step, int has_start, int has_end) {
    _tri_py_ensure_init();
    PyGILState_STATE g = PyGILState_Ensure();
    PyObject *o = _tri_to_py(obj);
    PyObject *ps = has_start ? PyLong_FromLongLong(start) : (Py_INCREF(Py_None), Py_None);
    PyObject *pe = has_end ? PyLong_FromLongLong(end) : (Py_INCREF(Py_None), Py_None);
    PyObject *pp = PyLong_FromLongLong(step);
    PyObject *sl = PySlice_New(ps, pe, pp);
    Py_DECREF(ps); Py_DECREF(pe); Py_DECREF(pp);
    PyObject *r = PyObject_GetItem(o, sl);
    Py_DECREF(o); Py_DECREF(sl);
    TriadValue out;
    if (r == NULL) {
        if (PyErr_Occurred()) PyErr_Print();
        out = TRIAD_NONE_VAL;
    } else {
        out = _py_to_tri(r);
        Py_DECREF(r);
    }
    PyGILState_Release(g);
    return out;
}

static TriadValue triad_py_getitem(TriadValue obj, TriadValue idx) {
    _tri_py_ensure_init();
    PyGILState_STATE g = PyGILState_Ensure();
    PyObject *o = _tri_to_py(obj);
    PyObject *k = _tri_to_py(idx);
    PyObject *r = PyObject_GetItem(o, k);
    Py_DECREF(o);
    Py_DECREF(k);
    TriadValue out;
    if (r == NULL) {
        if (PyErr_Occurred()) PyErr_Print();
        out = TRIAD_NONE_VAL;
    } else {
        out = _py_to_tri(r);
        Py_DECREF(r);
    }
    PyGILState_Release(g);
    return out;
}

static TriadValue triad_py_setitem(TriadValue obj, TriadValue idx, TriadValue val) {
    _tri_py_ensure_init();
    PyGILState_STATE g = PyGILState_Ensure();
    PyObject *o = _tri_to_py(obj);
    PyObject *k = _tri_to_py(idx);
    PyObject *v = _tri_to_py(val);
    if (PyObject_SetItem(o, k, v) < 0 && PyErr_Occurred()) PyErr_Print();
    Py_DECREF(o); Py_DECREF(k); Py_DECREF(v);
    PyGILState_Release(g);
    return TRIAD_NONE_VAL;
}

static TriadValue triad_py_len(TriadValue obj) {
    _tri_py_ensure_init();
    PyGILState_STATE g = PyGILState_Ensure();
    PyObject *o = _tri_to_py(obj);
    Py_ssize_t n = PyObject_Length(o);
    Py_DECREF(o);
    if (n < 0) { PyErr_Clear(); n = 0; }
    PyGILState_Release(g);
    return TRIAD_INT((int64_t)n);
}

static TriadValue triad_py_to_list(TriadValue obj) {
    _tri_py_ensure_init();
    PyGILState_STATE g = PyGILState_Ensure();
    PyObject *o = _tri_to_py(obj);
    PyObject *seq = PySequence_List(o);
    Py_DECREF(o);
    TriadValue out;
    if (seq == NULL) {
        if (PyErr_Occurred()) PyErr_Print();
        out = (TriadValue){.tag = TRIAD_LIST, .as = {.lval = triad_list_new()}};
    } else {
        out = _py_to_tri(seq);
        Py_DECREF(seq);
    }
    PyGILState_Release(g);
    return out;
}

static TriadValue triad_py_binop(char op, TriadValue l, TriadValue r) {
    _tri_py_ensure_init();
    PyGILState_STATE g = PyGILState_Ensure();
    PyObject *a = _tri_to_py(l);
    PyObject *b = _tri_to_py(r);
    PyObject *res = NULL;
    switch (op) {
    case '+': res = PyNumber_Add(a, b); break;
    case '-': res = PyNumber_Subtract(a, b); break;
    case '*': res = PyNumber_Multiply(a, b); break;
    case '/': res = PyNumber_TrueDivide(a, b); break;
    case '%': res = PyNumber_Remainder(a, b); break;
    case 'p': res = PyNumber_Power(a, b, Py_None); break;
    case 'm': res = PyNumber_MatrixMultiply(a, b); break;
    case 'f': res = PyNumber_FloorDivide(a, b); break;
    }
    Py_DECREF(a);
    Py_DECREF(b);
    TriadValue out;
    if (res == NULL) {
        if (PyErr_Occurred()) PyErr_Print();
        out = TRIAD_NONE_VAL;
    } else {
        out = _py_to_tri(res);
        Py_DECREF(res);
    }
    PyGILState_Release(g);
    return out;
}

static TriadValue triad_py_cmp(char op, TriadValue l, TriadValue r) {
    _tri_py_ensure_init();
    PyGILState_STATE g = PyGILState_Ensure();
    PyObject *a = _tri_to_py(l);
    PyObject *b = _tri_to_py(r);
    int opid = (op == '<') ? Py_LT : (op == '>') ? Py_GT :
               (op == 'l') ? Py_LE : (op == 'g') ? Py_GE :
               (op == '=') ? Py_EQ : Py_NE;
    PyObject *res = PyObject_RichCompare(a, b, opid);
    Py_DECREF(a);
    Py_DECREF(b);
    TriadValue out;
    if (res == NULL) {
        PyErr_Clear();
        out = TRIAD_BOOL(false);
    } else {
        out = _py_to_tri(res);
        Py_DECREF(res);
    }
    PyGILState_Release(g);
    return out;
}

static TriadValue triad_py_contains(TriadValue item, TriadValue container) {
    _tri_py_ensure_init();
    PyGILState_STATE g = PyGILState_Ensure();
    PyObject *c = _tri_to_py(container);
    PyObject *i = _tri_to_py(item);
    int res = PySequence_Contains(c, i);
    if (res < 0) { PyErr_Clear(); res = 0; }
    Py_DECREF(c);
    Py_DECREF(i);
    PyGILState_Release(g);
    return TRIAD_BOOL(res != 0);
}

static double triad_py_to_double(TriadValue obj) {
    _tri_py_ensure_init();
    PyGILState_STATE g = PyGILState_Ensure();
    PyObject *o = _tri_to_py(obj);
    PyObject *f = PyNumber_Float(o);
    Py_DECREF(o);
    double d = 0.0;
    if (f != NULL) {
        d = PyFloat_AsDouble(f);
        Py_DECREF(f);
    }
    if (PyErr_Occurred()) PyErr_Clear();
    PyGILState_Release(g);
    return d;
}

static int64_t triad_py_to_int(TriadValue obj) {
    _tri_py_ensure_init();
    PyGILState_STATE g = PyGILState_Ensure();
    PyObject *o = _tri_to_py(obj);
    PyObject *l = PyNumber_Long(o);
    Py_DECREF(o);
    int64_t r = 0;
    if (l != NULL) {
        r = (int64_t)PyLong_AsLongLong(l);
        Py_DECREF(l);
    }
    if (PyErr_Occurred()) PyErr_Clear();
    PyGILState_Release(g);
    return r;
}

static bool triad_py_y(TriadValue obj) {
    _tri_py_ensure_init();
    PyGILState_STATE g = PyGILState_Ensure();
    PyObject *o = _tri_to_py(obj);
    int t = PyObject_IsTrue(o);
    Py_DECREF(o);
    PyGILState_Release(g);
    return t > 0;
}

static TriadValue triad_py_call(TriadValue mod_v, TriadValue fn_v,
                                 int32_t nargs, TriadValue *args) {
    _tri_py_ensure_init();

    if (mod_v.tag != TRIAD_STRING || fn_v.tag != TRIAD_STRING) {
        fprintf(stderr, "py_call: module and function must be strings\n");
        return TRIAD_NONE_VAL;
    }

    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject *mod = PyImport_ImportModule(mod_v.as.sval->data);
    if (!mod) {
        fprintf(stderr, "py_call: cannot import module '%s'\n", mod_v.as.sval->data);
        if (PyErr_Occurred()) { PyErr_Print(); }
        PyGILState_Release(gstate);
        return TRIAD_NONE_VAL;
    }

    PyObject *func = PyObject_GetAttrString(mod, fn_v.as.sval->data);
    Py_DECREF(mod);
    if (!func || !PyCallable_Check(func)) {
        fprintf(stderr, "py_call: '%s' is not callable in '%s'\n",
                fn_v.as.sval->data, mod_v.as.sval->data);
        Py_XDECREF(func);
        if (PyErr_Occurred()) { PyErr_Print(); }
        PyGILState_Release(gstate);
        return TRIAD_NONE_VAL;
    }

    PyObject *py_args = PyTuple_New(nargs);
    for (int32_t i = 0; i < nargs; i++) {
        PyTuple_SET_ITEM(py_args, i, _tri_to_py(args[i]));
    }

    PyObject *result = PyObject_CallObject(func, py_args);
    Py_DECREF(func);
    Py_DECREF(py_args);

    TriadValue tv;
    if (!result) {
        if (PyErr_Occurred()) { PyErr_Print(); }
        tv = TRIAD_NONE_VAL;
    } else {
        tv = _py_to_tri(result);
        Py_DECREF(result);
    }

    PyGILState_Release(gstate);
    return tv;
}

static TriadValue triad_py_eval(TriadValue code_v) {
    _tri_py_ensure_init();

    if (code_v.tag != TRIAD_STRING) {
        fprintf(stderr, "py_eval: code must be a string\n");
        return TRIAD_NONE_VAL;
    }

    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject *result = PyRun_StringFlags(code_v.as.sval->data,
                                          Py_eval_input,
                                          PyEval_GetBuiltins(),
                                          PyEval_GetBuiltins(),
                                          NULL);

    TriadValue tv;
    if (!result) {
        if (PyErr_Occurred()) { PyErr_Print(); }
        tv = TRIAD_NONE_VAL;
    } else {
        tv = _py_to_tri(result);
        Py_DECREF(result);
    }

    PyGILState_Release(gstate);
    return tv;
}

static TriadValue triad_py_exec(TriadValue code_v) {
    _tri_py_ensure_init();

    if (code_v.tag != TRIAD_STRING) {
        fprintf(stderr, "py_exec: code must be a string\n");
        return TRIAD_NONE_VAL;
    }

    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject *result = PyRun_StringFlags(code_v.as.sval->data,
                                          Py_file_input,
                                          PyEval_GetBuiltins(),
                                          PyEval_GetBuiltins(),
                                          NULL);

    if (!result && PyErr_Occurred()) {
        PyErr_Print();
    }
    Py_XDECREF(result);

    PyGILState_Release(gstate);
    return TRIAD_NONE_VAL;
}

#endif
