#ifndef TRIAD_PYTHON_H
#define TRIAD_PYTHON_H

/*
 * TriadLang Python embedding layer.
 *
 * Provides bidirectional interop:
 *   triad_py_call(module, func, args...) -> TriadValue
 *   triad_py_eval(code) -> TriadValue
 *
 * Uses CPython C API. The interpreter is initialized lazily on first call.
 * Thread safety: GIL is managed per-call.
 */

#include "triad_rt.h"
#include <Python.h>

/* ── Conversion helpers ── */

/* Convert a TriadValue to a PyObject* (new reference). */
static PyObject *_tri_to_py(TriadValue v) {
    switch (v.tag) {
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
        for (int32_t i = 0; i < d->len; i++) {
            PyObject *key = PyUnicode_FromStringAndSize(
                d->entries[i].key->data, d->entries[i].key->len);
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

/* Convert a PyObject* to a TriadValue. Steals no reference. */
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
    /* fallback: return repr as string */
    PyObject *repr = PyObject_Repr(obj);
    if (repr && PyUnicode_Check(repr)) {
        Py_ssize_t len;
        const char *data = PyUnicode_AsUTF8AndSize(repr, &len);
        TriadValue sv = (TriadValue){.tag = TRIAD_STRING, .as = {.sval = triad_str_new_len(data, (int32_t)len)}};
        Py_DECREF(repr);
        return sv;
    }
    Py_XDECREF(repr);
    return TRIAD_NONE_VAL;
}

/* ── Python interpreter lifecycle ── */

static int _tri_py_initialized = 0;

static void _tri_py_ensure_init(void) {
    if (!_tri_py_initialized) {
        Py_Initialize();
        _tri_py_initialized = 1;
    }
}

/* ── Public API ── */

/*
 * triad_py_call(module_name, func_name, nargs, args[])
 *
 * Calls Python function: module_name.func_name(*args)
 * Returns TriadValue converted from the Python return value.
 */
static TriadValue triad_py_call(TriadValue mod_v, TriadValue fn_v,
                                 int32_t nargs, TriadValue *args) {
    _tri_py_ensure_init();

    if (mod_v.tag != TRIAD_STRING || fn_v.tag != TRIAD_STRING) {
        fprintf(stderr, "py_call: module and function must be strings\n");
        return TRIAD_NONE_VAL;
    }

    PyGILState_STATE gstate = PyGILState_Ensure();

    /* import module */
    PyObject *mod = PyImport_ImportModule(mod_v.as.sval->data);
    if (!mod) {
        fprintf(stderr, "py_call: cannot import module '%s'\n", mod_v.as.sval->data);
        if (PyErr_Occurred()) { PyErr_Print(); }
        PyGILState_Release(gstate);
        return TRIAD_NONE_VAL;
    }

    /* get function */
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

    /* build args tuple */
    PyObject *py_args = PyTuple_New(nargs);
    for (int32_t i = 0; i < nargs; i++) {
        PyTuple_SET_ITEM(py_args, i, _tri_to_py(args[i]));
    }

    /* call */
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

/*
 * triad_py_eval(code)
 *
 * Evaluates a Python code string and returns the result.
 */
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

/*
 * triad_py_exec(code)
 *
 * Executes Python code (statements). Returns none.
 */
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

#endif /* TRIAD_PYTHON_H */
