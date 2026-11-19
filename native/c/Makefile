CC      = gcc
CFLAGS  = -std=c11 -Wall -Wextra -O2 -fPIC -Iinclude
SRCDIR  = src
OBJDIR  = obj
LIB     = libtriad_rt.a

# Boehm GC is the default. Use GC=no to fall back to malloc/free.
ifneq ($(GC),no)
  CFLAGS += # Boehm is default via triad_rt.h
  GC_LDFLAGS = -lgc
else
  CFLAGS += -DTRIAD_NO_BOEHM
  GC_LDFLAGS =
endif

ifneq ($(wildcard /usr/include/fftw3.h),)
  CFLAGS += -DUSE_FFTW
  FFTW_LDFLAGS = -lfftw3
endif

# CUDA detection
NVCC   := $(shell which nvcc 2>/dev/null)
ifneq ($(NVCC),)
  CUDA_CFLAGS = -Iinclude
  CUDA_LDFLAGS = -L/opt/cuda/lib64 -lcublas -lcudart -lstdc++
  CFLAGS += -DHAVE_CUDA
endif

SRCS    = $(SRCDIR)/triad_gc.c \
          $(SRCDIR)/triad_string.c \
          $(SRCDIR)/triad_list.c \
          $(SRCDIR)/triad_dict.c \
          $(SRCDIR)/triad_stdlib.c \
          $(SRCDIR)/triad_ndarray.c \
          $(SRCDIR)/triad_object.c \
          $(SRCDIR)/triad_fftw.c \
          $(SRCDIR)/triad_blas.c \
          $(SRCDIR)/triad_solver.c \
          $(SRCDIR)/triad_autograd.c \
          $(SRCDIR)/triad_nn.c \
          $(SRCDIR)/triad_llm.c \
          $(SRCDIR)/triad_gguf.c \
          $(SRCDIR)/triad_safetensors.c \
          $(SRCDIR)/triad_model_index.c \
          $(SRCDIR)/triad_train.c \
          $(SRCDIR)/triad_arena.c \
          $(SRCDIR)/triad_lexer.c \
          $(SRCDIR)/triad_ast.c \
          $(SRCDIR)/triad_parser.c \
          $(SRCDIR)/triad_ast_dump.c \
          $(SRCDIR)/triad_lexer_legacy.c \
          $(SRCDIR)/triad_parser_legacy.c \
          $(SRCDIR)/triad_legacy_dump.c \
          $(SRCDIR)/triad_lower.c \
          $(SRCDIR)/triad_ir_dump.c \
          $(SRCDIR)/triad_check.c \
          $(SRCDIR)/triad_check_legacy.c \
          $(SRCDIR)/triad_format.c \
          $(SRCDIR)/triad_docgen.c \
          $(SRCDIR)/triad_lsp.c \
          $(SRCDIR)/triad_observables.c \
          $(SRCDIR)/triad_codec.c \
          $(SRCDIR)/triad_multi_runtime.c \
          $(SRCDIR)/triad_observables_atoms.c \
          $(SRCDIR)/triad_compiler.c

CU_SRCS = $(SRCDIR)/triad_cuda.cu

OBJS    = $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(SRCS))
ifneq ($(NVCC),)
  CU_OBJS = $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(CU_SRCS))
else
  CU_OBJS =
endif

CHAT_LDFLAGS = -L. -ltriad_rt -lm $(GC_LDFLAGS)
ifneq ($(NVCC),)
  CHAT_LDFLAGS += $(CUDA_LDFLAGS)
endif
ifneq ($(wildcard /usr/include/fftw3.h),)
  CHAT_LDFLAGS += $(FFTW_LDFLAGS)
endif

SO      = libtriad_rt.so

.PHONY: all clean static chat frontend parity_ast test_frontend triad-cli parity shared

all: static shared chat frontend triad-cli

frontend: parity_ast parity_ast_legacy parity_ir parity_check_native parity_check_legacy parity_format parity_docgen parity_lsp parity_codec parity_multi_runtime parity_multi_runtime_2d parity_multi_runtime_3d parity_multi_runtime_2d_modes parity_multi_runtime_3d_modes parity_atoms parity_compiler parity_solver_1d test_lexer test_parser test_legacy_frontend test_lower test_check test_codec test_multi_runtime test_observables_atoms test_compiler

parity_ast: static parity/parity_ast.c
	$(CC) $(CFLAGS) parity/parity_ast.c $(OBJS) $(CU_OBJS) $(CHAT_LDFLAGS) -o parity/parity_ast

parity_ast_legacy: static parity/parity_ast_legacy.c
	$(CC) $(CFLAGS) parity/parity_ast_legacy.c $(OBJS) $(CU_OBJS) $(CHAT_LDFLAGS) -o parity/parity_ast_legacy

parity_ir: static parity/parity_ir.c
	$(CC) $(CFLAGS) parity/parity_ir.c $(OBJS) $(CU_OBJS) $(CHAT_LDFLAGS) -o parity/parity_ir

parity_check_native: static parity/parity_check.c
	$(CC) $(CFLAGS) parity/parity_check.c $(OBJS) $(CU_OBJS) $(CHAT_LDFLAGS) -o parity/parity_check_native

parity_check_legacy: static parity/parity_check_legacy.c
	$(CC) $(CFLAGS) parity/parity_check_legacy.c $(OBJS) $(CU_OBJS) $(CHAT_LDFLAGS) -o parity/parity_check_legacy

parity_format: static parity/parity_format.c
	$(CC) $(CFLAGS) parity/parity_format.c $(OBJS) $(CU_OBJS) $(CHAT_LDFLAGS) -o parity/parity_format

parity_docgen: static parity/parity_docgen.c
	$(CC) $(CFLAGS) parity/parity_docgen.c $(OBJS) $(CU_OBJS) $(CHAT_LDFLAGS) -o parity/parity_docgen

parity_lsp: static parity/parity_lsp.c
	$(CC) $(CFLAGS) parity/parity_lsp.c $(OBJS) $(CU_OBJS) $(CHAT_LDFLAGS) -o parity/parity_lsp

parity_codec: static parity/parity_codec.c
	$(CC) $(CFLAGS) parity/parity_codec.c $(OBJS) $(CU_OBJS) $(CHAT_LDFLAGS) -o parity/parity_codec

parity_multi_runtime: static parity/parity_multi_runtime.c
	$(CC) $(CFLAGS) parity/parity_multi_runtime.c $(OBJS) $(CU_OBJS) $(CHAT_LDFLAGS) -o parity/parity_multi_runtime

parity_multi_runtime_2d: static parity/parity_multi_runtime_2d.c
	$(CC) $(CFLAGS) parity/parity_multi_runtime_2d.c $(OBJS) $(CU_OBJS) $(CHAT_LDFLAGS) -o parity/parity_multi_runtime_2d

parity_multi_runtime_3d: static parity/parity_multi_runtime_3d.c
	$(CC) $(CFLAGS) parity/parity_multi_runtime_3d.c $(OBJS) $(CU_OBJS) $(CHAT_LDFLAGS) -o parity/parity_multi_runtime_3d

parity_multi_runtime_2d_modes: static parity/parity_multi_runtime_2d_modes.c
	$(CC) $(CFLAGS) parity/parity_multi_runtime_2d_modes.c $(OBJS) $(CU_OBJS) $(CHAT_LDFLAGS) -o parity/parity_multi_runtime_2d_modes

parity_multi_runtime_3d_modes: static parity/parity_multi_runtime_3d_modes.c
	$(CC) $(CFLAGS) parity/parity_multi_runtime_3d_modes.c $(OBJS) $(CU_OBJS) $(CHAT_LDFLAGS) -o parity/parity_multi_runtime_3d_modes

parity_atoms: static parity/parity_atoms.c
	$(CC) $(CFLAGS) parity/parity_atoms.c $(OBJS) $(CU_OBJS) $(CHAT_LDFLAGS) -o parity/parity_atoms

parity_compiler: static parity/parity_compiler.c
	$(CC) $(CFLAGS) parity/parity_compiler.c $(OBJS) $(CU_OBJS) $(CHAT_LDFLAGS) -o parity/parity_compiler

parity_solver_1d: static parity/parity_solver_1d.c
	$(CC) $(CFLAGS) parity/parity_solver_1d.c $(OBJS) $(CU_OBJS) $(CHAT_LDFLAGS) -o parity/parity_solver_1d

test_codec: static tests/test_codec.c
	$(CC) $(CFLAGS) tests/test_codec.c $(OBJS) $(CU_OBJS) $(CHAT_LDFLAGS) -o tests/test_codec

test_multi_runtime: static tests/test_multi_runtime.c
	$(CC) $(CFLAGS) tests/test_multi_runtime.c $(OBJS) $(CU_OBJS) $(CHAT_LDFLAGS) -o tests/test_multi_runtime

test_observables_atoms: static tests/test_observables_atoms.c
	$(CC) $(CFLAGS) tests/test_observables_atoms.c $(OBJS) $(CU_OBJS) $(CHAT_LDFLAGS) -o tests/test_observables_atoms

test_compiler: static tests/test_compiler.c
	$(CC) $(CFLAGS) tests/test_compiler.c $(OBJS) $(CU_OBJS) $(CHAT_LDFLAGS) -o tests/test_compiler

triad-cli: static triad_cli.c
	$(CC) $(CFLAGS) triad_cli.c $(OBJS) $(CU_OBJS) $(CHAT_LDFLAGS) -o triad

test_lexer: static tests/test_lexer.c
	$(CC) $(CFLAGS) tests/test_lexer.c $(OBJS) $(CU_OBJS) $(CHAT_LDFLAGS) -o tests/test_lexer

test_parser: static tests/test_parser.c
	$(CC) $(CFLAGS) tests/test_parser.c $(OBJS) $(CU_OBJS) $(CHAT_LDFLAGS) -o tests/test_parser

test_legacy_frontend: static tests/test_legacy_frontend.c
	$(CC) $(CFLAGS) tests/test_legacy_frontend.c $(OBJS) $(CU_OBJS) $(CHAT_LDFLAGS) -o tests/test_legacy_frontend

test_lower: static tests/test_lower.c
	$(CC) $(CFLAGS) tests/test_lower.c $(OBJS) $(CU_OBJS) $(CHAT_LDFLAGS) -o tests/test_lower

test_check: static tests/test_check.c
	$(CC) $(CFLAGS) tests/test_check.c $(OBJS) $(CU_OBJS) $(CHAT_LDFLAGS) -o tests/test_check

test_solver_nd: static tests/test_solver_nd.c
	$(CC) $(CFLAGS) tests/test_solver_nd.c $(OBJS) $(CU_OBJS) $(CHAT_LDFLAGS) -o tests/test_solver_nd

test_observables: static tests/test_observables.c
	$(CC) $(CFLAGS) tests/test_observables.c $(OBJS) $(CU_OBJS) $(CHAT_LDFLAGS) -o tests/test_observables

test_frontend: test_lexer test_parser test_legacy_frontend test_lower test_check
	./tests/test_lexer && ./tests/test_parser && ./tests/test_legacy_frontend && ./tests/test_lower && ./tests/test_check

# Aggregate parity driver: runs every native parity_* binary side-by-side with
# its Python reference across all .tri fixtures in the repo. See parity_run_all.py.
parity: frontend
	PYTHONPATH=../../src python3 parity/parity_run_all.py

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu | $(OBJDIR)
	$(NVCC) $(CUDA_CFLAGS) -c $< -o $@

static: $(OBJS) $(CU_OBJS)
	ar rcs $(LIB) $^
	@echo "Built $(LIB)"

shared: $(OBJS) $(CU_OBJS)
	$(CC) -shared -o $(SO) $^ $(CHAT_LDFLAGS)
	@echo "Built $(SO)"

chat: $(OBJS) $(CU_OBJS)
	$(CC) $(CFLAGS) triad_chat.c $(OBJS) $(CU_OBJS) $(CHAT_LDFLAGS) -o triad-chat

clean:
	rm -rf $(OBJDIR) $(LIB) triad-chat
