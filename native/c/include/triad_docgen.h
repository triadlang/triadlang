#ifndef TRIAD_DOCGEN_H
#define TRIAD_DOCGEN_H

#include "triad_frontend.h"

#ifdef __cplusplus
extern "C" {
#endif

const char *triad_docgen_markdown(TriadArena *arena,
                                  const char *source,
                                  const char *filename);

const char *triad_docgen_html(TriadArena *arena,
                              const char *source,
                              const char *filename);

#ifdef __cplusplus
}
#endif

#endif
