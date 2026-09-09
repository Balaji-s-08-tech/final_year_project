"""Model implementations bundled in this checkout.

The upstream initializer eagerly imported Baichuan, LLaMA, and ChatGLM
implementations.  Those directories are not present in this repository, which
made every direct import of the bundled Phi-2/Phi-3 modules fail while Python
initialised this package.  Keep this namespace deliberately lazy: callers
already import their required implementation explicitly.
"""
