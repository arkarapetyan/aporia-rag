"""Infrastructure-free core: data models (and, later, the ports adapters implement).

Nothing in this package may import LangChain, LangGraph, Chroma, Ollama or any
other infrastructure library. Every other layer depends on `domain`; `domain`
depends on nothing but pydantic.
"""
