"""Implementations of the domain ports for concrete providers (Ollama, Chroma...).

This is the only package that imports provider libraries. Adding a provider
(Google, Voyage, Anthropic, another vector database) means adding a module
here and registering it in `aporia_rag.bootstrap`; nothing else changes.
"""
