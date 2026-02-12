Getting Started
===============

Installation
------------

.. code-block:: bash

   pip install contextkit

With optional backends:

.. code-block:: bash

   pip install contextkit[sqlite]    # Persistent memory
   pip install contextkit[chroma]    # ChromaDB retriever
   pip install contextkit[all]       # Everything

Quick Example
-------------

.. code-block:: python

   from contextkit import ContextWindow, ContextBlock, BlockType

   window = ContextWindow(model="claude-sonnet-4-5-20250929")

   window.add(ContextBlock(
       type=BlockType.SYSTEM_PROMPT,
       content="You are a helpful assistant.",
       priority=100,
   ))

   window.add(ContextBlock(
       type=BlockType.USER_CONTEXT,
       content="What is context engineering?",
       priority=90,
   ))

   # Inspect the assembled context
   print(window.inspect())

   # Check token usage
   print(f"Tokens: {window.token_count}/{window.max_tokens}")

Core Concepts
-------------

**ContextWindow**
   The central container that holds all context blocks, tracks token
   usage, enforces budgets, and emits lifecycle events.

**ContextBlock**
   A typed unit of context (system prompt, memory, RAG chunk, tool
   output, etc.) with a priority that controls assembly order.

**ContextAssembler**
   Composes blocks into a window by priority, producing an
   ``AssemblyReport`` explaining inclusion/exclusion decisions.

**ContextPipeline**
   Runs optimization steps (dedup, compress, trim, reorder) to fit
   context within token budgets.

See the :doc:`API Reference <autoapi/index>` for full details.
