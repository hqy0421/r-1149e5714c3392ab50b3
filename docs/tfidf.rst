.. _tfidf_weighted_minhash:

TF-IDF Weighted MinHash LSH Forest
===================================

Token sets ignore how often a token occurs, and treat every token as
equally informative, so the high-frequency words shared by most
documents dominate the similarity. :class:`datasketch.TFIDFWeightedMinHashLSHForest`
addresses both problems:

- each document is a *weighted* vector whose entry for a token is the
  document's **raw term frequency** (count) of that token multiplied by
  the token's **IDF**, learned from the whole corpus;
- the vector is sketched with :ref:`weighted_minhash`, so the sketch
  estimates the weighted Jaccard similarity of the TF-IDF vectors;
- a :ref:`minhash_lsh_forest` yields a candidate pool, and the final
  top-k is the *exact* weighted Jaccard ranking of those candidates.

Fitting
-------

One call to ``fit`` learns a vocabulary and an IDF table from the entire
corpus, then builds every sketch and the forest:

.. code:: python

    from datasketch import TFIDFWeightedMinHashLSHForest

    documents = {
        "doc-1": "the quick brown fox jumps over the lazy dog",
        "doc-2": "the quick brown fox leaps over a lazy dog",
        "doc-3": "an unrelated essay about tax policy",
        "doc-4": "",
    }
    index = TFIDFWeightedMinHashLSHForest(num_perm=128, l=8)
    index.fit(documents)

    print(len(index))                # 4: empty documents count too
    print(len(index.vocabulary_))    # terms seen anywhere in the corpus

The corpus can also be given as an iterable of documents with ``ids``,
or as ``(document_id, tokens)`` pairs; without IDs the zero-based
positions are used. A document that is ``str``/``bytes`` is split on
whitespace, while any other value is treated as a sequence of tokens.
Pass ``tokenizer=...`` to plug in your own tokenisation.

The IDF of a term ``t`` is

.. code:: text

    idf(t) = ln((1 + N) / (1 + df(t))) + 1

where ``N`` is the number of documents in the corpus -- documents with
no tokens included -- and ``df(t)`` is the number of documents
containing ``t``. The ``+1`` keeps the weight of a term that appears in
every document positive, which :class:`datasketch.WeightedMinHash`
requires.

IDs, terms and empty documents
------------------------------

- **Empty documents count.** A document with no tokens is part of ``N``
  and therefore influences every IDF. It is listed in
  ``index.empty_ids_`` and is contained in the index, but it has no
  sketch, is never returned by ``query``, and ``get_sketch`` raises
  ``KeyError`` for it. ``index.corpus_ids_`` lists the whole corpus and
  ``index.searchable_ids_`` the documents that have a sketch.
- **Unknown terms are dropped.** A token outside the fitted vocabulary
  contributes no weight. This holds for indexed documents, for
  ``transform`` and for ``query``; a query is never allowed to add a term
  to the vocabulary.

Querying
--------

.. code:: python

    for document_id, score in index.query("the quick brown fox", k=2):
        print(document_id, round(score, 3))
    # doc-1 0.432
    # doc-2 0.419

The result is a list of ``(document_id, score)`` pairs, already sorted by
decreasing score (ties keep corpus order). The score is the exact
weighted Jaccard similarity of the TF-IDF vectors under the fitted IDF
table, so ``1.0`` means identical weighted vectors.

``query`` is a two-stage search, and only the second stage is exact:

1. the query is sketched with the fitted vocabulary and IDF table, and
   the LSH Forest returns up to ``k * candidate_multiplier`` candidates;
2. those candidates are re-ranked by exact weighted Jaccard.

Because the forest is approximate, a document whose sketch fails to
collide with the query's can be missing from the candidate pool even if
it would have ranked highly. The returned list is the exact ranking of
the retrieved candidates rather than a guaranteed global top-k.
``candidate_multiplier`` (default 2) and ``num_perm``/``l`` trade query
cost for recall. A query with no known token -- ``""`` included -- returns
an empty list without touching the forest.

Two helpers expose the stages separately. ``token_weights(document)``
returns the fitted ``term frequency * IDF`` weights, and
``transform(document)`` returns the :class:`datasketch.WeightedMinHash`
that ``query`` would look up (``None`` if the document has no known
token). Neither learns anything. The underlying forest is reachable as
``index.forest`` if the raw candidate keys are needed.

Refreshing the corpus
---------------------

When the corpus changes, the vocabulary and IDF table must change with
it; reusing old statistics would mix two different weightings in one
ranking. ``refresh`` re-learns everything on the new corpus in one go:

.. code:: python

    index.refresh({"doc-1": "...", "doc-2": "...", "doc-5": "..."})

After ``refresh``, the vocabulary, the IDF table, the stored sketches,
the weighted vectors used for scoring and the forest are all replaced:
documents that disappeared are no longer searchable, and every remaining
document is summarised and scored under the new statistics. Old sketches,
query transformations and scores never survive a refresh, so they cannot
drift apart. The new state is assembled before it is published, so a
failed refresh leaves the previous, internally consistent index in place.

``fit`` performs the same rebuild when called again, and ``rebuild`` is
an alias for ``refresh``. There is no incremental update: add documents
by refreshing with the enlarged corpus.

Choosing parameters
-------------------

``num_perm`` is the number of samples in each weighted MinHash and the
number of hash values the forest indexes; larger values improve both the
sketch accuracy and the candidate recall. ``l`` is the number of prefix
trees. The weighted MinHash generator stores a
``num_perm x len(vocabulary)`` random table, so a very large vocabulary
is the dominant memory cost.
