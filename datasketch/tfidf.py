from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Iterable, Mapping
from typing import Callable, Optional, Union

import numpy as np
import scipy.sparse as sp

from datasketch.lshforest import MinHashLSHForest
from datasketch.weighted_minhash import WeightedMinHash, WeightedMinHashGenerator

DocumentId = Hashable
Document = Union[str, bytes, Iterable[Hashable]]


class NotFittedError(ValueError, AttributeError):
    """Raised when a TF-IDF weighted index is used before :meth:`fit`.

    Subclassing both :class:`ValueError` and :class:`AttributeError` mirrors
    scikit-learn's exception of the same name, so callers that check for
    either exception type keep working.
    """


class TFIDFWeightedMinHashLSHForest:
    """Top-k near-duplicate document retrieval with TF-IDF weighted MinHash.

    A shared vocabulary and an IDF table are learned from the *whole* corpus
    passed to :meth:`fit` (or :meth:`refresh`). Every document is turned into a
    weighted vector whose entries are the raw term frequency multiplied by the
    term's IDF, and that vector is sketched with
    :class:`~datasketch.WeightedMinHash`. A
    :class:`~datasketch.MinHashLSHForest` is used to collect a pool of
    candidates, which are re-ranked by the exact weighted Jaccard similarity of
    the *same* weighted vectors.

    The IDF of a term ``t`` is

    .. code-block:: text

        idf(t) = ln((1 + N) / (1 + df(t))) + 1

    where ``N`` is the number of documents in the fitted corpus -- documents
    with no tokens included -- and ``df(t)`` is the number of documents that
    contain ``t``. The ``+1`` keeps every weight strictly positive even when a
    term occurs in every document, which :class:`WeightedMinHash` requires.

    All fitted state is immutable with respect to queries: :meth:`query` and
    :meth:`transform` only read the learned vocabulary and IDF table. A
    document or query containing tokens outside the fitted vocabulary drops
    those tokens; they are never added to the vocabulary. Call
    :meth:`refresh` (or :meth:`fit` again) to re-learn on a new corpus; that
    rebuilds every stored sketch, so old summaries, query transformations and
    scores always share one set of statistics.

    Args:
        num_perm (int): The number of samples used by
            :class:`~datasketch.WeightedMinHash` (its ``sample_size``). This is
            also the number of hash values indexed by the LSH Forest.
        l (int): The number of prefix trees in the LSH Forest.
        seed (int): The random seed of the weighted MinHash generator. Use the
            same seed when restoring a pickled index.
        candidate_multiplier (int): The forest returns at most ``k *
            candidate_multiplier`` candidates, which are re-ranked to produce
            the final ``k`` results. A larger value explores more candidates at
            the cost of more exact similarity computations.
        tokenizer (Optional[Callable]): If given, called with each document to
            produce its tokens. By default a document that is ``str`` or
            ``bytes`` is split on whitespace, and any other document is assumed
            to be an iterable of tokens.

    Note:
        Candidate generation is approximate by nature: the LSH Forest can miss
        documents whose sketches do not collide, so the returned list is the
        exact ranking of the *retrieved candidates*, not a guaranteed global
        top-k. Increase ``num_perm``, ``l`` or ``candidate_multiplier`` to trade
        memory and query time for recall.

    Note:
        :class:`~datasketch.WeightedMinHashGenerator` needs to know the number
        of dimensions up front, so the generator holds a ``num_perm x
        len(vocabulary)`` random table. A large vocabulary makes fitting
        correspondingly memory hungry.

    Example:
        .. code-block:: python

            index = TFIDFWeightedMinHashLSHForest(num_perm=128, l=8)
            index.fit(
                {
                    "doc-1": ["the", "quick", "brown", "fox"],
                    "doc-2": ["the", "quick", "brown", "dog"],
                    "doc-3": ["completely", "different", "text"],
                }
            )
            [(doc_id, round(score, 3)) for doc_id, score in index.query(["the", "quick", "brown", "fox"], k=2)]
            # [('doc-1', 1.0), ('doc-2', 0.6...)]

    """

    def __init__(
        self,
        num_perm: int = 128,
        l: int = 8,
        seed: int = 1,
        candidate_multiplier: int = 2,
        tokenizer: Optional[Callable[[object], Iterable[Hashable]]] = None,
    ) -> None:
        if num_perm <= 0:
            raise ValueError("num_perm must be positive")
        if l <= 0 or l > num_perm:
            raise ValueError("l must be positive and cannot be greater than num_perm")
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be at least 1")
        if tokenizer is not None and not callable(tokenizer):
            raise TypeError("tokenizer must be callable")
        self.num_perm = int(num_perm)
        self.l = int(l)
        self.seed = int(seed)
        self.candidate_multiplier = int(candidate_multiplier)
        self._tokenizer = tokenizer
        self._clear()

    # ------------------------------------------------------------------
    # Fitting and corpus refresh
    # ------------------------------------------------------------------
    def fit(
        self,
        documents: Union[Mapping[DocumentId, Document], Iterable[Document], Iterable[tuple]],
        ids: Optional[Iterable[DocumentId]] = None,
    ) -> TFIDFWeightedMinHashLSHForest:
        """Learn the vocabulary and IDF table from a corpus and build the index.

        Args:
            documents: The corpus. This can be

                - a mapping from document ID to the document's tokens,
                - an iterable of documents, with IDs supplied through ``ids``,
                - an iterable of ``(document_id, tokens)`` tuples, or
                - an iterable of documents, in which case IDs are their
                  zero-based positions.

            ids (Optional[Iterable[DocumentId]]): Document IDs, required to have
                the same length as ``documents``. Must be omitted when
                ``documents`` is a mapping.

        Returns:
            TFIDFWeightedMinHashLSHForest: ``self``.

        Raises:
            ValueError: If IDs are duplicated, or ``ids`` has the wrong length,
                or ``ids`` is given together with a mapping.
            TypeError: If an ID is unhashable or a document contains
                unhashable tokens.

        """
        self._build(self._materialize_documents(documents, ids))
        return self

    def refresh(
        self,
        documents: Union[Mapping[DocumentId, Document], Iterable[Document], Iterable[tuple]],
        ids: Optional[Iterable[DocumentId]] = None,
    ) -> TFIDFWeightedMinHashLSHForest:
        """Re-learn all statistics on a new corpus and rebuild everything.

        This is a full rebuild: the vocabulary, the IDF table, the stored
        sketches, the weighted vectors used for scoring and the LSH Forest are
        all replaced. No state from the previous corpus is reused, so a
        document that disappeared from ``documents`` is no longer searchable
        and every remaining document is summarised under the new statistics.

        The new state is assembled before it is published, so if fitting raises
        the index keeps serving its previous, internally consistent state.

        Args:
            documents: The new corpus, in any form accepted by :meth:`fit`.
            ids (Optional[Iterable[DocumentId]]): Document IDs, as in
                :meth:`fit`.

        Returns:
            TFIDFWeightedMinHashLSHForest: ``self``.

        """
        return self.fit(documents, ids)

    def rebuild(
        self,
        documents: Union[Mapping[DocumentId, Document], Iterable[Document], Iterable[tuple]],
        ids: Optional[Iterable[DocumentId]] = None,
    ) -> TFIDFWeightedMinHashLSHForest:
        """Alias for :meth:`refresh`."""
        return self.fit(documents, ids)

    # ------------------------------------------------------------------
    # Query-side transformation
    # ------------------------------------------------------------------
    def token_weights(self, document: Document) -> dict:
        """Return the fitted TF-IDF weights of the known tokens in ``document``.

        Weights are ``raw term frequency * idf``, exactly as used to build the
        indexed documents. Tokens missing from the fitted vocabulary are
        omitted (they contribute zero weight); nothing is learned here.

        Args:
            document: A document, tokenised the same way as during fitting.

        Returns:
            dict: Mapping from token to its positive weight. Preserves the
            order in which tokens first appear in the document.

        """
        self._check_fitted()
        counts = Counter(self._iter_tokens(document))
        weights = {}
        for token, frequency in counts.items():
            column = self.vocabulary_.get(token)
            if column is not None:
                weights[token] = float(frequency) * float(self.idf_[column])
        return weights

    def transform(self, document: Document) -> Optional[WeightedMinHash]:
        """Sketch a document with the fitted statistics.

        The returned sketch lives in the fitted vocabulary's space; unknown
        tokens are dropped. This method never learns new terms and never
        modifies the fitted state.

        Args:
            document: A document, tokenised the same way as during fitting.

        Returns:
            Optional[WeightedMinHash]: The weighted MinHash of the document, or
            ``None`` if the document has no token in the fitted vocabulary
            (empty documents included).

        Raises:
            NotFittedError: If the index has not been fitted.

        """
        self._check_fitted()
        row = self._weighted_row(document)
        if row is None:
            return None
        return self._generator.minhash_many(row)[0]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def query(self, document: Document, k: int = 10) -> list:
        """Return the current document IDs most similar to ``document``.

        The query is transformed with the fitted vocabulary and IDF table; no
        token from the query is ever learned. Up to ``k *
        candidate_multiplier`` candidates are fetched from the LSH Forest and
        ranked by the exact weighted Jaccard similarity of the TF-IDF weighted
        vectors. Because the candidate stage is approximate, the result is the
        exact ranking of the retrieved candidates rather than a guaranteed
        global top-k.

        Args:
            document: The query document, tokenised like indexed documents.
            k (int): The maximum number of results.

        Returns:
            list: ``(document_id, score)`` tuples, ordered by decreasing score
            and then by corpus order for ties. Empty when the query has no
            known token, when the corpus is empty, or when the forest returns
            no candidate.

        Raises:
            ValueError: If ``k`` is not positive.
            NotFittedError: If the index has not been fitted.

        """
        if k <= 0:
            raise ValueError("k must be positive")
        self._check_fitted()
        row = self._weighted_row(document)
        if row is None or not self._positions:
            return []
        query_sketch = self._generator.minhash_many(row)[0]
        if query_sketch is None:
            return []
        candidate_count = min(len(self.searchable_ids_), k * self.candidate_multiplier)
        candidates = self._forest.query(query_sketch, candidate_count)
        if not candidates:
            return []
        positions = [self._positions[doc_id] for doc_id in candidates]
        scores = self._weighted_jaccard(positions, row)
        ranked = sorted(zip(candidates, scores), key=lambda item: (-item[1], self._positions[item[0]]))
        return [(doc_id, float(score)) for doc_id, score in ranked[:k]]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        """Whether the vocabulary and IDF table have been learned."""
        return self._is_fitted

    @property
    def forest(self) -> MinHashLSHForest:
        """The underlying LSH Forest, for callers that need the raw candidates.

        Raises:
            NotFittedError: If the index has not been fitted.

        """
        self._check_fitted()
        return self._forest

    def get_sketch(self, document_id: DocumentId) -> WeightedMinHash:
        """Return the stored weighted MinHash of an indexed document.

        Args:
            document_id: ID of an indexed document.

        Returns:
            WeightedMinHash: The sketch built during the last fit.

        Raises:
            KeyError: If the ID has no sketch -- it is not in the corpus, or it
                is an empty document that is counted in the corpus but has no
                weighted vector.
            NotFittedError: If the index has not been fitted.

        """
        self._check_fitted()
        try:
            return self.sketches_[document_id]
        except KeyError:
            raise KeyError("no sketch for document %r" % (document_id,)) from None

    def __len__(self) -> int:
        """The number of documents in the fitted corpus, empty ones included."""
        return self.num_docs_

    def __contains__(self, document_id: DocumentId) -> bool:
        """Whether ``document_id`` is part of the fitted corpus.

        Empty documents are part of the corpus, so they are contained even
        though they have no sketch and cannot be retrieved.
        """
        return document_id in self._corpus_id_set

    def __repr__(self) -> str:
        if not self._is_fitted:
            return "%s(num_perm=%d, l=%d, unfitted)" % (type(self).__name__, self.num_perm, self.l)
        return "%s(num_perm=%d, l=%d, documents=%d, vocabulary=%d)" % (
            type(self).__name__,
            self.num_perm,
            self.l,
            self.num_docs_,
            len(self.vocabulary_),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _clear(self) -> None:
        self.vocabulary_: dict = {}
        self.idf_: np.ndarray = np.empty(0, dtype=np.float64)
        self.num_docs_: int = 0
        self.corpus_ids_: list = []
        self.searchable_ids_: list = []
        self.empty_ids_: list = []
        self.sketches_: dict = {}
        self._corpus_id_set = set()
        self._weighted = sp.csr_matrix((0, 0), dtype=np.float64)
        self._row_sums: np.ndarray = np.empty(0, dtype=np.float64)
        self._positions: dict = {}
        self._generator: Optional[WeightedMinHashGenerator] = None
        self._forest: Optional[MinHashLSHForest] = None
        self._is_fitted = False

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise NotFittedError(
                "This %s instance is not fitted yet. Call 'fit' with an iterable of documents first."
                % type(self).__name__
            )

    def _iter_tokens(self, document: Document) -> Iterable[Hashable]:
        if self._tokenizer is not None:
            return self._tokenizer(document)
        # Treat raw text as whitespace-separated tokens; anything else is
        # already a token sequence.
        if isinstance(document, (str, bytes)):
            return document.split()
        return document

    def _materialize_documents(
        self,
        documents: Union[Mapping[DocumentId, Document], Iterable[Document], Iterable[tuple]],
        ids: Optional[Iterable[DocumentId]],
    ) -> list:
        if isinstance(documents, (str, bytes)):
            raise TypeError("documents must be a collection of documents, not a single string")
        if isinstance(documents, Mapping):
            if ids is not None:
                raise ValueError("ids must not be given when documents is a mapping")
            pairs = list(documents.items())
        else:
            items = list(documents)
            if ids is not None:
                if isinstance(ids, (str, bytes)):
                    raise TypeError("ids must be a collection of document ids, not a single string")
                id_list = list(ids)
                if len(id_list) != len(items):
                    raise ValueError(
                        "ids and documents must have the same length: %d != %d" % (len(id_list), len(items))
                    )
                pairs = list(zip(id_list, items))
            elif self._looks_like_id_document_pairs(items):
                pairs = items
            else:
                pairs = list(enumerate(items))
        seen = set()
        for document_id, _ in pairs:
            if document_id in seen:
                raise ValueError("duplicate document id: %r" % (document_id,))
            seen.add(document_id)
        return pairs

    @staticmethod
    def _looks_like_id_document_pairs(items: list) -> bool:
        """Tell ``[(id, tokens), ...]`` apart from a list of documents.

        A token sequence is a non-string iterable, so a document that happens
        to be a 2-tuple of strings is not mistaken for an ``(id, tokens)``
        pair. The remaining ambiguity -- a document that is itself
        ``(id, tokens)`` -- is resolved in favour of pairs; pass ``ids``
        explicitly to force per-document interpretation.
        """
        if not items:
            return False
        for item in items:
            if not isinstance(item, tuple) or len(item) != 2:
                return False
            tokens = item[1]
            if isinstance(tokens, (str, bytes)) or not isinstance(tokens, Iterable):
                return False
        return True

    def _count_tokens(self, document_id: DocumentId, document: Document) -> Counter:
        try:
            return Counter(self._iter_tokens(document))
        except TypeError as error:
            raise TypeError("document %r must be a sequence of hashable tokens" % (document_id,)) from error

    def _weighted_row(self, document: Document) -> Optional[sp.csr_matrix]:
        weights = self.token_weights(document)
        if not weights:
            return None
        columns = np.fromiter((self.vocabulary_[token] for token in weights), dtype=np.int64, count=len(weights))
        values = np.fromiter(weights.values(), dtype=np.float64, count=len(weights))
        indptr = np.array([0, len(columns)], dtype=np.int64)
        row = sp.csr_matrix((values, columns, indptr), shape=(1, len(self.vocabulary_)))
        row.sort_indices()
        return row

    def _weighted_jaccard(self, positions: list, query: sp.csr_matrix) -> np.ndarray:
        """Exact weighted Jaccard between the query row and stored rows.

        Only the columns where the query is nonzero can contribute to the
        intersection, so the candidate block is sliced down to them before the
        element-wise minimum. This also sidesteps scipy's sparse
        ``minimum``/``maximum`` broadcasting of a single row, which is not
        reliable across SciPy versions.
        """
        stored = self._weighted[positions]
        query_columns = query.indices
        query_values = query.data
        if query_columns.size:
            block = stored[:, query_columns].toarray()
            intersection = np.minimum(block, query_values).sum(axis=1)
        else:
            intersection = np.zeros(len(positions), dtype=np.float64)
        union = self._row_sums[positions] + float(query.sum()) - intersection
        return intersection / union

    def _build(self, pairs: list) -> None:
        # Everything is assembled in local variables and only published once
        # the whole rebuild has succeeded.
        document_count = len(pairs)
        counts = []
        vocabulary: dict = {}
        for document_id, document in pairs:
            counter = self._count_tokens(document_id, document)
            counts.append(counter)
            for token in counter:
                if token not in vocabulary:
                    vocabulary[token] = len(vocabulary)
        dimension = len(vocabulary)

        document_frequency = np.zeros(dimension, dtype=np.int64)
        for counter in counts:
            for token in counter:
                document_frequency[vocabulary[token]] += 1
        if dimension:
            idf = np.log((1.0 + document_count) / (1.0 + document_frequency)) + 1.0
        else:
            idf = np.empty(0, dtype=np.float64)

        indptr = np.zeros(document_count + 1, dtype=np.int64)
        indices: list = []
        data: list = []
        for row, counter in enumerate(counts):
            for token, frequency in counter.items():
                column = vocabulary[token]
                indices.append(column)
                data.append(float(frequency) * float(idf[column]))
            indptr[row + 1] = len(indices)
        weighted = sp.csr_matrix(
            (np.asarray(data, dtype=np.float64), np.asarray(indices, dtype=np.int64), indptr),
            shape=(document_count, dimension),
        )
        weighted.sort_indices()

        row_nnz = np.diff(weighted.indptr)
        indexed_rows = np.flatnonzero(row_nnz)
        weighted_indexed = weighted[indexed_rows]
        if indexed_rows.size:
            row_sums = np.asarray(weighted_indexed.sum(axis=1), dtype=np.float64).ravel()
        else:
            row_sums = np.empty(0, dtype=np.float64)
        searchable_ids = [pairs[int(row)][0] for row in indexed_rows]
        empty_ids = [pairs[row][0] for row in range(document_count) if row_nnz[row] == 0]

        forest = MinHashLSHForest(num_perm=self.num_perm, l=self.l)
        generator = None
        sketches = {}
        if dimension:
            generator = WeightedMinHashGenerator(dimension, sample_size=self.num_perm, seed=self.seed)
            for document_id, sketch in zip(searchable_ids, generator.minhash_many(weighted_indexed)):
                if sketch is None:
                    continue
                sketches[document_id] = sketch
                forest.add(document_id, sketch)
        forest.index()

        self.vocabulary_ = vocabulary
        self.idf_ = idf
        self.num_docs_ = document_count
        self.corpus_ids_ = [document_id for document_id, _ in pairs]
        self.searchable_ids_ = searchable_ids
        self.empty_ids_ = empty_ids
        self.sketches_ = sketches
        self._corpus_id_set = set(self.corpus_ids_)
        self._weighted = weighted_indexed
        self._row_sums = row_sums
        self._positions = {document_id: position for position, document_id in enumerate(searchable_ids)}
        self._generator = generator
        self._forest = forest
        self._is_fitted = True


# Shorter aliases for the same index.
TFIDFWeightedMinHash = TFIDFWeightedMinHashLSHForest
TFIDFMinHashLSHForest = TFIDFWeightedMinHashLSHForest
