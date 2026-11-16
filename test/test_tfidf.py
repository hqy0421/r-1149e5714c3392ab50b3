import pickle
import unittest

import numpy as np

from datasketch import NotFittedError, TFIDFWeightedMinHashLSHForest
from datasketch.tfidf import TFIDFMinHashLSHForest, TFIDFWeightedMinHash
from datasketch.weighted_minhash import WeightedMinHashGenerator


def _weighted_jaccard(left, right):
    keys = set(left) | set(right)
    numerator = sum(min(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    denominator = sum(max(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    return numerator / denominator


class TestFit(unittest.TestCase):
    def setUp(self):
        self.documents = {
            "a": "the quick brown fox jumps",
            "b": "the quick brown fox leaps",
            "c": "completely unrelated sentence about taxes",
            "d": "",
        }

    def test_fit_returns_self_and_marks_fitted(self):
        index = TFIDFWeightedMinHashLSHForest(num_perm=64, l=4)
        self.assertFalse(index.is_fitted)
        self.assertIs(index.fit(self.documents), index)
        self.assertTrue(index.is_fitted)

    def test_vocabulary_spans_whole_corpus(self):
        index = TFIDFWeightedMinHashLSHForest().fit(self.documents)
        self.assertEqual(
            set(index.vocabulary_),
            {"the", "quick", "brown", "fox", "jumps", "leaps", "completely", "unrelated", "sentence", "about", "taxes"},
        )
        # Intentionally deterministic: first appearance order.
        self.assertEqual(index.vocabulary_["the"], 0)
        self.assertEqual(len(index.idf_), len(index.vocabulary_))

    def test_empty_documents_count_towards_corpus(self):
        index = TFIDFWeightedMinHashLSHForest().fit(self.documents)
        self.assertEqual(len(index), 4)
        self.assertEqual(index.corpus_ids_, ["a", "b", "c", "d"])
        self.assertEqual(index.empty_ids_, ["d"])
        self.assertIn("d", index)
        with self.assertRaises(KeyError):
            index.get_sketch("d")

    def test_idf_formula(self):
        index = TFIDFWeightedMinHashLSHForest().fit(self.documents)
        # "the" appears in 2 of 4 documents (the empty document has no tokens
        # but still counts as a document).
        expected = np.log((1 + 4) / (1 + 2)) + 1
        self.assertAlmostEqual(index.idf_[index.vocabulary_["the"]], expected)
        # "jumps" is a singleton; "taxes" too.
        expected_singleton = np.log((1 + 4) / (1 + 1)) + 1
        self.assertAlmostEqual(index.idf_[index.vocabulary_["jumps"]], expected_singleton)

    def test_idf_is_strictly_positive_for_every_term(self):
        index = TFIDFWeightedMinHashLSHForest().fit({"a": "same same", "b": "same", "c": ""})
        self.assertTrue(np.all(index.idf_ > 0))
        # When a term occurs in every document, the +1 floor keeps it positive
        # and equal to 1, so WeightedMinHash never sees a zero weight.
        saturated = TFIDFWeightedMinHashLSHForest().fit({"a": "same", "b": "same", "c": "same"})
        self.assertAlmostEqual(float(saturated.idf_[saturated.vocabulary_["same"]]), 1.0)

    def test_weights_are_raw_term_frequency_times_idf(self):
        index = TFIDFWeightedMinHashLSHForest().fit(self.documents)
        weights = index.token_weights("the the quick")
        self.assertAlmostEqual(weights["the"], 2 * float(index.idf_[index.vocabulary_["the"]]))
        self.assertAlmostEqual(weights["quick"], 1 * float(index.idf_[index.vocabulary_["quick"]]))
        self.assertEqual(list(weights), ["the", "quick"])

    def test_unknown_tokens_get_no_weight(self):
        index = TFIDFWeightedMinHashLSHForest().fit(self.documents)
        self.assertEqual(index.token_weights("never seen before"), {})

    def test_fit_forms_are_equivalent(self):
        mapping = TFIDFWeightedMinHashLSHForest(seed=2).fit({"a": "x y", "b": "x z"})
        pairs = TFIDFWeightedMinHashLSHForest(seed=2).fit([("a", ["x", "y"]), ("b", ["x", "z"])])
        with_ids = TFIDFWeightedMinHashLSHForest(seed=2).fit(["x y", "x z"], ids=["a", "b"])
        positional = TFIDFWeightedMinHashLSHForest(seed=2).fit(["x y", "x z"])
        for other in (pairs, with_ids):
            self.assertEqual(mapping.vocabulary_, other.vocabulary_)
            self.assertEqual(mapping.searchable_ids_, other.searchable_ids_)
            np.testing.assert_array_equal(mapping.idf_, other.idf_)
        self.assertEqual(positional.corpus_ids_, [0, 1])

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            TFIDFWeightedMinHashLSHForest().fit(["x", "y"], ids=["a", "a"])

    def test_ids_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            TFIDFWeightedMinHashLSHForest().fit(["x", "y"], ids=["a"])

    def test_ids_with_mapping_rejected(self):
        with self.assertRaises(ValueError):
            TFIDFWeightedMinHashLSHForest().fit({"a": "x"}, ids=["a"])

    def test_single_string_corpus_rejected(self):
        with self.assertRaises(TypeError):
            TFIDFWeightedMinHashLSHForest().fit("just one string")

    def test_custom_tokenizer(self):
        index = TFIDFWeightedMinHashLSHForest(tokenizer=lambda text: text.lower().split(",")).fit(
            {"a": "A,B", "b": "B,C"}, ids=None
        )
        self.assertEqual(set(index.vocabulary_), {"a", "b", "c"})

    def test_empty_corpus(self):
        index = TFIDFWeightedMinHashLSHForest().fit({})
        self.assertTrue(index.is_fitted)
        self.assertEqual(len(index), 0)
        self.assertEqual(index.query("anything", k=3), [])

    def test_all_empty_corpus_has_no_sketches(self):
        index = TFIDFWeightedMinHashLSHForest().fit({"a": "", "b": ""})
        self.assertEqual(len(index), 2)
        self.assertEqual(index.searchable_ids_, [])
        self.assertEqual(index.query("anything", k=3), [])

    def test_unhashable_tokens_rejected(self):
        with self.assertRaises(TypeError):
            TFIDFWeightedMinHashLSHForest().fit([["ok", ["nested"]]])


class TestTransform(unittest.TestCase):
    def setUp(self):
        self.index = TFIDFWeightedMinHashLSHForest(num_perm=64, l=4).fit(
            {"a": "the quick brown fox", "b": "the quick brown dog", "c": "taxes"}
        )

    def test_transform_matches_weighted_minhash_of_same_weights(self):
        weights = self.index.token_weights("the quick fox")
        dimension = len(self.index.vocabulary_)
        vector = np.zeros(dimension, dtype=np.float64)
        for token, weight in weights.items():
            vector[self.index.vocabulary_[token]] = weight
        generator = WeightedMinHashGenerator(dimension, sample_size=64, seed=1)
        expected = generator.minhash(vector)
        sketch = self.index.transform("the quick fox")
        np.testing.assert_array_equal(sketch.hashvalues, expected.hashvalues)

    def test_transform_stored_sketch_equals_transform_of_same_text(self):
        self.assertEqual(self.index.transform("the quick brown fox"), self.index.get_sketch("a"))
        self.assertEqual(self.index.transform(["the", "quick", "brown", "fox"]), self.index.get_sketch("a"))

    def test_transform_empty_document_is_none(self):
        self.assertIsNone(self.index.transform(""))

    def test_transform_unknown_only_is_none(self):
        self.assertIsNone(self.index.transform("unseen words"))

    def test_transform_never_learns(self):
        vocabulary = dict(self.index.vocabulary_)
        idf = self.index.idf_.copy()
        self.index.transform("brand new words")
        self.index.query("brand new words", k=2)
        self.assertEqual(self.index.vocabulary_, vocabulary)
        np.testing.assert_array_equal(self.index.idf_, idf)


class TestQuery(unittest.TestCase):
    def test_exact_reranking_matches_brute_force(self):
        documents = {
            "a": "the quick brown fox",
            "b": "the quick brown dog",
            "c": "the quick red fox",
            "d": "a slow green turtle",
            "e": "",
        }
        index = TFIDFWeightedMinHashLSHForest(num_perm=256, l=8, candidate_multiplier=100).fit(documents)
        query = "the quick blue fox"
        results = index.query(query, k=10)
        query_weights = index.token_weights(query)
        reference = sorted(
            (
                (document_id, _weighted_jaccard(query_weights, index.token_weights(text)))
                for document_id, text in documents.items()
                if index.token_weights(text)
            ),
            key=lambda item: -item[1],
        )
        self.assertEqual(
            [document_id for document_id, _ in results],
            [document_id for document_id, _ in reference[: len(results)]],
        )
        for (_, score), (_, expected) in zip(results, reference):
            self.assertAlmostEqual(score, expected)

    def test_results_are_sorted_and_bounded(self):
        documents = {i: "common token %d" % (i % 7) for i in range(50)}
        index = TFIDFWeightedMinHashLSHForest(num_perm=128, l=8, candidate_multiplier=4).fit(documents)
        results = index.query("common token 3", k=5)
        self.assertLessEqual(len(results), 5)
        scores = [score for _, score in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        for document_id, _ in results:
            self.assertIn(document_id, index)

    def test_ties_follow_corpus_order(self):
        documents = {"first": "alpha beta", "second": "alpha beta", "third": "alpha beta"}
        index = TFIDFWeightedMinHashLSHForest(num_perm=128, l=8, candidate_multiplier=10).fit(documents)
        self.assertEqual(
            [document_id for document_id, _ in index.query("alpha beta", k=3)], ["first", "second", "third"]
        )

    def test_identical_document_is_returned_first(self):
        index = TFIDFWeightedMinHashLSHForest(num_perm=128, l=8).fit(
            {"a": "one two three", "b": "one two three four", "c": "five six seven"}
        )
        results = index.query("one two three", k=3)
        self.assertEqual(results[0][0], "a")
        self.assertEqual(results[0][1], 1.0)

    def test_unknown_tokens_in_query_are_ignored(self):
        index = TFIDFWeightedMinHashLSHForest(num_perm=128, l=8).fit({"a": "alpha beta", "b": "gamma delta"})
        self.assertEqual(index.query("alpha beta", k=2), index.query("alpha beta unseen words", k=2))

    def test_query_with_no_known_token_is_empty(self):
        index = TFIDFWeightedMinHashLSHForest().fit({"a": "alpha beta"})
        self.assertEqual(index.query("unknown", k=5), [])
        self.assertEqual(index.query("", k=5), [])

    def test_empty_documents_are_never_returned(self):
        index = TFIDFWeightedMinHashLSHForest(num_perm=128, l=8, candidate_multiplier=20).fit(
            {"a": "alpha beta", "empty": "", "other": ""}
        )
        self.assertEqual([document_id for document_id, _ in index.query("alpha beta", k=10)], ["a"])

    def test_k_must_be_positive(self):
        index = TFIDFWeightedMinHashLSHForest().fit({"a": "alpha"})
        with self.assertRaises(ValueError):
            index.query("alpha", k=0)

    def test_candidate_forest_is_approximate_but_consistent(self):
        # The public score must be the exact weighted Jaccard of the indexed
        # weights, so it cannot exceed 1 and cannot be negative.
        index = TFIDFWeightedMinHashLSHForest(num_perm=128, l=8).fit({"a": "x x y", "b": "x y z", "c": "z z z"})
        for _, score in index.query("x x y", k=3):
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestRefresh(unittest.TestCase):
    def setUp(self):
        self.index = TFIDFWeightedMinHashLSHForest(num_perm=64, l=4, seed=1).fit(
            {"a": "alpha beta", "b": "alpha gamma", "gone": "removed document"}
        )

    def test_refresh_rebuilds_statistics(self):
        old_idf = self.index.idf_.copy()
        old_sketch = self.index.get_sketch("a")
        # A different corpus size changes every IDF, so every stored sketch
        # the remaining documents had must be rebuilt.
        self.index.refresh({"a": "alpha beta", "b": "alpha gamma", "c": "delta epsilon", "d": "beta delta"})
        self.assertEqual(self.index.corpus_ids_, ["a", "b", "c", "d"])
        self.assertNotIn("gone", self.index)
        self.assertNotEqual(self.index.get_sketch("a"), old_sketch)
        self.assertFalse(np.array_equal(self.index.idf_, old_idf))

    def test_scores_use_refreshed_statistics(self):
        self.index.refresh({"a": "alpha beta", "b": "alpha beta", "c": "gamma"})
        # IDF of "alpha"/"beta" dropped from ln(4/3)+1 to ln(4/3?) -- recompute
        # directly from the refreshed tables instead of hard-coding.
        expected = float(np.log((1 + 3) / (1 + 2)) + 1)
        self.assertAlmostEqual(float(self.index.idf_[self.index.vocabulary_["alpha"]]), expected)
        top = self.index.query("alpha beta", k=2)
        self.assertEqual([document_id for document_id, _ in top], ["a", "b"])
        for _, score in top:
            self.assertAlmostEqual(score, 1.0)

    def test_query_transform_tracks_refresh(self):
        self.index.refresh({"a": "alpha beta", "b": "gamma"})
        self.assertAlmostEqual(self.index.token_weights("alpha")["alpha"], float(self.index.idf_[0]))
        self.assertIsNone(self.index.transform("removed document"))
        self.assertEqual(self.index.query("removed document", k=2), [])

    def test_refresh_is_atomic_on_failure(self):
        vocabulary = dict(self.index.vocabulary_)
        idf = self.index.idf_.copy()
        with self.assertRaises(ValueError):
            self.index.refresh(["alpha"], ids=["a", "b"])
        self.assertEqual(self.index.vocabulary_, vocabulary)
        np.testing.assert_array_equal(self.index.idf_, idf)
        self.assertEqual(self.index.query("alpha beta", k=1)[0][0], "a")

    def test_refresh_keeps_searchable_after_intermediate_empty_corpus(self):
        self.index.refresh({})
        self.assertEqual(len(self.index), 0)
        self.index.refresh({"a": "alpha beta"})
        self.assertEqual(self.index.query("alpha beta", k=1)[0][0], "a")


class TestValidationAndAccessors(unittest.TestCase):
    def test_constructor_validation(self):
        with self.assertRaises(ValueError):
            TFIDFWeightedMinHashLSHForest(num_perm=0)
        with self.assertRaises(ValueError):
            TFIDFWeightedMinHashLSHForest(num_perm=8, l=9)
        with self.assertRaises(ValueError):
            TFIDFWeightedMinHashLSHForest(l=0)
        with self.assertRaises(ValueError):
            TFIDFWeightedMinHashLSHForest(candidate_multiplier=0)
        with self.assertRaises(TypeError):
            TFIDFWeightedMinHashLSHForest(tokenizer="not callable")

    def test_unfitted_operations_raise(self):
        index = TFIDFWeightedMinHashLSHForest()
        for call in (
            lambda: index.query("x"),
            lambda: index.transform("x"),
            lambda: index.token_weights("x"),
            lambda: index.get_sketch("x"),
            lambda: index.forest,
        ):
            with self.assertRaises(NotFittedError):
                call()
        # NotFittedError is catchable as either ValueError or AttributeError.
        with self.assertRaises(ValueError):
            index.query("x")
        with self.assertRaises(AttributeError):
            index.query("x")

    def test_len_and_contains(self):
        index = TFIDFWeightedMinHashLSHForest().fit({"a": "x", "b": ""})
        self.assertEqual(len(index), 2)
        self.assertIn("a", index)
        self.assertIn("b", index)
        self.assertNotIn("missing", index)

    def test_alias_classes_are_the_same(self):
        self.assertIs(TFIDFMinHashLSHForest, TFIDFWeightedMinHashLSHForest)
        self.assertIs(TFIDFWeightedMinHash, TFIDFWeightedMinHashLSHForest)

    def test_pickle_round_trip(self):
        index = TFIDFWeightedMinHashLSHForest(num_perm=64, l=4, seed=3).fit(
            {"a": "alpha beta", "b": "alpha gamma", "c": ""}
        )
        restored = pickle.loads(pickle.dumps(index))
        self.assertEqual(restored.vocabulary_, index.vocabulary_)
        np.testing.assert_array_equal(restored.idf_, index.idf_)
        self.assertEqual(restored.corpus_ids_, index.corpus_ids_)
        before = index.query("alpha beta", k=2)
        after = restored.query("alpha beta", k=2)
        self.assertEqual(before, after)

    def test_repr(self):
        self.assertIn("unfitted", repr(TFIDFWeightedMinHashLSHForest()))
        index = TFIDFWeightedMinHashLSHForest().fit({"a": "x y"})
        self.assertIn("documents=1", repr(index))
        self.assertIn("vocabulary=2", repr(index))


if __name__ == "__main__":
    unittest.main()
