from datasketch import TFIDFWeightedMinHashLSHForest

documents = {
    "doc-1": "the quick brown fox jumps over the lazy dog",
    "doc-2": "the quick brown fox leaps over a lazy dog",
    "doc-3": "an unrelated essay about tax policy",
    "doc-4": "tax policy and the quick brown fox",
    "doc-5": "",
}

index = TFIDFWeightedMinHashLSHForest(num_perm=128, l=8)
index.fit(documents)
print("documents: %d, vocabulary: %d" % (len(index), len(index.vocabulary_)))

# The fitted weights are raw term frequency * IDF; "the" is common and gets a
# much smaller weight than a rare term such as "fox" or "essay".
for term, weight in index.token_weights("the the essay").items():
    print("  %-6s %.4f" % (term, weight))

# transform() applies the fitted statistics without learning anything new.
print("sketch of an unseen document:", index.transform("never seen before"))

for document_id, score in index.query("the quick brown fox", k=3):
    print("  %s %.4f" % (document_id, score))

print("top-2 raw candidates:", index.forest.query(index.transform("the quick brown fox"), 2))

# Refreshing re-learns the vocabulary and IDF table on the new corpus and
# rebuilds every stored sketch, so scores always share one set of statistics.
index.refresh({"doc-1": "the quick brown fox", "doc-6": "the quick brown fox"})
print("after refresh:", index.corpus_ids_)
print("doc-3 searchable:", "doc-3" in index)
for document_id, score in index.query("the quick brown fox", k=2):
    print("  %s %.4f" % (document_id, score))
