"""
few_shot_retriever.py

Loads the few-shot example bank from few_shot_examples.json and retrieves
the top-k most relevant examples for a given instruction.

Two responsibilities:
  1. Loading: reads few_shot_examples.json and returns a clean list of
     example dicts.
  2. Retrieval: fits TF-IDF on all example instructions at startup, then
     ranks examples by cosine similarity to a query instruction and returns
     the top-k most relevant ones.

On top of pure top-k ranking, retrieve() applies two coverage guarantees:
  1. At least one REJECTION example (status != "ok") is always present in
     the retrieved set, so the LLM always sees how to handle instructions
     it should be refusing.
  2. At least one OK example (status == "ok") is always present in the
     retrieved set, so the LLM always sees what a valid plan looks like.

Reranking:
  After the initial TF-IDF + cosine retrieval, a cross-encoder reranker
  rescores the candidates more accurately by looking at the (query, example)
  pair jointly rather than independently. This catches cases where TF-IDF
  picked an example with overlapping keywords but mismatched meaning.
"""

import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder

# File path

EXAMPLES_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "knowledge_base",
    "few_shot_examples.json",
)

# Cross-encoder model for reranking. Lightweight and runs on CPU.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# Loading

def load_examples(filepath: str = EXAMPLES_FILE) -> list[dict]:
    """
    Load the few-shot bank from JSON. Returns a list of example dicts.

    Each returned dict has:
      - id          : unique example identifier
      - instruction : the natural language instruction
      - output      : the expected JSON output (status + subtasks or reason)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    examples = []
    for entry in raw["examples"]:
        examples.append({
            "id":          entry["id"],
            "instruction": entry["instruction"],
            "output":      entry["output"],
        })
    return examples


def bank_summary(examples: list[dict]) -> str:
    "One-line breakdown of the bank, for start-up logging."
    by_status = {}
    for ex in examples:
        status = ex["output"]["status"]
        by_status[status] = by_status.get(status, 0) + 1
    status_str = ", ".join(f"{n} {s}" for s, n in sorted(by_status.items()))
    return f"{len(examples)} examples ({status_str})"


# Retrieval

class CosineFewShotRetriever:
    def __init__(self, examples: list[dict], reranker_model: str = RERANKER_MODEL):
        """
        examples: list of dicts from load_examples().

        At startup:
          1. Fits TF-IDF vectorizer on all example instructions.
          2. Loads the cross-encoder reranker model.
        Both are done once and reused on every retrieve() call.
        """
        if not examples:
            raise ValueError("Few-shot bank is empty, nothing to retrieve from.")

        self.examples = examples

        # TF-IDF vectorizer fitted on all example instructions.
        # unigrams + bigrams catches "pick up", "navigate to" etc.
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words="english",
        )
        self.example_matrix = self.vectorizer.fit_transform(
            [ex["instruction"] for ex in examples]
        )

        # Cross-encoder reranker loaded once at startup.
        self.reranker = CrossEncoder(reranker_model)

    def _rank_by_tfidf(self, query_instruction: str) -> list[tuple[dict, float]]:
        """
        Ranks the entire bank by TF-IDF cosine similarity to the query.
        Returns (example, score) pairs, highest score first.
        This is the first stage of retrieval, fast but approximate.
        """
        query_vec = self.vectorizer.transform([query_instruction])
        scores = cosine_similarity(query_vec, self.example_matrix)[0]
        ranked = [
            (self.examples[i], float(scores[i]))
            for i in np.argsort(scores)[::-1]
        ]
        return ranked

    def _rerank(self, query_instruction: str,
                candidates: list[tuple[dict, float]]) -> list[tuple[dict, float]]:
        """
        Reranks candidates using the cross-encoder.

        The cross-encoder looks at (query, candidate instruction) as a pair
        rather than independently, much more accurate than TF-IDF cosine
        similarity for catching semantic matches.

        Returns (example, reranker_score) pairs, highest score first.
        This is the second stage of retrieval, slower but more accurate.
        """
        pairs = [
            (query_instruction, ex["instruction"])
            for ex, _ in candidates
        ]
        reranker_scores = self.reranker.predict(pairs)
        reranked = sorted(
            zip([ex for ex, _ in candidates], reranker_scores),
            key=lambda pair: -pair[1],
        )
        return [(ex, float(score)) for ex, score in reranked]

    def retrieve(self, query_instruction: str, k: int = 4,
                 min_score: float = 0.0,
                 guarantee_rejection: bool = True,
                 guarantee_ok: bool = True) -> list[tuple[dict, float]]:
        """
        Returns up to k (example, score) tuples, highest reranker score first.

        Two stage retrieval:
          Stage 1, TF-IDF cosine similarity ranks the full bank and pulls a
                   wider candidate pool (k * 3) cheaply.
          Stage 2, cross-encoder reranker rescores that pool more accurately
                   and we take the final top-k from it.

        Two coverage guarantees:
          1. guarantee_rejection, if the final top-k has no rejection example
             (status != "ok"), the weakest pick is swapped out for the best
             scoring rejection example from the reranked pool, so the LLM
             always sees at least one example of an instruction it should refuse.
          2. guarantee_ok, if the final top-k has no ok example (status == "ok"),
             the weakest pick is swapped out for the best scoring ok example
             from the reranked pool, so the LLM always sees what a valid
             plan looks like.

        min_score: applied to Stage 1 scores only, to avoid pulling completely
                   irrelevant candidates into the reranker. The guarantees
                   ignore min_score intentionally.
        """
        # Stage 1, TF-IDF retrieval of wider candidate pool
        full_ranking = self._rank_by_tfidf(query_instruction)
        candidate_pool = [
            (ex, s) for ex, s in full_ranking
            if s >= min_score
        ] or [full_ranking[0]]

        # Pull k*3 candidates into reranker so it has enough to work with
        # and the guarantees have candidates to swap in from.
        pool_size = min(k * 3, len(candidate_pool))
        candidates_for_reranking = candidate_pool[:pool_size]

        # Stage 2, rerank the candidate pool
        reranked = self._rerank(query_instruction, candidates_for_reranking)

        # Take top-k from reranked pool
        selected = reranked[:k]
        selected_ids = {ex["id"] for ex, _ in selected}
        guaranteed_ids = set()

        def swap_in_best(predicate) -> None:
            """
            Replace the weakest non-guaranteed pick with the best example
            from the reranked pool that satisfies predicate and is not
            already selected.
            """
            for ex, score in reranked:
                if ex["id"] in selected_ids:
                    continue
                if not predicate(ex):
                    continue
                # Found a matching example not yet selected.
                # Find weakest displaceable slot.
                displaceable = sorted(
                    (pair for pair in selected
                     if pair[0]["id"] not in guaranteed_ids),
                    key=lambda pair: pair[1],
                )
                if not displaceable:
                    return
                removed = displaceable[0]
                selected.remove(removed)
                selected_ids.discard(removed[0]["id"])
                selected.append((ex, score))
                selected_ids.add(ex["id"])
                guaranteed_ids.add(ex["id"])
                return

        def ensure_covered(predicate) -> None:
            """
            Make sure at least one selected example satisfies predicate.
            If one already does, protect it so nothing can displace it.
            If none does, swap in the best matching one from the reranked pool.
            """
            already_covered = next(
                (ex["id"] for ex, _ in selected if predicate(ex)), None
            )
            if already_covered is not None:
                guaranteed_ids.add(already_covered)
                return
            swap_in_best(predicate)

        if guarantee_rejection:
            ensure_covered(lambda ex: ex["output"]["status"] != "ok")

        if guarantee_ok:
            ensure_covered(lambda ex: ex["output"]["status"] == "ok")

        selected.sort(key=lambda pair: -pair[1])
        return selected


# Prompt formatting

def format_examples_for_prompt(retrieved: list[tuple[dict, float]],
                                indent: int = 2) -> str:
    """
    Renders retrieved examples into a prompt-ready text block.

    Synthesises plan_id and original_instruction on ok plans so every
    rendered example is a valid instance of the output schema the LLM
    is being asked to follow.
    """
    blocks = []
    for i, (ex, score) in enumerate(retrieved, 1):
        output = dict(ex["output"])
        if output["status"] == "ok":
            output = {
                "status":               "ok",
                "plan_id":              ex["id"],
                "original_instruction": ex["instruction"],
                "subtasks":             output.get("subtasks", []),
            }
        blocks.append(
            f"========== EXAMPLE {i} - {output['status']} ==========\n"
            f"Instruction: \"{ex['instruction']}\"\n"
            f"Output:\n{json.dumps(output, indent=indent)}\n"
        )
    return "\n".join(blocks)