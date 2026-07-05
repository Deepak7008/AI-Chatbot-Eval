"""
embeddings.py — Semantic similarity scoring using sentence embeddings.

Why this file exists:
  Cosine similarity is the "cheap first-pass" scorer in the eval cascade.
  It runs locally (zero API cost), takes milliseconds, and catches obvious
  mismatches before we spend LLM tokens on the expensive judge.

Model: all-MiniLM-L6-v2
  - 80MB, runs on CPU
  - 384-dimensional embeddings
  - ~100 sentences/sec
  - Free, local, no API key needed

Concept: Cosine Similarity
  Instead of comparing strings character-by-character, we convert text into
  dense vectors (embeddings) that capture *meaning*. Two sentences that say
  the same thing in different words will have vectors pointing in the same
  direction — their cosine similarity will be close to 1.0.

  Formula: cos(A, B) = (A · B) / (||A|| × ||B||)
  Range:   -1.0 (opposite) to 1.0 (identical meaning)
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# ── MODEL LOADING ─────────────────────────────────────────────────────────────
# The model is loaded once and cached for the lifetime of the process.
# First call downloads the model (~80MB) if not already cached locally.

_model = None

def get_model() -> SentenceTransformer:
    """
    Lazy-loads the sentence-transformers model.
    
    Why lazy-load?
      Importing this module shouldn't block app startup with a 2-second
      model load. The model only loads on the first call to cosine_similarity().
    """
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


# ── CORE API ──────────────────────────────────────────────────────────────────

def cosine_similarity(text_a: str, text_b: str) -> float:
    """
    Compute semantic similarity between two texts.
    
    Args:
        text_a: First text (e.g., the chatbot's actual answer)
        text_b: Second text (e.g., the reference answer)
        
    Returns:
        Float between -1.0 and 1.0, where:
          1.0  = identical meaning
          0.7+ = semantically similar (good match)
          0.3  = loosely related
          0.0  = unrelated
         -1.0  = opposite meaning (rare in practice)
         
    Edge cases:
        - Empty strings return 0.0 (can't compute meaningful similarity)
        - Identical strings return ~1.0 (not exactly 1.0 due to float precision)
    """
    # Guard: empty input is meaningless
    if not text_a or not text_b:
        return 0.0
    
    if not text_a.strip() or not text_b.strip():
        return 0.0
    
    model = get_model()
    
    # Encode both texts into 384-dimensional vectors
    # normalize_embeddings=True ensures unit vectors, so dot product = cosine similarity
    embeddings = model.encode(
        [text_a, text_b],
        normalize_embeddings=True,
        show_progress_bar=False
    )
    
    # With normalized vectors: cosine_similarity = dot product
    # This is faster than computing the full formula because ||A|| = ||B|| = 1
    similarity = float(np.dot(embeddings[0], embeddings[1]))
    
    # Clamp to [-1.0, 1.0] to handle floating point drift
    return max(-1.0, min(1.0, similarity))


def batch_cosine_similarity(pairs: list) -> list:
    """
    Compute cosine similarity for multiple text pairs efficiently.
    
    Why batch?
      Encoding 100 sentences at once is ~10x faster than encoding them
      one-by-one because the model can use GPU/CPU parallelism on a
      single forward pass.
    
    Args:
        pairs: List of (text_a, text_b) tuples
        
    Returns:
        List of similarity scores, one per pair
    """
    if not pairs:
        return []
    
    # Separate into two flat lists for batch encoding
    texts_a = []
    texts_b = []
    valid_indices = []
    
    for i, (a, b) in enumerate(pairs):
        if a and b and a.strip() and b.strip():
            texts_a.append(a)
            texts_b.append(b)
            valid_indices.append(i)
    
    # Pre-fill results with 0.0 for invalid pairs
    results = [0.0] * len(pairs)
    
    if not texts_a:
        return results
    
    model = get_model()
    
    # Batch encode all texts at once (much faster than one-by-one)
    embeddings_a = model.encode(texts_a, normalize_embeddings=True, show_progress_bar=False)
    embeddings_b = model.encode(texts_b, normalize_embeddings=True, show_progress_bar=False)
    
    # Compute pairwise dot products (= cosine similarity for normalized vectors)
    for j, idx in enumerate(valid_indices):
        sim = float(np.dot(embeddings_a[j], embeddings_b[j]))
        results[idx] = max(-1.0, min(1.0, sim))
    
    return results


# ── STANDALONE TEST ───────────────────────────────────────────────────────────
# Run: python -m evals.embeddings

if __name__ == "__main__":
    print("=" * 60)
    print("Embeddings Module — Verification Test")
    print("=" * 60)
    
    test_pairs = [
        # Test 1: Near-identical meaning → should be high (~0.85+)
        (
            "You can return items within 30 days of purchase.",
            "Our return window is 30 days from the date of delivery.",
        ),
        # Test 2: Same topic, different info → should be moderate (~0.5-0.7)
        (
            "We accept Visa, Mastercard, and PayPal.",
            "Our return policy allows exchanges within 14 days.",
        ),
        # Test 3: Completely unrelated → should be low (~0.0-0.3)
        (
            "Your order ORD-1042 has been shipped via express delivery.",
            "The weather in Paris is sunny with a high of 25 degrees.",
        ),
        # Test 4: Exact same text → should be ~1.0
        (
            "Hello, how can I help you today?",
            "Hello, how can I help you today?",
        ),
        # Test 5: Empty string → should be 0.0
        (
            "",
            "Some text here.",
        ),
    ]
    
    print("\nLoading model (first run downloads ~80MB)...")
    print()
    
    for i, (a, b) in enumerate(test_pairs, 1):
        score = cosine_similarity(a, b)
        print(f"Test {i}: {score:.4f}")
        print(f"  A: \"{a[:60]}{'...' if len(a) > 60 else ''}\"")
        print(f"  B: \"{b[:60]}{'...' if len(b) > 60 else ''}\"")
        print()
    
    # Also test batch version
    print("-" * 60)
    print("Batch test (should match individual results):")
    batch_scores = batch_cosine_similarity(test_pairs)
    for i, score in enumerate(batch_scores, 1):
        print(f"  Test {i}: {score:.4f}")
    
    print()
    print("[PASS] Embeddings module verified successfully!")
