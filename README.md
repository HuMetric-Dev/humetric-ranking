# humetric-ranking

Two-stage reranking.

**Stage 1: LightGBM LambdaRank** over a small handful of hand features
(BM25 score, text cosine, graph cosine, skill overlap, log followers,
recency, history-centroid cosine). Trains in seconds on CPU.

**Stage 2: cross-encoder** (`BAAI/bge-reranker-base`) over the top-30,
GPU-accelerated. If the model checkpoint isn't on disk or torch lacks CUDA,
`rerank()` returns the input order unchanged so the runtime degrades
gracefully.
