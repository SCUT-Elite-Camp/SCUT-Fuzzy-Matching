# e-means replacement and A/B protocol

## Changes

- `clustering/kmeans_cosine.py` keeps the four-argument `run_cosine_kmeans` interface and `(centroids, assignments)` output. Its default algorithm is now e-means. Set `SCUT_CLUSTER_ALGORITHM=kmeans` to run the original implementation, retained verbatim as `run_legacy_cosine_kmeans`.
- The e-means branch directly instantiates FAISS 1.16.0's `faiss.contrib.e_means.NumpyEmeans(d, k, nepochs=iterations, seed=random_state, spherical=True)`. The previous hand-written EMA, batch-law and revival code has been removed. FAISS chooses its own algorithm parameters.
- Inputs are L2-normalized and converted to contiguous float32, as required for official spherical e-means. The returned centroids are exposed as float64 for the existing downstream interface; labels are recomputed with FAISS `assign` against the final centroids, as the baseline previously did. `iterations` remains full training passes, preserving the configured value of 20. Minor numerical differences from the former float64 implementation are expected.
- No matching, encryption, dataset, configuration, or downstream interfaces were changed.

## Run and verify

Use Python >=3.10 and install `requirements.txt` in a Python environment (`python -m pip install -r requirements.txt`). The official CPU API first ships in `faiss-cpu==1.16.0`; the requirement is constrained to the 1.16 series. Verify with `python -c "import faiss; from faiss.contrib.e_means import NumpyEmeans; print(faiss.__version__)"`. From `baseline_Integration`:

```powershell
python -m pytest tests/test_emeans.py tests/test_choose_k.py -q
$env:SCUT_CLUSTER_ALGORITHM = 'emeans'
python scripts/evaluate_dataset.py --config config/examples/ncvr_10k_evaluation.json --output-dir artifacts/ab/emeans
$env:SCUT_CLUSTER_ALGORITHM = 'kmeans'
python scripts/evaluate_dataset.py --config config/examples/ncvr_10k_evaluation.json --output-dir artifacts/ab/kmeans
```

On Windows with Python 3.12, official FAISS 1.16.0 was imported and the focused tests passed (7 passed). A wider test collection was blocked by the environment's missing `tenseal` dependency; it was not an e-means failure. The focused tests check output shapes, determinism, normalization, input validation, and control-arm availability.

## Fair A/B experiment

1. Freeze the same dataset files and their hashes, evaluation JSON, preprocessing, `k_mode`, projection settings, threshold, probing strategy, machine, and software environment. The `ncvr_10k_evaluation.json` configuration is a runnable evaluation example; a labeled fused multilingual dataset may also be evaluated through its existing multi-attribute runner.
2. Run the commands above for both settings with identical `random_state` values (for example 7, 13, 42). Save each run to a separate output location so reports are not overwritten. Repeat timings after one warm-up run. The only experimental switch is `SCUT_CLUSTER_ALGORITHM`.
3. Record clustering time, full pipeline time, number of epochs, cosine distortion `mean(1 - cosine(x_i,c_label))`, Euclidean MSE `mean(||x_i-c_label||²)` on the identical standardized features, and cluster size distribution including empty clusters. Report training and held-out values when a split is available.
4. From the same labeled query set, report candidate coverage (fraction of ground-truth matches present after candidate generation), candidate count per query, final recall and precision, and Bad Event rate using the project's existing definition. Keep denominators and treatment of unmatched queries explicit. Report candidate-generation time separately from end-to-end time.
5. Record serialized message bytes sent and received, ciphertext count/size, and total communication using the existing `evaluation/benchmark.py` counters. Check whether cluster size and `max_size` changes alter encrypted work, even with identical `k`.
6. Report per-seed paired differences, their median and range, and raw results. Treat quality, runtime, and communication as separate outcomes. Do not infer a benefit from clustering distortion alone or from this implementation's passing tests.

Paper: https://arxiv.org/abs/2611.09501 . Official API: https://github.com/facebookresearch/faiss/tree/v1.16.0/contrib/e_means . Release: https://github.com/facebookresearch/faiss/releases/tag/v1.16.0 . The spherical objective differs from the paper's Euclidean MSE experiments, so performance gains must be measured in this baseline.
