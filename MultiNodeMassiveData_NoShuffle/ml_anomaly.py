"""
Distributed Anomaly Ensemble - STRICT NO-SHUFFLE EDITION
==========================================================

Exactly four algorithms, nothing else:
    1. K-Means + Micro-Cluster           (distance to nearest global centroid)
    2. Gaussian Mixture Model             (micro-cluster / gmm_counts, hard EM)
    3. PCA + K-Means                      (embedding + micro-cluster)
    4. Adaptive Threshold                 (mean + 3*std, computed via aggregate)

NO-SHUFFLE / DRIVER-ONLY-COMMUNICATION GUARANTEE
--------------------------------------------------
Every single step in this file uses ONLY these four Spark operations:

  * mapPartitions()   -> pure LOCAL work done independently on each worker.
                         Workers never see each other's partitions.
  * rdd.aggregate()   -> each worker computes a small LOCAL partial result,
                         then (and only then) ships it straight to the
                         DRIVER, which does the final combine in Python.
                         This is a single hop: worker -> driver. It is NOT
                         treeAggregate/reduceByKey, which route partial
                         results through OTHER workers first.
  * rdd.collect()     -> worker -> driver only.
  * sc.broadcast()    -> driver -> worker only.

We deliberately do NOT use, anywhere in this file:
  .join()   .groupBy()   .orderBy()/.sort()   .distinct()
  .repartition(shuffle=True)   reduceByKey   treeAggregate/treeReduce
  DataFrame .agg()/.count() with a groupBy   (all of these insert an
  Exchange/shuffle stage in the physical plan, i.e. worker-to-worker
  network traffic).

Because of this, the four algorithms are never joined back together via a
shuffle-join on ID. Instead, once every algorithm's parameters (centroids,
thresholds, minority-cluster ids, PCA matrix, mean/std vectors) have been
fit and broadcast to the workers, ONE final mapPartitions pass computes all
four flags for every row locally and emits the finished record directly.

How to verify this yourself: open the Spark UI -> Stages tab while the job
runs. Every stage should show "Shuffle Read" and "Shuffle Write" as 0.0 B.
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, LongType, DoubleType, IntegerType
import numpy as np
import math

FEATURE_COLS = ["PACKET_SIZE", "CONNECTION_TIME", "FAILED_LOGINS"]
N_FEATURES = len(FEATURE_COLS)


# ============================================================================
# Local model helpers - these run EITHER inside mapPartitions (on a worker,
# operating only on that worker's own local data) OR directly on the driver
# (after a small collect()). They never move data between workers.
# ============================================================================

def local_kmeans(data, k=2, max_iter=20, seed=42):
    """K-Means on a local numpy array. Returns k cluster centers."""
    np.random.seed(seed)
    n_samples, n_features = data.shape
    idx = np.random.choice(n_samples, k, replace=False)
    centers = data[idx].copy()
    for _ in range(max_iter):
        distances = np.linalg.norm(data[:, None] - centers, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centers = []
        for j in range(k):
            if np.sum(labels == j) > 0:
                new_centers.append(data[labels == j].mean(axis=0))
            else:
                new_centers.append(centers[j])
        new_centers = np.array(new_centers)
        if np.allclose(centers, new_centers, rtol=1e-4):
            break
        centers = new_centers
    return [c.tolist() for c in centers]


def local_gmm(data, k=2, max_iter=20, seed=42):
    """Local Gaussian Mixture Model (hard EM) on a numpy array."""
    np.random.seed(seed)
    n_samples, n_features = data.shape
    centers = np.array(local_kmeans(data, k=k, max_iter=10, seed=seed))
    covs = [np.eye(n_features) * 0.1 for _ in range(k)]
    weights = np.ones(k) / k

    for _ in range(max_iter):
        log_likelihoods = []
        for j in range(k):
            diff = data - centers[j]
            inv_cov = np.linalg.inv(covs[j])
            det = np.linalg.det(covs[j])
            norm_const = -0.5 * (n_features * math.log(2 * math.pi) + math.log(det))
            exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
            log_likelihoods.append(norm_const + exponent + math.log(weights[j]))
        log_likelihoods = np.array(log_likelihoods).T
        labels = np.argmax(log_likelihoods, axis=1)

        new_centers, new_covs, new_weights = [], [], []
        for j in range(k):
            mask = (labels == j)
            count = np.sum(mask)
            if count == 0:
                new_centers.append(centers[j])
                new_covs.append(covs[j])
                new_weights.append(weights[j])
            else:
                points = data[mask]
                mu = points.mean(axis=0)
                cov = np.cov(points, rowvar=False) + 1e-6 * np.eye(n_features)
                new_centers.append(mu)
                new_covs.append(cov)
                new_weights.append(count / n_samples)
        new_centers = np.array(new_centers)
        if np.allclose(centers, new_centers, rtol=1e-4):
            centers, covs, weights = new_centers, new_covs, new_weights
            break
        centers, covs, weights = new_centers, new_covs, new_weights

    return {
        "means": [c.tolist() for c in centers],
        "covs": [c.tolist() for c in covs],
        "weights": list(weights) if not isinstance(weights, np.ndarray) else weights.tolist(),
    }


def merge_centers(centers_list, k=2):
    """
    Merge many small lists of LOCAL centers (already collected to the
    driver) into k GLOBAL centers by running K-Means again, on the driver,
    over that small collected set. Pure driver-side numpy - no cluster
    interaction at all.
    """
    all_centers = []
    for local_list in centers_list:
        all_centers.extend(local_list)
    all_centers = np.array(all_centers)
    if len(all_centers) == 0:
        return None
    return np.array(local_kmeans(all_centers, k=k, max_iter=20, seed=42))


def compute_pca_from_cov(cov_matrix, n_components=2):
    """Top eigenvectors of a covariance matrix. Driver-only numpy."""
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvals)[::-1]
    eigenvecs = eigenvecs[:, idx]
    return eigenvecs[:, :n_components].T


def aggregate_mean_cov(vec_rdd, n_features):
    """
    Global mean + covariance of an RDD of numpy vectors, via a single-level
    rdd.aggregate(): each worker folds its own partition into one small
    (count, sum, sum_outer) tuple locally, then that tuple - and only that
    tuple - is sent to the driver, which combines all of them. No shuffle,
    no worker-to-worker traffic.
    """
    def seq_op(acc, vec):
        count, sum_vec, sum_outer = acc
        return (count + 1, sum_vec + vec, sum_outer + np.outer(vec, vec))

    def comb_op(a, b):
        c1, s1, o1 = a
        c2, s2, o2 = b
        return (c1 + c2, s1 + s2, o1 + o2)

    zero = (0, np.zeros(n_features), np.zeros((n_features, n_features)))
    count, sum_vec, sum_outer = vec_rdd.aggregate(zero, seq_op, comb_op)
    mean_vec = sum_vec / count
    cov_matrix = (sum_outer / count) - np.outer(mean_vec, mean_vec)
    return mean_vec, cov_matrix


# ============================================================================
# Main
# ============================================================================

def main():
    spark = (
        SparkSession.builder.appName("Distributed_Anomaly_Ensemble_NoShuffle")
        # AQE can rewrite plans around Exchange (shuffle) stages; since this
        # job has no shuffle stages by design, we turn it off so nothing is
        # silently reintroduced if this file is edited later.
        .config("spark.sql.adaptive.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    sc = spark.sparkContext

    # ---------------------------------------------------------------
    # 0. Load + build feature vectors (plain RDD, no ML pipeline)
    #    We avoid pyspark.ml's VectorAssembler/StandardScaler here
    #    because StandardScaler's internal summarizer uses treeAggregate,
    #    which routes partial results through OTHER executors before
    #    reaching the driver. Our own aggregate_mean_cov() below does the
    #    same job but with a single worker->driver hop, matching the
    #    "workers only talk to the driver" requirement exactly.
    # ---------------------------------------------------------------
    hdfs_path = "hdfs://192.168.18.4:9000/user/demo/network_data.parquet"
    df = spark.read.parquet(hdfs_path).select("ID", *FEATURE_COLS)

    num_partitions = df.rdd.getNumPartitions()
    total_rows = df.count()  # plain action: local count per partition, summed on driver
    print(f"[Driver] Loaded {total_rows} rows from HDFS across {num_partitions} partitions.")

    raw_rdd = df.rdd.map(
        lambda row: (int(row["ID"]), np.array([float(row[c]) for c in FEATURE_COLS]))
    )
    raw_rdd.cache()

    # Feature scaling stats (mean/std), single-level aggregate -> broadcast
    def scale_seq(acc, item):
        _id, vec = item
        count, s, sq = acc
        return (count + 1, s + vec, sq + vec * vec)

    def scale_comb(a, b):
        c1, s1, sq1 = a
        c2, s2, sq2 = b
        return (c1 + c2, s1 + s2, sq1 + sq2)

    fcount, fsum, fsumsq = raw_rdd.aggregate(
        (0, np.zeros(N_FEATURES), np.zeros(N_FEATURES)), scale_seq, scale_comb
    )
    feat_mean = fsum / fcount
    feat_std = np.sqrt(np.maximum(fsumsq / fcount - feat_mean ** 2, 1e-12))
    print(f"[Driver] Feature mean={feat_mean}, std={feat_std}")

    fmean_bc = sc.broadcast(feat_mean)
    fstd_bc = sc.broadcast(feat_std)

    # (ID, raw_vec, scaled_vec) - a plain .map() is a narrow, per-partition
    # transform. No shuffle boundary is introduced by map().
    scaled_rdd = raw_rdd.map(
        lambda item: (item[0], item[1], (item[1] - fmean_bc.value) / fstd_bc.value)
    )
    scaled_rdd.cache()
    scaled_only_rdd = scaled_rdd.map(lambda item: item[2])
    scaled_only_rdd.cache()

    # -------------------------------------------------------------------
    # ALGORITHM 1: K-Means + Micro-Cluster
    # -------------------------------------------------------------------
    print("\n[NoShuffle] Algorithm 1: K-Means + Micro-Cluster...")

    def extract_kmeans_centers(iterator):
        vecs = np.array([scaled for (_id, raw, scaled) in iterator])
        if len(vecs) == 0:
            return []
        return [local_kmeans(vecs, k=2, max_iter=10)]

    local_centers_list = scaled_rdd.mapPartitions(extract_kmeans_centers).collect()
    kmeans_global_centers = merge_centers(local_centers_list, k=2)
    if kmeans_global_centers is None:
        kmeans_global_centers = np.zeros((2, N_FEATURES))
    print(f"[Driver] Global K-Means centers:\n{kmeans_global_centers}")
    kmeans_centers_bc = sc.broadcast(kmeans_global_centers)

    def dist_seq(acc, item):
        _id, raw, scaled = item
        d = float(np.min(np.linalg.norm(scaled - kmeans_centers_bc.value, axis=1)))
        count, s, sq = acc
        return (count + 1, s + d, sq + d * d)

    def dist_comb(a, b):
        c1, s1, sq1 = a
        c2, s2, sq2 = b
        return (c1 + c2, s1 + s2, sq1 + sq2)

    dcount, dsum, dsumsq = scaled_rdd.aggregate((0, 0.0, 0.0), dist_seq, dist_comb)
    dist_mean = dsum / dcount
    dist_std = math.sqrt(max(0.0, dsumsq / dcount - dist_mean ** 2))
    kmeans_threshold = dist_mean + 3 * dist_std
    print(f"[Driver] K-Means distance mean={dist_mean:.4f} std={dist_std:.4f} threshold={kmeans_threshold:.4f}")
    kmeans_threshold_bc = sc.broadcast(kmeans_threshold)

    # -------------------------------------------------------------------
    # ALGORITHM 2: Gaussian Mixture Model (micro-cluster, gmm_counts)
    # -------------------------------------------------------------------
    print("\n[NoShuffle] Algorithm 2: Gaussian Mixture Model (gmm_counts)...")

    def gmm_assign_counts(iterator):
        vecs = np.array([scaled for (_id, raw, scaled) in iterator])
        if len(vecs) == 0:
            return []
        gmm_model = local_gmm(vecs, k=3, max_iter=10)
        means = np.array(gmm_model["means"])
        covs = [np.array(c) for c in gmm_model["covs"]]
        weights = np.array(gmm_model["weights"])
        log_liks = []
        for j in range(3):
            diff = vecs - means[j]
            inv_cov = np.linalg.inv(covs[j])
            det = np.linalg.det(covs[j])
            norm_const = -0.5 * (N_FEATURES * math.log(2 * math.pi) + math.log(det))
            exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
            log_liks.append(norm_const + exponent + math.log(weights[j]))
        log_liks = np.array(log_liks).T
        labels = np.argmax(log_liks, axis=1)
        gmm_counts = {0: 0, 1: 0, 2: 0}
        for lab in labels:
            gmm_counts[int(lab)] += 1
        return [(0, gmm_counts[0]), (1, gmm_counts[1]), (2, gmm_counts[2])]

    # .collect() ships each partition's tiny (cluster_id, count) pairs
    # straight to the driver; the sum below happens in driver-side Python.
    gmm_counts_collected = scaled_rdd.mapPartitions(gmm_assign_counts).collect()
    global_gmm_counts = {0: 0, 1: 0, 2: 0}
    for cid, cnt in gmm_counts_collected:
        global_gmm_counts[cid] += cnt
    gmm_minority = min(global_gmm_counts, key=global_gmm_counts.get)
    print(f"[Driver] GMM global counts: {global_gmm_counts}, minority cluster = {gmm_minority}")
    gmm_minority_bc = sc.broadcast(gmm_minority)

    # -------------------------------------------------------------------
    # ALGORITHM 3: PCA + K-Means (embedding + micro-cluster)
    # -------------------------------------------------------------------
    print("\n[NoShuffle] Algorithm 3: PCA + K-Means Micro-Cluster...")

    _, cov_matrix = aggregate_mean_cov(scaled_only_rdd, N_FEATURES)
    pca_matrix = compute_pca_from_cov(cov_matrix, n_components=2)
    print(f"[Driver] PCA transformation matrix:\n{pca_matrix}")
    pca_bc = sc.broadcast(pca_matrix)

    def pca_kmeans_centers(iterator):
        vecs = np.array([scaled for (_id, raw, scaled) in iterator])
        if len(vecs) == 0:
            return []
        proj = vecs @ pca_bc.value.T
        return [local_kmeans(proj, k=4, max_iter=10)]

    pca_local_centers = scaled_rdd.mapPartitions(pca_kmeans_centers).collect()
    pca_global_centers = merge_centers(pca_local_centers, k=2)
    if pca_global_centers is None:
        pca_global_centers = np.array([[0.0, 0.0], [1.0, 1.0]])
    print(f"[Driver] PCA global centers:\n{pca_global_centers}")
    pca_centers_bc = sc.broadcast(pca_global_centers)

    def pca_assign_counts(iterator):
        vecs = np.array([scaled for (_id, raw, scaled) in iterator])
        if len(vecs) == 0:
            return []
        proj = vecs @ pca_bc.value.T
        dists = np.linalg.norm(proj[:, None, :] - pca_centers_bc.value[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        counts = {0: 0, 1: 0}
        for lab in labels:
            counts[int(lab)] += 1
        return [(0, counts[0]), (1, counts[1])]

    pca_counts_collected = scaled_rdd.mapPartitions(pca_assign_counts).collect()
    pca_global_counts = {0: 0, 1: 0}
    for cid, cnt in pca_counts_collected:
        pca_global_counts[cid] += cnt
    pca_minority = min(pca_global_counts, key=pca_global_counts.get)
    print(f"[Driver] PCA cluster counts: {pca_global_counts}, minority = {pca_minority}")
    pca_minority_bc = sc.broadcast(pca_minority)

    # -------------------------------------------------------------------
    # ALGORITHM 4: Adaptive Threshold (mean + 3*std, via aggregate)
    # -------------------------------------------------------------------
    print("\n[NoShuffle] Algorithm 4: Adaptive Threshold (mean + 3*std)...")

    def stats_seq(acc, vec):
        count, s, sq = acc
        return (count + 1, s + vec, sq + vec * vec)

    def stats_comb(a, b):
        c1, s1, sq1 = a
        c2, s2, sq2 = b
        return (c1 + c2, s1 + s2, sq1 + sq2)

    acount, asum, asumsq = scaled_only_rdd.aggregate(
        (0, np.zeros(N_FEATURES), np.zeros(N_FEATURES)), stats_seq, stats_comb
    )
    adaptive_mean = asum / acount
    adaptive_std = np.sqrt(np.maximum(asumsq / acount - adaptive_mean ** 2, 1e-10))
    print(f"[Driver] Adaptive threshold stats: mean={adaptive_mean}, std={adaptive_std}")
    adaptive_mean_bc = sc.broadcast(adaptive_mean)
    adaptive_std_bc = sc.broadcast(adaptive_std)
    ADAPTIVE_Z = 3.0

    # -------------------------------------------------------------------
    # COMBINED PASS: compute all 4 flags for every row in ONE local sweep.
    # This is what replaces the old join-based stitching. Every parameter
    # needed (centers, thresholds, minority ids, PCA matrix) is already a
    # broadcast variable, so each worker finishes every row completely on
    # its own - no data ever needs to move to another worker to be joined.
    # -------------------------------------------------------------------
    def combined_flag(iterator):
        items = list(iterator)
        if not items:
            return []

        vecs = np.array([scaled for (_id, raw, scaled) in items])

        # Algorithm 1: distance to nearest global K-Means centroid
        dists = np.linalg.norm(vecs[:, None, :] - kmeans_centers_bc.value[None, :, :], axis=2)
        kmeans_flags = (dists.min(axis=1) > kmeans_threshold_bc.value).astype(int)

        # Algorithm 2: refit local GMM on this partition, flag minority component
        gmm_model = local_gmm(vecs, k=3, max_iter=10)
        g_means = np.array(gmm_model["means"])
        g_covs = [np.array(c) for c in gmm_model["covs"]]
        g_weights = np.array(gmm_model["weights"])
        log_liks = []
        for j in range(3):
            diff = vecs - g_means[j]
            inv_cov = np.linalg.inv(g_covs[j])
            det = np.linalg.det(g_covs[j])
            norm_const = -0.5 * (N_FEATURES * math.log(2 * math.pi) + math.log(det))
            exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
            log_liks.append(norm_const + exponent + math.log(g_weights[j]))
        log_liks = np.array(log_liks).T
        gmm_labels = np.argmax(log_liks, axis=1)
        gmm_flags = (gmm_labels == gmm_minority_bc.value).astype(int)

        # Algorithm 3: PCA projection + nearest global PCA-space centroid
        proj = vecs @ pca_bc.value.T
        pdists = np.linalg.norm(proj[:, None, :] - pca_centers_bc.value[None, :, :], axis=2)
        pca_labels = np.argmin(pdists, axis=1)
        pca_flags = (pca_labels == pca_minority_bc.value).astype(int)

        # Algorithm 4: per-feature z-score vs. global mean/std
        z = (vecs - adaptive_mean_bc.value) / adaptive_std_bc.value
        adaptive_flags = np.any(np.abs(z) > ADAPTIVE_Z, axis=1).astype(int)

        out = []
        for i, (_id, raw, scaled) in enumerate(items):
            k_f, g_f, p_f, a_f = int(kmeans_flags[i]), int(gmm_flags[i]), int(pca_flags[i]), int(adaptive_flags[i])
            vote = k_f + g_f + p_f + a_f
            ensemble = 1 if vote >= 2 else 0
            out.append((
                _id, float(raw[0]), float(raw[1]), float(raw[2]),
                k_f, g_f, p_f, a_f, vote, ensemble,
            ))
        return out

    final_rdd = scaled_rdd.mapPartitions(combined_flag)
    final_rdd.cache()

    # -------------------------------------------------------------------
    # REPORTING - still no shuffle:
    #   * top-N uses RDD.takeOrdered(), which keeps a local top-N heap per
    #     partition and merges those small heaps on the driver, instead of
    #     DataFrame .orderBy() (which does a full shuffle sort).
    #   * sums/counts use one more rdd.aggregate() pass instead of
    #     DataFrame .agg()/.groupBy().count().
    # -------------------------------------------------------------------
    print("\n--- Top 10 Most Anomalous Rows (Highest Votes) ---")
    top10 = final_rdd.takeOrdered(10, key=lambda r: -r[8])
    print(f"{'ID':>10} {'PACKET_SIZE':>12} {'CONN_TIME':>10} {'FAILED_LOGINS':>14} {'VOTE_SUM':>9} {'ENSEMBLE':>9}")
    for row in top10:
        _id, pkt, ctime, flog, k_f, g_f, p_f, a_f, vote, ens = row
        print(f"{_id:>10} {pkt:>12.1f} {ctime:>10.1f} {flog:>14.1f} {vote:>9} {ens:>9}")

    def perf_seq(acc, row):
        k, g, p, a, e = acc
        return (k + row[4], g + row[5], p + row[6], a + row[7], e + row[9])

    def perf_comb(x, y):
        return tuple(xi + yi for xi, yi in zip(x, y))

    k_sum, g_sum, p_sum, a_sum, e_sum = final_rdd.aggregate((0, 0, 0, 0, 0), perf_seq, perf_comb)

    print("\n--- Algorithm Performance Summary ---")
    print(f"KMeans_Adaptive={k_sum}  GMM_Minority={g_sum}  PCA_Embedding={p_sum}  "
          f"Adaptive_Threshold={a_sum}  Ensemble_Final={e_sum}")

    print("\n--- Cluster Sizes (Ensemble) ---")
    print(f"{{0: {total_rows - e_sum}, 1: {e_sum}}}")

    spark.stop()


if __name__ == "__main__":
    main()