"""
common.py - shared code for the split-up Distributed Anomaly Ensemble.

WHY THIS FILE EXISTS
=====================
The original ml_anomaly.py was a single spark-submit job that ran all
14 anomaly-detection algorithms back-to-back. If algorithm #9 (GAN)
crashed, Airflow's only retry option was "re-run everything from
algorithm #1", including several hours of KMeans/GMM/PCA/DBSCAN/
IsolationForest/AE work that had already succeeded.

This package splits that one job into one task per algorithm (plus a
few shared prep/shard/report tasks) so each can be retried in Airflow
independently. Every task is its own `spark-submit ... task_x.py` call;
this module holds everything they have in common: config, the 8 torch
model classes, the 3 generic DDP training harnesses, the pandas_udf
factories, the DBSCAN/IsolationForest per-partition helpers, the RL
threshold tuner, and small helpers for passing data between tasks via
HDFS (since separate spark-submit calls don't share memory).
# #!/bin/bash
# set -e

# echo "Starting containers..."
# sudo docker-compose up -d

# echo "Waiting for Oracle..."
# sleep 15

# echo "Copying CSV into Oracle container..."
# sudo docker-compose exec -T oracle bash -c 'cat > /tmp/dcs_test.csv' < dcs_test.csv

# echo "Initializing Oracle table (drop + recreate)..."
# cat init.sql | sudo docker-compose exec -T oracle sqlplus system/demo_password@localhost/XE

# echo "Running ETL job..."
# python etl_job.py

# echo "Submitting anomaly detection to Spark..."
# sudo docker-compose exec spark-master /opt/spark/bin/spark-submit \
#   --master spark://localhost:7077 \
#   --conf spark.ui.port=4040 \
#   --conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension \
#   --conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog \
#   /app/ml_anomaly.py

# echo "Pipeline completed."
IMPORTANT - DEPLOYMENT NOTE (read before adding a new task):
Because the model classes below now live in `common.py` instead of a
task script's own `__main__`, cloudpickle will try to pickle any
closure that references them (mapInPandas functions, broadcast
variables holding a class reference, etc.) BY REFERENCE - i.e. it will
tell each executor "import common and look up TorchAE" rather than
shipping the class definition inline the way it could when everything
lived in one __main__ script. That means every spark-submit call in
the DAG MUST include `--py-files /app/common.py` so executors can
actually resolve `import common`. All task scripts in this directory
assume that flag is present - see network_traffic_pipeline_dag.py.

WHAT IS ACTUALLY PARALLEL, AND WHY (unchanged from the original):
  * KMeans / GaussianMixture / PCA (algorithms 1-3, and the GMM step
    inside DAGMM, and the KMeans step inside Deep Embedding)
    -> Native pyspark.ml. Always was distributed.
  * DBSCAN, Isolation Forest (algorithms 4-5)
    -> No distributed-training API exists for either. Approximated by
       fitting one small model per Spark partition in parallel via
       mapInPandas, then combining (core-point union for DBSCAN,
       ensemble-averaging for Isolation Forest).
  * TorchAE / VAE / GAN / USAD / Transformer (algorithms 6-11)
    -> Real PyTorch DistributedDataParallel training via
       pyspark.ml.torch.distributor.TorchDistributor, local_mode=False,
       scheduled across executors. Each rank reads only its own HDFS
       shard.
  * LSTM, OmniAnomaly (algorithms 12-13)
    -> Also DDP-trained, but sharded by contiguous ID ranges so a
       sliding window stays locally sequential. Scoring is a fully
       distributed mapInPandas pass over those same shards.
  * RL Adaptive Threshold (algorithm 14)
    -> One Spark aggregation computes flagged-counts for a grid of
       thresholds in a single pass; the RL loop itself is a handful of
       NumPy table lookups on the driver.

THE ONE DELIBERATELY NON-PARALLEL STEP: assigning contiguous shard IDs
for LSTM/OmniAnomaly requires a single global `Window.orderBy("ID")`.
It happens once (inside task_write_seq_shards.py), not once per epoch.
"""

import math

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType, StringType, StructField, StructType
from pyspark.sql.window import Window

try:
    from pyspark.ml.torch.distributor import TorchDistributor
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "TorchDistributor needs pyspark >= 3.4 (pyspark.ml.torch.distributor)."
    ) from e

try:
    import torch
    import torch.nn as nn
except ImportError as e:  # pragma: no cover
    raise ImportError("This module needs PyTorch: pip install torch") from e

try:
    from sklearn.cluster import DBSCAN
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import NearestNeighbors
except ImportError as e:  # pragma: no cover
    raise ImportError("This module needs scikit-learn: pip install scikit-learn") from e

# ----------------------------------------------------------------------
# Config (unchanged from the monolithic script)
# ----------------------------------------------------------------------
FEATURE_COLS = ["pixel_temp_C", "lar_pressure_bar", "lv_voltage_V", "hv_current_uA", "cooling_flow_lpm"]
Z_COLS = [f"z_{c}" for c in FEATURE_COLS]
N_FEATURES = len(FEATURE_COLS)
RANDOM_SEED = 42

NUM_WORKERS = 2
LOCAL_MODE = False
USE_GPU = False
EPOCHS = 30
BATCH_SIZE = 256
HIDDEN = 16
LATENT = 2
LSTM_WINDOW = 5

EPS_SAMPLE_ROWS = 5_000
ISO_ESTIMATORS_PER_PARTITION = 25

# ----------------------------------------------------------------------
# HDFS layout - the "wire format" between tasks now that each algorithm
# is a separate spark-submit process instead of one in-memory pipeline.
# ----------------------------------------------------------------------
HDFS_ROOT = "hdfs://namenode:9000/user/demo"

SOURCE_HDFS_PATH = f"{HDFS_ROOT}/network_data"          # written by etl_job.py (Oracle -> Delta)
FEATURES_PATH = f"{HDFS_ROOT}/_features"                # written by task_prepare_features.py
ROW_SHARD_PATH = f"{HDFS_ROOT}/_torch_row_shards"        # written by task_write_row_shards.py
SEQ_SHARD_PATH = f"{HDFS_ROOT}/_torch_seq_shards"        # written by task_write_seq_shards.py
MODELS_ROOT = f"{HDFS_ROOT}/_models"
AE_MODEL_PATH = f"{MODELS_ROOT}/autoencoder"             # written by task_train_autoencoder.py
SCORES_ROOT = f"{HDFS_ROOT}/_scores"                     # one sub-path per algorithm, written by each algo task
FINAL_OUTPUT_PATH = f"{HDFS_ROOT}/_final_ensemble"       # written by task_ensemble_report.py

# Canonical list of every algorithm's output directory name + its flag
# column. task_ensemble_report.py joins all of these back together.
ALGO_FLAG_COLS = {
    "kmeans": "kmeans_flag",
    "gmm": "gmm_flag",
    "pca_kmeans": "pca_flag",
    "dbscan": "dbscan_flag",
    "isolation_forest": "iso_flag",
    "deep_embedding": "deepemb_flag",
    "dagmm": "dagmm_flag",
    "vae": "vae_flag",
    "gan": "gan_flag",
    "usad": "usad_flag",
    "transformer": "transformer_flag",
    "lstm": "lstm_flag",
    "omnianomaly": "omni_flag",
    "rl_adaptive_threshold": "adaptive_flag",
}

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ========================================================================
# Torch model zoo (unchanged architectures)
# ========================================================================
class TorchAE(nn.Module):
    def __init__(self, n_features=N_FEATURES, latent=LATENT, hidden=HIDDEN):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, n_features))

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


class TorchVAE(nn.Module):
    def __init__(self, n_features=N_FEATURES, latent=LATENT, hidden=HIDDEN):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden)
        self.fc_mu = nn.Linear(hidden, latent)
        self.fc_logvar = nn.Linear(hidden, latent)
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, n_features))

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class Generator(nn.Module):
    def __init__(self, noise_dim, n_features=N_FEATURES, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(noise_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_features))

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, n_features=N_FEATURES, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


class USADModel(nn.Module):
    def __init__(self, n_features=N_FEATURES, latent=4, hidden=HIDDEN):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, latent))
        self.decoder1 = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, n_features))
        self.decoder2 = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, n_features))

    def forward(self, x):
        z = self.encoder(x)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        return w1, w2, w3


class TinyTransformerAE(nn.Module):
    def __init__(self, n_features=N_FEATURES, d_model=8, nhead=2, hidden=HIDDEN):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.randn(n_features, d_model) * 0.01)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=hidden, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.decode = nn.Linear(d_model, 1)

    def forward(self, x):
        emb = self.embed(x.unsqueeze(-1)) + self.pos
        return self.decode(self.encoder(emb)).squeeze(-1)


class LSTMAE(nn.Module):
    def __init__(self, n_features, window, hidden=HIDDEN):
        super().__init__()
        self.window = window
        self.n_features = n_features
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.decoder = nn.Linear(hidden, window * n_features)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        flat = self.decoder(h[-1])
        return flat.view(-1, self.window, self.n_features)


class OmniAnomalyLite(nn.Module):
    def __init__(self, n_features, hidden=HIDDEN, latent=4):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.fc_mu = nn.Linear(hidden, latent)
        self.fc_logvar = nn.Linear(hidden, latent)
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, n_features))

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        mu = self.fc_mu(last)
        logvar = self.fc_logvar(last)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decoder(z), mu, logvar


# ========================================================================
# Spark session helper
# ========================================================================
def get_spark_session(app_name):
    """All Delta/driver configs are passed via spark-submit --conf (see
    the DAG), so this just fetches the already-configured session."""
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ========================================================================
# HDFS shard writers - the data-plane side of "distributed training"
# ========================================================================
def write_row_shards(df, cols, base_path, num_shards, seed=RANDOM_SEED):
    """Random shard assignment - fine for i.i.d. row-wise models (AE,
    VAE, GAN, USAD, Transformer). One Spark write, Hive-partitioned by
    `_shard` so each rank can read its slice as `base_path/_shard=<rank>`
    with plain pandas/pyarrow, no Spark session needed inside the
    worker process."""
    (df.select("ID", *cols)
       .withColumn("_shard", (F.rand(seed) * num_shards).cast("int"))
       .write.mode("overwrite").partitionBy("_shard").parquet(base_path))
    print(f"[Driver] Wrote {num_shards} random row-wise shards -> {base_path}")


def write_seq_shards(df, cols, base_path, num_shards):
    """Contiguous ID-ordered shard assignment, required so each rank's
    slice is still locally sequential and can build valid sliding
    windows. The Window.orderBy("ID") here is the one deliberately
    non-parallel step in this whole pipeline - it happens once, not
    once per epoch, and only within this single task."""
    w = Window.orderBy("ID")
    numbered = df.select("ID", *cols).withColumn("_rn", F.row_number().over(w) - 1)
    total = numbered.count()
    shard_size = max(1, math.ceil(total / num_shards))
    (numbered.withColumn("_shard", (F.col("_rn") / F.lit(shard_size)).cast("int"))
             .drop("_rn")
             .write.mode("overwrite").partitionBy("_shard").parquet(base_path))
    print(f"[Driver] Wrote {num_shards} contiguous ID-ordered shards ({total} rows) -> {base_path}")
    return total


# ========================================================================
# Generic DDP training harnesses (pyspark.ml.torch.distributor)
# ========================================================================
def distributed_train(model_builder, loss_fn, shard_path, feature_cols, label,
                       epochs=EPOCHS, lr=1e-3, batch_size=BATCH_SIZE,
                       num_processes=NUM_WORKERS, local_mode=LOCAL_MODE):
    """Row-wise DDP trainer, reused by AE / VAE / USAD / Transformer.
    Every rank reads only `shard_path/_shard=<rank>` - the whole
    training set is never materialized in one process."""

    def train_fn():
        import torch as _torch
        import torch.distributed as dist
        import pandas as _pd
        from torch.nn.parallel import DistributedDataParallel as DDP

        backend = "nccl" if _torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank, world = dist.get_rank(), dist.get_world_size()
        device = _torch.device(f"cuda:{rank}" if _torch.cuda.is_available() else "cpu")

        base = model_builder().to(device)
        model = DDP(base) if world > 1 else base

        pdf = _pd.read_parquet(f"{shard_path}/_shard={rank}")
        X = _torch.tensor(pdf[feature_cols].values, dtype=_torch.float32, device=device)
        opt = _torch.optim.Adam(model.parameters(), lr=lr)
        n = X.shape[0]

        for ep in range(epochs):
            perm = _torch.randperm(n)
            total = 0.0
            for i in range(0, n, batch_size):
                xb = X[perm[i:i + batch_size]]
                if xb.shape[0] < 2:
                    continue
                opt.zero_grad()
                loss = loss_fn(model, xb)
                loss.backward()
                opt.step()
                total += loss.item() * xb.shape[0]
            if rank == 0 and ((ep + 1) % 5 == 0 or ep == epochs - 1):
                print(f"    [{label}][rank0/{world}] epoch {ep + 1}/{epochs} loss={total / max(n, 1):.5f}")

        dist.barrier()
        state = {k: v.detach().cpu() for k, v in base.state_dict().items()} if rank == 0 else None
        dist.destroy_process_group()
        return state

    print(f"[TorchDistributor] {label}: launching {num_processes}-way DDP "
          f"({'local processes' if local_mode else 'across cluster executors'})...")
    state_dict = TorchDistributor(num_processes=num_processes, local_mode=local_mode, use_gpu=USE_GPU).run(train_fn)
    model = model_builder()
    model.load_state_dict(state_dict)
    model.eval()
    return model


def distributed_train_gan(gen_builder, disc_builder, shard_path, feature_cols, noise_dim, label="GAN",
                           epochs=EPOCHS, lr=2e-4, batch_size=BATCH_SIZE,
                           num_processes=NUM_WORKERS, local_mode=LOCAL_MODE):
    def train_fn():
        import torch as _torch
        import torch.nn as _nn
        import torch.distributed as dist
        import pandas as _pd
        from torch.nn.parallel import DistributedDataParallel as DDP

        backend = "nccl" if _torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank, world = dist.get_rank(), dist.get_world_size()
        device = _torch.device(f"cuda:{rank}" if _torch.cuda.is_available() else "cpu")

        g_base, d_base = gen_builder().to(device), disc_builder().to(device)
        G = DDP(g_base) if world > 1 else g_base
        D = DDP(d_base) if world > 1 else d_base

        pdf = _pd.read_parquet(f"{shard_path}/_shard={rank}")
        X = _torch.tensor(pdf[feature_cols].values, dtype=_torch.float32, device=device)
        optG, optD = _torch.optim.Adam(G.parameters(), lr=lr), _torch.optim.Adam(D.parameters(), lr=lr)
        bce = _nn.BCELoss()
        n = X.shape[0]

        for ep in range(epochs):
            perm = _torch.randperm(n)
            d_loss = g_loss = _torch.tensor(0.0)
            for i in range(0, n, batch_size):
                real = X[perm[i:i + batch_size]]
                bs = real.shape[0]
                if bs < 2:
                    continue
                noise = _torch.randn(bs, noise_dim, device=device)
                fake = G(noise).detach()
                optD.zero_grad()
                d_loss = bce(D(real), _torch.ones(bs, 1, device=device)) + bce(D(fake), _torch.zeros(bs, 1, device=device))
                d_loss.backward()
                optD.step()

                noise = _torch.randn(bs, noise_dim, device=device)
                fake = G(noise)
                optG.zero_grad()
                g_loss = bce(D(fake), _torch.ones(bs, 1, device=device))
                g_loss.backward()
                optG.step()
            if rank == 0 and ((ep + 1) % 5 == 0 or ep == epochs - 1):
                print(f"    [{label}][rank0/{world}] epoch {ep + 1}/{epochs} "
                      f"D={d_loss.item():.4f} G={g_loss.item():.4f}")

        dist.barrier()
        state = None
        if rank == 0:
            state = {
                "G": {k: v.detach().cpu() for k, v in g_base.state_dict().items()},
                "D": {k: v.detach().cpu() for k, v in d_base.state_dict().items()},
            }
        dist.destroy_process_group()
        return state

    print(f"[TorchDistributor] {label}: launching {num_processes}-way DDP "
          f"({'local processes' if local_mode else 'across cluster executors'})...")
    states = TorchDistributor(num_processes=num_processes, local_mode=local_mode, use_gpu=USE_GPU).run(train_fn)
    G, D = gen_builder(), disc_builder()
    G.load_state_dict(states["G"]); G.eval()
    D.load_state_dict(states["D"]); D.eval()
    return G, D


def distributed_train_sequential(model_builder, loss_fn, shard_path, feature_cols, window, label,
                                  epochs=EPOCHS, lr=1e-3, batch_size=BATCH_SIZE,
                                  num_processes=NUM_WORKERS, local_mode=LOCAL_MODE):
    """Same DDP shape as distributed_train, but reads a *contiguous*
    ID-ordered shard and builds sliding windows locally before
    training - needed for LSTM / OmniAnomaly."""

    def train_fn():
        import torch as _torch
        import torch.distributed as dist
        import pandas as _pd
        import numpy as _np
        from torch.nn.parallel import DistributedDataParallel as DDP

        backend = "nccl" if _torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank, world = dist.get_rank(), dist.get_world_size()
        device = _torch.device(f"cuda:{rank}" if _torch.cuda.is_available() else "cpu")

        base = model_builder().to(device)
        model = DDP(base) if world > 1 else base

        pdf = _pd.read_parquet(f"{shard_path}/_shard={rank}").sort_values("ID").reset_index(drop=True)
        Xnp = pdf[feature_cols].values.astype(_np.float32)
        opt = _torch.optim.Adam(model.parameters(), lr=lr)

        if len(Xnp) > window:
            windows = _np.stack([Xnp[i:i + window] for i in range(len(Xnp) - window + 1)])
            X = _torch.tensor(windows, dtype=_torch.float32, device=device)
            n = X.shape[0]
            for ep in range(epochs):
                perm = _torch.randperm(n)
                total = 0.0
                for i in range(0, n, batch_size):
                    xb = X[perm[i:i + batch_size]]
                    if xb.shape[0] < 2:
                        continue
                    opt.zero_grad()
                    loss = loss_fn(model, xb)
                    loss.backward()
                    opt.step()
                    total += loss.item() * xb.shape[0]
                if rank == 0 and ((ep + 1) % 5 == 0 or ep == epochs - 1):
                    print(f"    [{label}][rank0/{world}] epoch {ep + 1}/{epochs} loss={total / max(n, 1):.5f}")
        elif rank == 0:
            print(f"    [{label}] shard smaller than window={window} - contributed no gradient steps.")

        dist.barrier()
        state = {k: v.detach().cpu() for k, v in base.state_dict().items()} if rank == 0 else None
        dist.destroy_process_group()
        return state

    print(f"[TorchDistributor] {label}: launching {num_processes}-way DDP over contiguous shards "
          f"({'local processes' if local_mode else 'across cluster executors'})...")
    state_dict = TorchDistributor(num_processes=num_processes, local_mode=local_mode, use_gpu=USE_GPU).run(train_fn)
    model = model_builder()
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ========================================================================
# Row-wise distributed scoring (pandas_udf - unchanged pattern)
# ========================================================================
def make_torch_scalar_udf(spark, build_model_fn, model, score_fn, name):
    state_np = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    bc = spark.sparkContext.broadcast(state_np)
    cache = {}

    @F.pandas_udf(DoubleType())
    def _udf(*cols):
        if "m" not in cache:
            m = build_model_fn()
            m.load_state_dict({k: torch.tensor(v) for k, v in bc.value.items()})
            m.eval()
            cache["m"] = m
        X = np.column_stack([c.to_numpy(dtype=np.float32) for c in cols])
        with torch.no_grad():
            s = score_fn(cache["m"], torch.from_numpy(X))
        return pd.Series(np.asarray(s, dtype=np.float64))

    _udf.__name__ = name
    return _udf


def make_torch_array_udf(spark, build_model_fn, model, score_fn, out_dim, name):
    state_np = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    bc = spark.sparkContext.broadcast(state_np)
    cache = {}

    @F.pandas_udf(ArrayType(DoubleType()))
    def _udf(*cols):
        if "m" not in cache:
            m = build_model_fn()
            m.load_state_dict({k: torch.tensor(v) for k, v in bc.value.items()})
            m.eval()
            cache["m"] = m
        X = np.column_stack([c.to_numpy(dtype=np.float32) for c in cols])
        with torch.no_grad():
            arr = score_fn(cache["m"], torch.from_numpy(X))
        arr = np.asarray(arr, dtype=np.float64).reshape(-1, out_dim)
        return pd.Series(list(arr))

    _udf.__name__ = name
    return _udf


def threshold_flag(df, score_col, flag_col, n_std=3.0):
    stats = df.select(F.mean(score_col).alias("m"), F.stddev_pop(score_col).alias("s")).first()
    thr = stats["m"] + n_std * stats["s"]
    print(f"[Driver] {flag_col} threshold on {score_col}: {thr:.4f}")
    return df.withColumn(flag_col, (F.col(score_col) > F.lit(thr)).cast("int"))


def minority_cluster_flag(df, cluster_col, flag_col):
    counts = {r[cluster_col]: r["count"] for r in df.groupBy(cluster_col).count().collect()}
    minority = min(counts, key=counts.get)
    print(f"[Driver] {cluster_col} counts: {counts}, minority={minority}")
    return df.withColumn(flag_col, (F.col(cluster_col) == F.lit(minority)).cast("int"))


# ========================================================================
# Sequential distributed scoring (mapInPandas over the same ID-ordered
# shards used for training - full-dataset coverage, not a driver sample)
# ========================================================================
def score_sequential_distributed(spark, seq_shard_df, model, model_builder, feature_cols, window, id_type, score_fn, out_name):
    # NOTE: only the plain numpy state dict goes through broadcast() - broadcast
    # values are serialized with plain pickle, which stores classes BY REFERENCE
    # (module + qualname). model_builder is a closure instead, captured directly
    # by score_partitions below; that closure travels with mapInPandas's function
    # object, which Spark serializes with cloudpickle. Since model_builder here
    # is itself just a lambda wrapping a class defined in this `common` module
    # (not __main__), cloudpickle will *also* resolve it by reference - which is
    # exactly why every spark-submit call must ship `--py-files /app/common.py`
    # so `import common` succeeds inside the executor's Python worker.
    state_np = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    bc = spark.sparkContext.broadcast(state_np)
    out_schema = StructType([StructField("ID", id_type, True), StructField(out_name, DoubleType(), True)])
    win = window

    def score_partitions(pdf_iter):
        import torch as _torch
        state = bc.value
        m = model_builder()
        m.load_state_dict({k: _torch.tensor(v) for k, v in state.items()})
        m.eval()
        for pdf in pdf_iter:
            for _, group in pdf.groupby("_shard"):
                g = group.sort_values("ID").reset_index(drop=True)
                X = g[feature_cols].values.astype(np.float32)
                ids = g["ID"].values
                if len(X) <= win:
                    continue
                windows = np.stack([X[i:i + win] for i in range(len(X) - win + 1)])
                last_ids = ids[win - 1:]
                Xt = _torch.tensor(windows, dtype=_torch.float32)
                with _torch.no_grad():
                    scores = score_fn(m, Xt)
                yield pd.DataFrame({"ID": last_ids, out_name: np.asarray(scores, dtype=np.float64)})

    return seq_shard_df.mapInPandas(score_partitions, schema=out_schema)


# ========================================================================
# DBSCAN / Isolation Forest - parallel per-partition fit via mapInPandas
# ========================================================================
def estimate_dbscan_eps(X, k):
    nn_model = NearestNeighbors(n_neighbors=k).fit(X)
    dists, _ = nn_model.kneighbors(X)
    return float(np.percentile(np.sort(dists[:, -1]), 97))


def distributed_dbscan_core_points(spark, scaled_df, feature_cols, num_partitions=NUM_WORKERS,
                                    eps_sample_rows=EPS_SAMPLE_ROWS, seed=RANDOM_SEED):
    eps_sample = (scaled_df.select(*feature_cols).orderBy(F.rand(seed)).limit(eps_sample_rows).toPandas())
    k = 3 * len(feature_cols)
    eps = estimate_dbscan_eps(eps_sample[feature_cols].values.astype(np.float32), k)
    print(f"[Driver] DBSCAN eps={eps:.4f} (from a {len(eps_sample)}-row sample), min_samples={k}")

    out_schema = StructType([StructField(f"c_{i}", DoubleType(), True) for i in range(len(feature_cols))])

    def fit_partition(pdf_iter):
        rows = list(pdf_iter)
        if not rows:
            return
        X = pd.concat(rows)[feature_cols].values.astype(np.float32)
        if len(X) < k:
            return
        db = DBSCAN(eps=eps, min_samples=k).fit(X)
        mask = np.zeros(len(X), dtype=bool)
        if len(db.core_sample_indices_):
            mask[db.core_sample_indices_] = True
        core = X[mask]
        if len(core):
            yield pd.DataFrame(core, columns=[f"c_{i}" for i in range(len(feature_cols))])

    part_df = scaled_df.select(*feature_cols).repartition(num_partitions)
    print(f"[Spark] Fitting {num_partitions} local DBSCAN models in parallel across the cluster...")
    core_pdf = part_df.mapInPandas(fit_partition, schema=out_schema).toPandas()
    core_points = core_pdf.values.astype(np.float32)
    print(f"[Driver] Merged {len(core_points)} core points from {num_partitions} parallel partitions.")
    return core_points, eps


def make_dbscan_udf(spark, core_points, eps):
    bc = spark.sparkContext.broadcast(core_points)

    @F.pandas_udf(DoubleType())
    def _udf(*cols):
        from sklearn.metrics import pairwise_distances_argmin_min
        X = np.column_stack([c.to_numpy(dtype=np.float32) for c in cols])
        core = bc.value
        if len(core) == 0:
            return pd.Series(np.full(len(X), eps + 1.0))
        _, dists = pairwise_distances_argmin_min(X, core)
        return pd.Series(dists.astype(np.float64))

    return _udf


def distributed_isolation_forest(spark, scaled_df, feature_cols, num_partitions=NUM_WORKERS,
                                  n_estimators_per_partition=ISO_ESTIMATORS_PER_PARTITION, seed=RANDOM_SEED):
    """One small Isolation Forest per partition, fit concurrently across
    executors; scoring ensemble-averages all of them - a standard way to
    scale a bagging-style model without a native distributed trainer."""
    out_schema = StructType([StructField("forest_b64", StringType(), True)])

    def fit_partition(pdf_iter):
        import pickle, base64
        rows = list(pdf_iter)
        if not rows:
            return
        X = pd.concat(rows)[feature_cols].values.astype(np.float32)
        forest = IsolationForest(n_estimators=n_estimators_per_partition, contamination="auto", random_state=seed)
        forest.fit(X)
        yield pd.DataFrame({"forest_b64": [base64.b64encode(pickle.dumps(forest)).decode("ascii")]})

    part_df = scaled_df.select(*feature_cols).repartition(num_partitions)
    print(f"[Spark] Fitting {num_partitions} local Isolation Forests in parallel across the cluster...")
    blobs = part_df.mapInPandas(fit_partition, schema=out_schema).toPandas()["forest_b64"].tolist()

    import pickle, base64
    forests = [pickle.loads(base64.b64decode(b)) for b in blobs]
    print(f"[Driver] Collected {len(forests)} partition-local forests for ensemble scoring.")
    return forests


def make_iso_ensemble_udf(spark, forests):
    bc = spark.sparkContext.broadcast(forests)

    @F.pandas_udf(DoubleType())
    def _udf(*cols):
        X = np.column_stack([c.to_numpy(dtype=np.float32) for c in cols])
        scores = np.mean([-(f.score_samples(X)) for f in bc.value], axis=0)
        return pd.Series(scores.astype(np.float64))

    return _udf


# ========================================================================
# RL Adaptive Threshold (unchanged - already efficient)
# ========================================================================
def rl_tune_adaptive_threshold(df, feature_cols, target_rate=0.05, episodes=150,
                                epsilon_start=1.0, epsilon_min=0.05, decay=0.97, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    stat_exprs = []
    for c in feature_cols:
        stat_exprs += [F.mean(c).alias(f"{c}_mean"), F.stddev_pop(c).alias(f"{c}_std")]
    stats_row = df.select(*stat_exprs).first()

    zdf = df
    absz_cols = []
    for c in feature_cols:
        mean_c, std_c = stats_row[f"{c}_mean"], max(stats_row[f"{c}_std"], 1e-10)
        zdf = zdf.withColumn(f"_absz_{c}", F.abs((F.col(c) - F.lit(mean_c)) / F.lit(std_c)))
        absz_cols.append(f"_absz_{c}")
    zdf = zdf.withColumn("_max_abs_z", F.greatest(*[F.col(c) for c in absz_cols])).cache()

    total = zdf.count()
    thresholds = np.round(np.arange(2.0, 4.01, 0.1), 2)
    count_exprs = [F.sum((F.col("_max_abs_z") > F.lit(float(t))).cast("int")).alias(f"c_{i}")
                   for i, t in enumerate(thresholds)]
    counts_row = zdf.select(*count_exprs).first()
    rates = np.array([counts_row[f"c_{i}"] / total for i in range(len(thresholds))])
    zdf.unpersist()

    q, visits = np.zeros(len(thresholds)), np.zeros(len(thresholds))
    epsilon = epsilon_start
    for _ in range(episodes):
        a = rng.integers(0, len(thresholds)) if rng.random() < epsilon else int(np.argmax(q))
        reward = -abs(rates[a] - target_rate)
        visits[a] += 1
        q[a] += (1.0 / visits[a]) * (reward - q[a])
        epsilon = max(epsilon_min, epsilon * decay)

    best = int(np.argmax(q))
    chosen = float(thresholds[best])
    print(f"[RL] Adaptive-threshold agent converged: threshold={chosen} "
          f"(observed flag-rate={rates[best]:.4f}, target={target_rate})")
    return chosen, stats_row


def active_exploration_queue(final_df, n_algorithms, top_k=25, epsilon=0.2, seed=RANDOM_SEED):
    half = n_algorithms / 2.0
    scored = final_df.withColumn("boundary_distance", F.abs(F.col("vote") - F.lit(half)))
    n_random = max(1, int(top_k * epsilon))
    n_exploit = max(0, top_k - n_random)
    exploit_df = scored.orderBy(F.col("boundary_distance").asc(), F.col("vote").desc()).limit(n_exploit)
    random_df = scored.orderBy(F.rand(seed)).limit(n_random)
    return exploit_df.unionByName(random_df).dropDuplicates(["ID"])


# ========================================================================
# NEW: task-boundary I/O helpers
# Separate spark-submit processes don't share memory, so anything one
# task computes that a later task needs has to round-trip through HDFS.
# These wrap that in one place so each task_*.py file stays short.
# ========================================================================
def save_torch_state(spark, state_dict, path):
    """Persist a trained model's state_dict so a later task can reload
    it without retraining. Uses the same base64-pickle-in-a-dataframe
    idiom the original script already used to move IsolationForest
    objects from executors back to the driver - no new dependencies."""
    import pickle, base64
    cpu_state = {k: v.detach().cpu() for k, v in state_dict.items()}
    blob = base64.b64encode(pickle.dumps(cpu_state)).decode("ascii")
    spark.createDataFrame([(blob,)], ["state_b64"]).write.mode("overwrite").parquet(path)
    print(f"[Driver] Saved model state -> {path}")


def load_torch_state(spark, path):
    import pickle, base64
    row = spark.read.parquet(path).first()
    return pickle.loads(base64.b64decode(row["state_b64"]))


def save_algo_output(df, name):
    """Every algorithm task writes ID + its flag (+ any diagnostic score
    columns) here. mode('overwrite') makes retries idempotent."""
    out_path = f"{SCORES_ROOT}/{name}"
    df.write.format("delta").mode("overwrite").save(out_path)
    print(f"[Driver] Wrote '{name}' output -> {out_path}")


def load_algo_output(spark, name):
    return spark.read.format("delta").load(f"{SCORES_ROOT}/{name}")