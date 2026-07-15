"""
Distributed Anomaly Ensemble - Fully Distributed Edition
===========================================================
 
Same 14-algorithm ensemble as the multi-paradigm version, but every
training step - not just scoring - now runs spread across the Spark
cluster's worker nodes instead of on the driver.
 
WHAT IS ACTUALLY PARALLEL NOW, AND WHY (please read before assuming
everything is embarrassingly parallel - it isn't, and the honest
exceptions are called out explicitly):
 
  * KMeans / GaussianMixture / PCA (Algorithms 1-3, and the GMM step
    inside DAGMM, and the KMeans step inside Deep Embedding)
    -> Native pyspark.ml. Always was distributed; unchanged.
 
  * DBSCAN, Isolation Forest (Algorithms 4-5)
    -> No distributed-training API exists for either in sklearn or
       MLlib. Approximated by fitting one small model per Spark
       partition IN PARALLEL via mapInPandas (each partition's fit
       runs on a different executor, concurrently), then combining:
       Isolation Forest -> ensemble-average the per-partition forests'
       scores at inference time (a standard bagging-style scale-out).
       DBSCAN -> union the "core point" sets found by each partition's
       local fit (a known approximation: a point near what would be a
       global cluster boundary can be mis-judged if its partition
       happens to be sparse there - acceptable as long as each
       partition still has rows/min_samples >> 1, exactly like the
       approximation the original script's docstring flagged for its
       old per-partition K-Means merge).
 
  * TorchAE / VAE / GAN / USAD / Transformer (Algorithms 6-11 incl.
    the AE feeding DAGMM)
    -> Real PyTorch DistributedDataParallel training via
       pyspark.ml.torch.distributor.TorchDistributor, launched with
       local_mode=False so Spark schedules the training processes
       across actual executors/nodes (not just local CPU cores on the
       driver). Each rank reads only its own shard of the training
       data from HDFS - nobody, including the driver, ever holds the
       full training set in memory. Only the trained weights (a few KB
       state_dict) come back to the driver, then get broadcast out
       again for the (already-distributed) pandas_udf scoring pass.
 
  * LSTM, OmniAnomaly (Algorithms 12-13)
    -> Also DDP-trained via TorchDistributor, but sharded by
       CONTIGUOUS ID ranges rather than randomly, since a sliding
       window needs local sequential order. Scoring is now ALSO a
       fully distributed mapInPandas pass (previously this ran once,
       driver-side, on a capped sample - now it covers the whole
       dataset except for (window-1) boundary rows per shard).
 
  * RL Adaptive Threshold (Algorithm 14), Active Exploration
    -> Unchanged: one Spark aggregation computes flagged-counts for a
       whole grid of thresholds in a single pass; the RL loop itself
       is a handful of NumPy table lookups on the driver, which is
       correct because it's cheap, not because it's been parallelized.
 
THE ONE DELIBERATELY NON-PARALLEL STEP: assigning contiguous shard IDs
for LSTM/OmniAnomaly requires a single global `Window.orderBy("ID")`,
which Spark executes by funneling everything through one partition to
compute row numbers. There is no way around this without giving up a
true global order, and a true global order is what a sequence model
needs to be correct. It happens once, not once per epoch.
 
OPERATIONAL PREREQUISITES:
  * pyspark >= 3.4 (for pyspark.ml.torch.distributor.TorchDistributor)
  * torch, scikit-learn, pandas, numpy, pyarrow on every executor node
  * The cluster must actually have >= NUM_WORKERS free executor slots,
    or TorchDistributor's barrier-mode launch will hang waiting for
    processes it can't schedule. Set LOCAL_MODE=True to sanity-check
    on a single machine first (spawns NUM_WORKERS local processes
    instead of scheduling across the cluster).
"""
 
import math
 
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType, StringType, StructField, StructType
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.clustering import KMeans, GaussianMixture
from pyspark.ml.functions import array_to_vector, vector_to_array
 
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
    raise ImportError("This script needs PyTorch: pip install torch") from e
 
try:
    from sklearn.cluster import DBSCAN
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import NearestNeighbors
except ImportError as e:  # pragma: no cover
    raise ImportError("This script needs scikit-learn: pip install scikit-learn") from e
 
# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
FEATURE_COLS = ["PACKET_SIZE", "CONNECTION_TIME", "FAILED_LOGINS"]
Z_COLS = [f"z_{c}" for c in FEATURE_COLS]
N_FEATURES = len(FEATURE_COLS)
RANDOM_SEED = 42
 
NUM_WORKERS = 2         # TorchDistributor DDP processes == number of HDFS shards written
LOCAL_MODE = False        # False: schedule across real cluster executors. True: local sanity check.
USE_GPU = False
EPOCHS = 15
BATCH_SIZE = 256
HIDDEN = 16
LATENT = 2
LSTM_WINDOW = 10
 
HDFS_ROOT = "hdfs://namenode:9000/user/demo"
ROW_SHARD_PATH = f"{HDFS_ROOT}/_torch_row_shards"
SEQ_SHARD_PATH = f"{HDFS_ROOT}/_torch_seq_shards"
 
EPS_SAMPLE_ROWS = 5_000        # small driver-side sample, just for the DBSCAN eps heuristic
ISO_ESTIMATORS_PER_PARTITION = 25
 
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
 
 
# ========================================================================
# Torch model zoo (unchanged architectures from the multi-paradigm build)
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
    non-parallel step in this whole pipeline - seq numbering forces a
    single-partition pass. It happens once, not per epoch."""
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
    # object, which Spark serializes with cloudpickle - and cloudpickle serializes
    # classes/closures defined in __main__ BY VALUE. Broadcasting model.__class__
    # directly (the old code) looked up "LSTMAE"/"OmniAnomalyLite" inside
    # pyspark.daemon on the worker and failed, since that class only exists in
    # the driver script's __main__, not in the daemon's namespace.
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
    return float(np.percentile(np.sort(dists[:, -1]), 95))
 
 
def distributed_dbscan_core_points(spark, scaled_df, feature_cols, num_partitions=NUM_WORKERS,
                                    eps_sample_rows=EPS_SAMPLE_ROWS, seed=RANDOM_SEED):
    eps_sample = (scaled_df.select(*feature_cols).orderBy(F.rand(seed)).limit(eps_sample_rows).toPandas())
    k = 2 * len(feature_cols)
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
        scores = np.mean([-(f.score_samples(X)) for f in bc.value], axis=0)  # higher = more anomalous
        return pd.Series(scores.astype(np.float64))
 
    return _udf
 
 
# ========================================================================
# RL Adaptive Threshold (unchanged - already efficient, see notes above)
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
# Main
# ========================================================================
def main():
    spark = SparkSession.builder.appName("Distributed_Anomaly_Ensemble_FullyDistributed").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
 
    # ------------------------------------------------------------------
    # 0. Load + assemble + scale
    # ------------------------------------------------------------------
    hdfs_path = "hdfs://namenode:9000/user/demo/network_data.parquet"
    df = spark.read.parquet(hdfs_path).select("ID", *FEATURE_COLS)
    id_type = df.schema["ID"].dataType
    total_rows = df.count()
    print(f"[Driver] Loaded {total_rows} rows from HDFS.")
 
    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="raw_features")
    assembled = assembler.transform(df)
    scaler = StandardScaler(inputCol="raw_features", outputCol="scaled_features", withMean=True, withStd=True)
    scaled_df = scaler.fit(assembled).transform(assembled)
    scaled_df = scaled_df.withColumn("_scaled_arr", vector_to_array("scaled_features"))
    for i, c in enumerate(FEATURE_COLS):
        scaled_df = scaled_df.withColumn(f"z_{c}", F.col("_scaled_arr")[i])
    scaled_df = scaled_df.drop("_scaled_arr").cache()
 
    # ==================================================================
    # SECTION 1 - Spark-native clustering (already fully distributed)
    # ==================================================================
    print("\n[MLlib] Algorithm 1: KMeans + Micro-Cluster...")
    kmeans = KMeans(featuresCol="scaled_features", predictionCol="kmeans_cluster", k=2, seed=RANDOM_SEED)
    kmeans_model = kmeans.fit(scaled_df)
    centers_bc = spark.sparkContext.broadcast(np.array([c.tolist() for c in kmeans_model.clusterCenters()]))
 
    @F.pandas_udf(DoubleType())
    def dist_to_nearest_center(*cols):
        X = np.column_stack([c.to_numpy(dtype=np.float64) for c in cols])
        C = centers_bc.value
        d = np.sqrt(((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2))
        return pd.Series(d.min(axis=1))
 
    working_df = kmeans_model.transform(scaled_df).withColumn("kmeans_dist", dist_to_nearest_center(*Z_COLS))
    working_df = threshold_flag(working_df, "kmeans_dist", "kmeans_flag", n_std=0.7)
 
    print("\n[MLlib] Algorithm 2: Gaussian Mixture Model...")
    gmm = GaussianMixture(featuresCol="scaled_features", predictionCol="gmm_cluster",
                           probabilityCol="gmm_probability", k=3, seed=RANDOM_SEED)
    working_df = gmm.fit(scaled_df).transform(working_df)
    working_df = minority_cluster_flag(working_df, "gmm_cluster", "gmm_flag")
 
    print("\n[MLlib] Algorithm 3: PCA + KMeans Micro-Cluster...")
    pca = PCA(k=2, inputCol="scaled_features", outputCol="pca_features")
    working_df = pca.fit(scaled_df).transform(working_df)
    pca_kmeans = KMeans(featuresCol="pca_features", predictionCol="pca_cluster", k=2, seed=RANDOM_SEED)
    working_df = pca_kmeans.fit(working_df).transform(working_df)
    working_df = minority_cluster_flag(working_df, "pca_cluster", "pca_flag")
 
    # ==================================================================
    # SECTION 2 - DBSCAN / Isolation Forest (parallel per-partition fit)
    # ==================================================================
    print("\n[sklearn + Spark] Algorithm 4: DBSCAN (per-partition, parallel)...")
    core_points, eps = distributed_dbscan_core_points(spark, scaled_df, Z_COLS)
    dbscan_udf = make_dbscan_udf(spark, core_points, eps)
    working_df = working_df.withColumn("dbscan_dist", dbscan_udf(*Z_COLS))
    working_df = working_df.withColumn("dbscan_flag", (F.col("dbscan_dist") > F.lit(eps)).cast("int"))
 
    print("\n[sklearn + Spark] Algorithm 5: Isolation Forest (per-partition ensemble, parallel)...")
    forests = distributed_isolation_forest(spark, scaled_df, Z_COLS)
    iso_udf = make_iso_ensemble_udf(spark, forests)
    working_df = working_df.withColumn("iso_score", iso_udf(*Z_COLS))
    working_df = threshold_flag(working_df, "iso_score", "iso_flag", n_std=2.3)
 
    # ------------------------------------------------------------------
    # Write HDFS training shards once, reused by every DDP job below
    # ------------------------------------------------------------------
    write_row_shards(scaled_df, Z_COLS, ROW_SHARD_PATH, NUM_WORKERS)
    write_seq_shards(scaled_df, Z_COLS, SEQ_SHARD_PATH, NUM_WORKERS)
 
    # ==================================================================
    # SECTION 3 - Deep Embedding (DDP-trained AE + pyspark.ml KMeans)
    # ==================================================================
    print("\n[TorchDistributor + MLlib] Algorithm 6: Deep Embedding (Autoencoder + KMeans)...")
 
    def ae_loss(m, xb):
        _, xhat = m(xb)
        return nn.functional.mse_loss(xhat, xb)
 
    ae = distributed_train(TorchAE, ae_loss, ROW_SHARD_PATH, Z_COLS, label="AE")
 
    def embed_latent(m, xb):
        z, _ = m(xb)
        return z.numpy()
 
    embed_udf = make_torch_array_udf(spark, TorchAE, ae, embed_latent, LATENT, "deep_embed")
    working_df = working_df.withColumn("deepemb_arr", embed_udf(*Z_COLS))
    working_df = working_df.withColumn("deepemb_vec", array_to_vector("deepemb_arr"))
    deepemb_kmeans = KMeans(featuresCol="deepemb_vec", predictionCol="deepemb_cluster", k=2, seed=RANDOM_SEED)
    working_df = deepemb_kmeans.fit(working_df).transform(working_df)
    working_df = minority_cluster_flag(working_df, "deepemb_cluster", "deepemb_flag")
 
    # ==================================================================
    # SECTION 4 - DAGMM (same DDP-trained AE + pyspark.ml GaussianMixture)
    # ==================================================================
    print("\n[TorchDistributor + MLlib] Algorithm 7: DAGMM...")
 
    def dagmm_repr(m, xb):
        z, xhat = m(xb)
        euclid = torch.norm(xb - xhat, dim=1, keepdim=True)
        cos = nn.functional.cosine_similarity(xb, xhat, dim=1).unsqueeze(1)
        return torch.cat([z, euclid, cos], dim=1).numpy()
 
    dagmm_udf = make_torch_array_udf(spark, TorchAE, ae, dagmm_repr, LATENT + 2, "dagmm_repr")
    working_df = working_df.withColumn("dagmm_arr", dagmm_udf(*Z_COLS))
    working_df = working_df.withColumn("dagmm_vec", array_to_vector("dagmm_arr"))
    dagmm_gmm = GaussianMixture(featuresCol="dagmm_vec", predictionCol="dagmm_cluster",
                                 probabilityCol="dagmm_probability", k=3, seed=RANDOM_SEED)
    working_df = dagmm_gmm.fit(working_df).transform(working_df)
    working_df = working_df.withColumn(
        "dagmm_energy", -F.log(F.array_max(vector_to_array(F.col("dagmm_probability"))) + F.lit(1e-12))
    )
    working_df = threshold_flag(working_df, "dagmm_energy", "dagmm_flag", n_std=0.3)
 
    # ==================================================================
    # SECTION 5 - VAE (DDP-trained)
    # ==================================================================
    print("\n[TorchDistributor] Algorithm 8: Variational Autoencoder...")
 
    def vae_loss(m, xb):
        xhat, mu, logvar = m(xb)
        recon = nn.functional.mse_loss(xhat, xb)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + 0.1 * kl
 
    vae = distributed_train(TorchVAE, vae_loss, ROW_SHARD_PATH, Z_COLS, label="VAE")
 
    def vae_score(m, xb):
        mu, logvar = m.encode(xb)
        xhat = m.decoder(mu)
        recon = ((xhat - xb) ** 2).mean(dim=1)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return (recon + 0.1 * kl).numpy()
 
    vae_udf = make_torch_scalar_udf(spark, TorchVAE, vae, vae_score, "vae_score")
    working_df = working_df.withColumn("vae_score", vae_udf(*Z_COLS))
    working_df = threshold_flag(working_df, "vae_score", "vae_flag", n_std=1.8)
 
    # ==================================================================
    # SECTION 6 - GAN (DDP-trained G + D)
    # ==================================================================
    print("\n[TorchDistributor] Algorithm 9: GAN (Generator / Discriminator)...")
    noise_dim = 8
    _, D = distributed_train_gan(lambda: Generator(noise_dim), Discriminator, ROW_SHARD_PATH, Z_COLS, noise_dim,
                                  epochs=30, lr=5e-5)
 
    def gan_score(m, xb):
        return (1.0 - m(xb)).squeeze(-1).numpy()
 
    gan_udf = make_torch_scalar_udf(spark, Discriminator, D, gan_score, "gan_score")
    working_df = working_df.withColumn("gan_score", gan_udf(*Z_COLS))
    working_df = threshold_flag(working_df, "gan_score", "gan_flag", n_std=1.5)
 
    # ==================================================================
    # SECTION 7 - USAD (DDP-trained)
    # ==================================================================
    print("\n[TorchDistributor] Algorithm 10: USAD...")
 
    def usad_loss(m, xb):
        w1, w2, w3 = m(xb)
        l_ae1 = nn.functional.mse_loss(w1, xb)
        l_ae2 = nn.functional.mse_loss(w2, xb)
        l_adv = nn.functional.mse_loss(w3, xb)
        return l_ae1 + l_ae2 - 0.5 * l_adv.detach() + 0.5 * l_adv
 
    usad = distributed_train(USADModel, usad_loss, ROW_SHARD_PATH, Z_COLS, label="USAD")
 
    def usad_score(m, xb):
        w1, _, w3 = m(xb)
        s1 = ((xb - w1) ** 2).mean(dim=1)
        s2 = ((xb - w3) ** 2).mean(dim=1)
        return (0.5 * s1 + 0.5 * s2).numpy()
 
    usad_udf = make_torch_scalar_udf(spark, USADModel, usad, usad_score, "usad_score")
    working_df = working_df.withColumn("usad_score", usad_udf(*Z_COLS))
    working_df = threshold_flag(working_df, "usad_score", "usad_flag", n_std=0.3)
 
    # ==================================================================
    # SECTION 8 - Transformer (DDP-trained)
    # ==================================================================
    print("\n[TorchDistributor] Algorithm 11: Transformer-based reconstruction...")
 
    def trf_loss(m, xb):
        return nn.functional.mse_loss(m(xb), xb)
 
    trf = distributed_train(TinyTransformerAE, trf_loss, ROW_SHARD_PATH, Z_COLS, label="Transformer")
 
    def trf_score(m, xb):
        return ((m(xb) - xb) ** 2).mean(dim=1).numpy()
 
    trf_udf = make_torch_scalar_udf(spark, TinyTransformerAE, trf, trf_score, "transformer_score")
    working_df = working_df.withColumn("transformer_score", trf_udf(*Z_COLS))
    working_df = threshold_flag(working_df, "transformer_score", "transformer_flag", n_std=0.75)
 
    # ==================================================================
    # SECTION 9 & 10 - LSTM / OmniAnomaly: DDP training + distributed
    # mapInPandas scoring, both over contiguous ID-ordered shards
    # ==================================================================
    print("\n[TorchDistributor] Algorithm 12: LSTM sequence-autoencoder...")
 
    def lstm_loss(m, xb):
        xhat = m(xb)
        return nn.functional.mse_loss(xhat, xb)
 
    lstm_model = distributed_train_sequential(
        lambda: LSTMAE(N_FEATURES, LSTM_WINDOW), lstm_loss, SEQ_SHARD_PATH, Z_COLS, LSTM_WINDOW, label="LSTM"
    )
 
    def lstm_score_fn(m, xb):
        xhat = m(xb)
        return ((xhat[:, -1, :] - xb[:, -1, :]) ** 2).mean(dim=1).numpy()
 
    seq_shard_df = spark.read.parquet(SEQ_SHARD_PATH)
    lstm_scores_df = score_sequential_distributed(
        spark, seq_shard_df, lstm_model, lambda: LSTMAE(N_FEATURES, LSTM_WINDOW),
        Z_COLS, LSTM_WINDOW, id_type, lstm_score_fn, "lstm_score"
    )
    working_df = working_df.join(lstm_scores_df, on="ID", how="left")
    lstm_stats = lstm_scores_df.select(F.mean("lstm_score").alias("m"), F.stddev_pop("lstm_score").alias("s")).first()
    lstm_thr = lstm_stats["m"] + 0.6 * lstm_stats["s"]
    working_df = working_df.withColumn("lstm_flag", F.when(F.col("lstm_score") > F.lit(lstm_thr), 1).otherwise(0))
 
    print("\n[TorchDistributor] Algorithm 13: OmniAnomaly (GRU + stochastic VAE)...")
 
    def omni_loss(m, xb):
        xhat_last, mu, logvar = m(xb)
        recon = nn.functional.mse_loss(xhat_last, xb[:, -1, :])
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + 0.1 * kl
 
    omni_model = distributed_train_sequential(
        lambda: OmniAnomalyLite(N_FEATURES), omni_loss, SEQ_SHARD_PATH, Z_COLS, LSTM_WINDOW, label="OmniAnomaly"
    )
 
    def omni_score_fn(m, xb):
        xhat_last, mu, logvar = m(xb)
        recon = ((xhat_last - xb[:, -1, :]) ** 2).mean(dim=1)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return (recon + 0.1 * kl).numpy()
 
    omni_scores_df = score_sequential_distributed(
        spark, seq_shard_df, omni_model, lambda: OmniAnomalyLite(N_FEATURES),
        Z_COLS, LSTM_WINDOW, id_type, omni_score_fn, "omni_score"
    )
    working_df = working_df.join(omni_scores_df, on="ID", how="left")
    omni_stats = omni_scores_df.select(F.mean("omni_score").alias("m"), F.stddev_pop("omni_score").alias("s")).first()
    omni_thr = omni_stats["m"] + 1.5 * omni_stats["s"]
    working_df = working_df.withColumn("omni_flag", F.when(F.col("omni_score") > F.lit(omni_thr), 1).otherwise(0))
 
    # rows in a boundary gap (last window-1 of each shard) get no score - default to 0, not anomalous
    working_df = working_df.fillna({"lstm_flag": 0, "omni_flag": 0})
 
    # ==================================================================
    # SECTION 11 - RL Adaptive Threshold
    # ==================================================================
    print("\n[RL] Algorithm 14: RL-tuned Adaptive Threshold...")
    chosen_threshold, stats_row = rl_tune_adaptive_threshold(df, FEATURE_COLS)
    z_conditions = []
    for c in FEATURE_COLS:
        mean_c, std_c = stats_row[f"{c}_mean"], max(stats_row[f"{c}_std"], 1e-10)
        z_conditions.append(F.abs((F.col(c) - F.lit(mean_c)) / F.lit(std_c)) > F.lit(chosen_threshold))
    adaptive_condition = z_conditions[0]
    for cond in z_conditions[1:]:
        adaptive_condition = adaptive_condition | cond
    working_df = working_df.withColumn("adaptive_flag", adaptive_condition.cast("int"))
 
    # ==================================================================
    # ENSEMBLE VOTE
    # ==================================================================
    ALGO_FLAG_COLS = [
        "kmeans_flag", "gmm_flag", "pca_flag",
        "dbscan_flag", "iso_flag",
        "deepemb_flag", "dagmm_flag",
        "vae_flag", "gan_flag", "usad_flag",
        "transformer_flag", "lstm_flag", "omni_flag",
        "adaptive_flag",
    ]
    n_algorithms = len(ALGO_FLAG_COLS)
    min_votes = math.ceil(n_algorithms / 2)
 
    vote_expr = F.col(ALGO_FLAG_COLS[0])
    for c in ALGO_FLAG_COLS[1:]:
        vote_expr = vote_expr + F.col(c)
 
    final_df = working_df.withColumn("vote", vote_expr).withColumn(
        "ensemble", (F.col("vote") >= F.lit(min_votes)).cast("int")
    ).cache()
 
    explore_df = active_exploration_queue(final_df, n_algorithms)
 
    # ==================================================================
    # REPORTING
    # ==================================================================
    print(f"\n[Driver] {n_algorithms} algorithms in the ensemble; flagging anomaly at vote >= {min_votes}.")
 
    print("\n--- Top 10 Most Anomalous Rows (Highest Votes) ---")
    final_df.select("ID", *FEATURE_COLS, "vote", "ensemble").orderBy(F.desc("vote")).limit(10).show(truncate=False)
 
    print("\n--- Algorithm Performance Summary ---")
    summary_exprs = [F.sum(c).alias(c) for c in ALGO_FLAG_COLS] + [F.sum("ensemble").alias("Ensemble_Final")]
    print(final_df.select(*summary_exprs).first().asDict())
 
    print("\n--- Cluster Sizes (Ensemble) ---")
    final_df.groupBy("ensemble").count().show()
 
    print(f"\n--- Active Exploration Queue (top {explore_df.count()} rows for analyst review) ---")
    explore_df.select("ID", *FEATURE_COLS, "vote", "boundary_distance").orderBy("boundary_distance").show(truncate=False)
 
    spark.stop()
 
 
if __name__ == "__main__":
    main()