#!/bin/bash
# Wrapper around Spark's own entrypoint.sh.
#
# PyArrow's HDFS filesystem (used by pandas.read_parquet("hdfs://...") inside
# the TorchDistributor worker processes) talks to HDFS via libhdfs, which
# embeds its own JVM. That embedded JVM finds the Hadoop client jars purely
# through the CLASSPATH environment variable - it does NOT use Spark's own
# classpath, so Spark working fine tells us nothing about whether pyarrow
# will work. Without this, workers fail with "CLASSPATH not set" /
# "getJNIEnv failed" / "HDFS connection failed" as soon as they try to read.
set -e

export HADOOP_HOME="${HADOOP_HOME:-/opt/hadoop}"
export HADOOP_CONF_DIR="${HADOOP_CONF_DIR:-$HADOOP_HOME/etc/hadoop}"
export PATH="$HADOOP_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$HADOOP_HOME/lib/native:$LD_LIBRARY_PATH"
export ARROW_LIBHDFS_DIR="$HADOOP_HOME/lib/native"

# --glob gives a wildcard-form classpath (dir/*) which the JVM expands itself;
# this is the form the Hadoop/Arrow docs recommend for CLASSPATH.
if HADOOP_CP="$("$HADOOP_HOME/bin/hadoop" classpath --glob 2>/tmp/hadoop-classpath.err)"; then
    export CLASSPATH="$HADOOP_CP:$CLASSPATH"
else
    echo "WARNING: 'hadoop classpath --glob' failed, CLASSPATH not set - PyArrow HDFS reads may fail. See /tmp/hadoop-classpath.err" >&2
fi

exec /opt/entrypoint.sh "$@"