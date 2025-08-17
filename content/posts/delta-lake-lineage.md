---
author: "Sercan Ahi"
title: "Delta Lake Lineage Essentials in Databricks Unity Catalog"
date: "2025-08-16"
---

As teams build data products on Databricks Unity Catalog (UC), one question always comes up:

> _"How do we track which tables feed into which outputs?"_

This post summarizes the essentials of **Delta Lake lineage** in a Unity Catalog–enabled Databricks workspace.

## Delta Lake = Parquet + `_delta_log`

- Delta Lake is an open-source storage layer that enables Databricks Workspaces and other platforms to reliably manage **Delta Tables** with ACID transactions and governance features.
- A Delta Table is the fundamental storage unit of Delta Lake, a Parquet-based table enhanced with a transaction log (`_delta_log`) that brings database-like reliability and auditability to data lake.
- Think of Delta Lake like an _operating system_, and Delta Tables like the _files_ it manages.

A Delta Table is:

- Parquet data files (the rows and columns), plus
- a hidden `_delta_log` folder that stores all changes ever happened to that table.

That `_delta_log` is what enables:

1. ACID transactions
2. Time Travel
3. Schema enforcement
4. Auditability

## Unity Catalog Captures Lineage Automatically

When you run a **write action**, such as `saveAsTable`, `MERGE`, or `INSERT`, Spark builds a query plan. UC inspects that plan and records:

- **Target table**: the one being written.
- **Source tables**: all UC-managed tables referenced in the plan.

Lineage is captured per action, not for the whole notebook or script. Each write generates its own **lineage edge**.

## Reads Do not Show Up in `_delta_log`

- `_delta_log` only logs writes.
- Reads from a table do not leave a trace.
- That is why UC’s query-plan lineage is valuable. It can see both inputs and outputs.

## Tagging Commits with `userMetadata`

Unity Catalog is great at technical lineage (table A → table B), but sometimes you want extra context, such as pipeline name, source list, or environment.

You can add this yourself using `spark.databricks.delta.commitInfo.userMetadata`.

Example: Tagging a Write

```python
import json

payload = {
    "sources": ["main.raw.orders", "main.ref.customers"],
    "pipeline": "daily_sales_build",
    "env": "prod-apac",
}
spark.conf.set(
    "spark.databricks.delta.commitInfo.userMetadata",
    json.dumps(payload),
)

(
    df
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("main.analytics.customer_sales")
)

# Clean up
spark.conf.unset("spark.databricks.delta.commitInfo.userMetadata")
```

Now, if you inspect the target table's `_delta_log`, you will see your metadata alongside the commit info.

## How can we locate target table's `_delta_log`?

If you want to work with a specific table (for example, `table_name = "main.analytics.my_table"`) in a UC–enabled workspace, you can choose from at least two approaches.

1. Unity Catalog (recommended)

    ```python
    loc = (
        spark
        .sql(f"DESCRIBE DETAIL {table_name}")
        .select("location")
        .first()[0]
    )
    log_path = f"{loc.rstrip('/')}/_delta_log/*.json"
    ```

2. DeltaTable API

    ```python
    from delta.tables import DeltaTable

    dt = DeltaTable.forName(spark, table_name)
    loc = dt.detail().select("location").first()[0]
    log_path = f"{loc.rstrip('/')}/_delta_log/*.json"
    ```

Once you have `log_path`, you can parse the commit logs as follows.

```python
from pyspark.sql.functions import (
    col,
    input_file_name,
    regexp_extract,
)

commits_df = (
    spark
    .read
    .json(log_path)
    # Capture file names
    .withColumn("file", input_file_name())
    # Extract numeric version from the filename
    # (e.g., 00000000000000000123.json -> 123)
    .withColumn(
        "version",
        (
            regexp_extract(col("file"), r"/_delta_log/0+([0-9]+)\.json$", 1)
            .cast("long")
        )
    )
    .dropDuplicates()
)

commits_df.display()
```

## Best Practices

- Use UC names everywhere (`catalog.schema.table`) so lineage can be captured.
- Register external sources as UC External Locations — do not rely on raw `s3://...` (AWS), `abfss://...` (Azure), or `gc://...` (GCP) paths.
- Tag critical pipelines with `userMetadata` to carry business context.
- Automate inspection by for example scheduling jobs that parse `_delta_log` and publish **lineage/audit reports**.

## Summary

- Delta Lake adds reliability to Parquet with `_delta_log`.
- Unity Catalog captures lineage automatically from query plans. Each `WRITE` produces an input-output record.
- Custom tags (`userMetadata`) let you enrich lineage with pipeline context.

Together, these tools give tech leads a solid foundation for **auditable**, **governed** pipelines in Databricks.
