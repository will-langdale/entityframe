# Starlings

**High-performance entity resolution evaluation and transport for Python**

Ever wondered if your entity resolution is actually working? Or spent hours recomputing clusters just to test different thresholds? Starlings lets you explore the entire resolution space instantly—because sometimes you need to know whether 0.85 or 0.87 is the magic number that makes everything click.

## Why Starlings?

* **Test any threshold, instantly** – Build once, query forever. No more recomputing clusters every time someone asks "but what if we used 0.9?"
* **Compare resolution methods fairly** – Put Splink, Dedupe, and your custom algorithm head-to-head at their optimal thresholds, not some arbitrary guess
* **Transport complete results** – Ship the full resolution hierarchy to colleagues, not just frozen clusters. Let downstream users pick their own thresholds
* **Find stability sweet spots** – Discover threshold ranges where small changes won't blow up your entire analysis
* **Handle real-world messiness** – Multiple data sources? Check. Millions of records? Check. That one CSV with weird encodings? ...probably check

## The lightning tour

```python
import starlings as sl

# Load your data
ef = sl.from_records("customers", your_dataframe)

# Add different resolution attempts
ef.add_collection_from_edges("splink_try1", splink_edges)
ef.add_collection_from_edges("splink_try2_tweaked", other_edges)
ef.add_collection_from_entities("ground_truth", known_entities)

# Find what actually works
results = ef.analyse(
    sl.col("splink_try1").sweep(0.5, 0.95),  # Test 45 thresholds
    sl.col("ground_truth").at(1.0),
    metrics=[sl.Metrics.eval.f1]
)

# Pick your threshold with confidence
partition = ef["splink_try1"].at(0.87)  # Instant, even at weird thresholds
entities = partition.to_list()
```

Built with Rust for speed, wrapped in Python for joy. Because entity resolution is hard enough without your tools getting in the way.
