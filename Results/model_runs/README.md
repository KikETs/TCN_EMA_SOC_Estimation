# Local Model Run Outputs

`Models/train.py` writes local training outputs here by default:

```text
Results/model_runs/<run_id>/
  run_config.json
  input_schema.csv
  scaler_stats.csv
  metrics_history.csv
  summary_metrics.csv
  by_temperature.csv
  test_predictions.csv
```

These run directories are intentionally ignored by Git because they may contain large intermediate prediction rows or optional checkpoints.
