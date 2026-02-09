from __future__ import annotations

RUN_STEPS_FIELDNAMES = [
    "ts_utc",
    "run_name",
    "stage",
    "dt_s",
    "detail",
]

CLASS_METRICS_FIELDNAMES = [
    "ts_utc",
    "split",
    "class_id",
    "class_name",
    "precision",
    "recall",
    "f1",
    "support",
]

TEST_METRICS_FIELDNAMES = [
    "ts_utc",
    "run_id",
    "test_loss",
    "test_acc",
    "test_f1_macro",
    "test_precision_macro",
    "test_recall_macro",
    "test_balanced_acc",
]

TRAIN_EPOCH_METRICS_FIELDNAMES = [
    "ts_utc",
    "run_id",
    "epoch",
    "epoch_start_utc",
    "epoch_end_utc",
    "epoch_s",
    "train_loss",
    "train_acc",
    "train_f1_macro",
    "train_precision_macro",
    "train_recall_macro",
    "train_balanced_acc",
    "val_loss",
    "val_acc",
    "val_f1_macro",
    "val_precision_macro",
    "val_recall_macro",
    "val_balanced_acc",
    "lr",
    "is_best",
]

KD_EPOCH_METRICS_FIELDNAMES = [
    "ts_utc",
    "run_id",
    "epoch",
    "epoch_start_utc",
    "epoch_end_utc",
    "epoch_s",
    "train_loss",
    "train_acc",
    "train_f1_macro",
    "train_precision_macro",
    "train_recall_macro",
    "train_balanced_acc",
    "val_loss",
    "val_acc",
    "val_f1_macro",
    "val_precision_macro",
    "val_recall_macro",
    "val_balanced_acc",
    "lr",
    "alpha",
    "tau",
    "is_best",
]
