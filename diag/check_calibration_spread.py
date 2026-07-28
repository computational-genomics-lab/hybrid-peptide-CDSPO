"""
Diagnostic: run once, after training, on your real test set.
Reports the actual spread of calibrated probabilities the generation
loop is steering on — not AUC, the raw distribution.
"""
import numpy as np

def diagnose(predict_fn, test_seqs, test_labels, label='AMP'):
    y = np.array(test_labels)
    p = predict_fn(list(test_seqs))

    pos = p[y == 1]
    neg = p[y == 0]

    print(f"--- {label} calibrated score spread (test set, n={len(p)}) ---")
    print(f"  positives: mean={pos.mean():.3f}  std={pos.std():.4f}  "
          f"min={pos.min():.3f}  max={pos.max():.3f}  unique_values={len(np.unique(pos))}")
    print(f"  negatives: mean={neg.mean():.3f}  std={neg.std():.4f}  "
          f"min={neg.min():.3f}  max={neg.max():.3f}  unique_values={len(np.unique(neg))}")
    print(f"  overall unique probability values in test set: {len(np.unique(p))} / {len(p)}")
    print(f"  fraction of test set within ±0.02 of the median: "
          f"{np.mean(np.abs(p - np.median(p)) < 0.02):.2%}")

    # This is the number that actually matters for your hypothesis:
    if len(np.unique(p)) < 15:
        print(f"  ⚠ Only {len(np.unique(p))} distinct probability values across "
              f"{len(p)} test sequences — isotonic has collapsed to a near-step "
              f"function. This WILL flatten softmax_sample_seeds().")
