"""
Live terminal progress bar for a training run's history CSV. Run once, leave the window
open -- it polls the CSV every 2s and redraws in place (no need to ask Claude for updates).

Usage: python scripts/watch_progress.py experiments/mask2former_v3_history.csv --epochs 50
"""
import argparse
import csv
import os
import time


def read_history(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("history_csv")
    ap.add_argument("--epochs", type=int, required=True, help="Total planned epochs")
    ap.add_argument("--poll", type=float, default=2.0)
    args = ap.parse_args()

    start = time.time()
    last_epoch = 0
    epoch_times = []

    while True:
        rows = read_history(args.history_csv)
        n = len(rows)
        if n > last_epoch:
            epoch_times.append(time.time())
            last_epoch = n

        bar_width = 40
        filled = int(bar_width * n / args.epochs) if args.epochs else 0
        bar = "#" * filled + "-" * (bar_width - filled)
        pct = 100 * n / args.epochs if args.epochs else 0

        if n > 0:
            last = rows[-1]
            val_dice = last.get("val_dice", "?")
            val_iou = last.get("val_iou", "?")
            best_dice = max((float(r["val_dice"]) for r in rows), default=0.0)
        else:
            val_dice = val_iou = "?"
            best_dice = 0.0

        # Real per-epoch duration from actual epoch-to-epoch gaps this watcher has observed,
        # not (watcher's own runtime so far) / (epochs done) -- that ratio is meaningless if
        # the watcher started mid-epoch. Uses at most the last 5 gaps (a rolling window) so
        # one anomalously slow/fast epoch early on doesn't dominate the average forever --
        # with only 2 data points a single bad epoch swings the ETA wildly.
        window = epoch_times[-6:]
        if len(window) >= 2:
            avg_epoch_time = (window[-1] - window[0]) / (len(window) - 1)
            remaining = (args.epochs - n) * avg_epoch_time
            eta_str = f"ETA={remaining/60:.0f}min"
        else:
            eta_str = "ETA=calculating..."

        if n > 0:
            body = (f"[{bar}] {pct:5.1f}%  epoch {n}/{args.epochs}  "
                    f"last_dice={float(val_dice):.4f}  last_iou={float(val_iou):.4f}  "
                    f"best_dice={best_dice:.4f}  {eta_str}")
        else:
            body = f"[{bar}] {pct:5.1f}%  epoch {n}/{args.epochs}  waiting for epoch 1..."
        # Pad to a fixed width so a shorter new line always fully overwrites a longer old
        # one -- \r alone doesn't erase leftover characters past the new line's end, which
        # is what was making this look garbled/mixed rather than cleanly redrawing in place.
        line = "\r" + body.ljust(110)
        print(line, end="", flush=True)

        if args.epochs and n >= args.epochs:
            print("\nDone.")
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
