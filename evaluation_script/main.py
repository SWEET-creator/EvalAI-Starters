import json
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


BENCHMARK_ROOT = Path("/data01/sosui-ko/anime_colorization/benchmark/geometric_sequences")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


def _read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _seg_ids(seg_rgb):
    return (
        (seg_rgb[:, :, 0].astype(np.int32) << 16)
        | (seg_rgb[:, :, 1].astype(np.int32) << 8)
        | seg_rgb[:, :, 2].astype(np.int32)
    )


def _load_sample_metadata(sample_dir):
    with (sample_dir / "metadata.json").open("r") as f:
        return json.load(f)


def _palette_from_metadata(metadata):
    palette = []
    for shape in metadata.get("shapes", []):
        color = tuple(int(v) for v in shape["color"])
        if color not in palette:
            palette.append(color)
    return palette


def _segment_palette_labels(metadata, palette):
    labels = {}
    for shape in metadata.get("shapes", []):
        seg_rgb = tuple(int(v) for v in shape["segment_rgb_id"])
        seg_id = (seg_rgb[0] << 16) | (seg_rgb[1] << 8) | seg_rgb[2]
        color = tuple(int(v) for v in shape["color"])
        labels[seg_id] = palette.index(color)
    return labels


def _quantize_to_palette(image_rgb, palette):
    palette_arr = np.asarray(palette, dtype=np.int16)
    pixels = image_rgb.astype(np.int16)
    distances = np.sum((pixels[:, :, None, :] - palette_arr[None, None, :, :]) ** 2, axis=3)
    return np.argmin(distances, axis=2).astype(np.int32)


def _majority_label(labels):
    if labels.size == 0:
        return None
    values, counts = np.unique(labels, return_counts=True)
    return int(values[np.argmax(counts)])


def _resolve_benchmark_root(test_annotation_file):
    annotation_path = Path(test_annotation_file)
    if annotation_path.is_dir():
        return annotation_path

    if annotation_path.is_file():
        for line in annotation_path.read_text().splitlines():
            candidate = Path(line.strip())
            if candidate.is_dir():
                return candidate

    return BENCHMARK_ROOT


def _prepare_submission_root(user_annotation_file):
    submission_path = Path(user_annotation_file)
    if submission_path.is_dir():
        return submission_path, None

    if zipfile.is_zipfile(submission_path):
        tmp = tempfile.TemporaryDirectory(prefix="evalai_submission_")
        with zipfile.ZipFile(submission_path, "r") as zf:
            zf.extractall(tmp.name)
        return Path(tmp.name), tmp

    return submission_path.parent, None


def _prediction_candidates(submission_root, sample_name, frame_name):
    stem = Path(frame_name).stem
    candidates = []
    for ext in IMAGE_EXTS:
        candidates.extend(
            [
                submission_root / sample_name / "gt" / f"{stem}{ext}",
                submission_root / sample_name / "colorized" / f"{stem}{ext}",
                submission_root / sample_name / f"{stem}{ext}",
                submission_root / "gt" / sample_name / f"{stem}{ext}",
                submission_root / "colorized" / sample_name / f"{stem}{ext}",
                submission_root / f"{sample_name}_{stem}{ext}",
                submission_root / f"{sample_name}-{stem}{ext}",
            ]
        )
    return candidates


def _find_prediction(submission_root, sample_name, frame_name):
    for candidate in _prediction_candidates(submission_root, sample_name, frame_name):
        if candidate.is_file():
            return candidate

    suffixes = {
        f"{sample_name}/gt/{frame_name}",
        f"{sample_name}/colorized/{frame_name}",
        f"{sample_name}/{frame_name}",
        f"gt/{sample_name}/{frame_name}",
        f"colorized/{sample_name}/{frame_name}",
    }
    matches = []
    for path in submission_root.rglob(frame_name):
        if not path.is_file():
            continue
        rel = path.relative_to(submission_root).as_posix()
        if rel in suffixes or rel.endswith(f"/{sample_name}/{frame_name}"):
            matches.append(path)

    return sorted(matches)[0] if matches else None


def _missing_frame_metrics(gt_rgb, seg_rgb, palette, segment_gt_labels):
    seg_id_map = _seg_ids(seg_rgb)
    gt_labels = _quantize_to_palette(gt_rgb, palette)

    segment_total = 0
    for seg_id in segment_gt_labels.keys():
        if np.any(seg_id_map == seg_id):
            segment_total += 1

    valid_mask = np.isin(seg_id_map, list(segment_gt_labels.keys()))
    miou_classes = 0
    for class_id in range(len(palette)):
        if np.any((gt_labels == class_id) & valid_mask):
            miou_classes += 1

    return {
        "segment_correct": 0,
        "segment_total": segment_total,
        "miou_sum": 0.0,
        "miou_classes": miou_classes,
    }


def _compute_frame_metrics(
    pred_rgb,
    gt_rgb,
    seg_rgb,
    palette,
    segment_gt_labels,
):
    if pred_rgb.shape != gt_rgb.shape:
        pred_rgb = np.asarray(
            Image.fromarray(pred_rgb).resize((gt_rgb.shape[1], gt_rgb.shape[0]), Image.NEAREST),
            dtype=np.uint8,
        )

    seg_id_map = _seg_ids(seg_rgb)
    pred_labels = _quantize_to_palette(pred_rgb, palette)
    gt_labels = _quantize_to_palette(gt_rgb, palette)

    correct_segments = 0
    total_segments = 0
    for seg_id in segment_gt_labels.keys():
        mask = seg_id_map == seg_id
        if not np.any(mask):
            continue
        gt_label = _majority_label(gt_labels[mask])
        pred_label = _majority_label(pred_labels[mask])
        if gt_label is None or pred_label is None:
            continue
        total_segments += 1
        correct_segments += int(pred_label == gt_label)

    valid_mask = np.isin(seg_id_map, list(segment_gt_labels.keys()))
    intersections = np.zeros(len(palette), dtype=np.float64)
    unions = np.zeros(len(palette), dtype=np.float64)
    for class_id in range(len(palette)):
        pred_mask = (pred_labels == class_id) & valid_mask
        gt_mask = (gt_labels == class_id) & valid_mask
        intersections[class_id] = np.logical_and(pred_mask, gt_mask).sum()
        unions[class_id] = np.logical_or(pred_mask, gt_mask).sum()

    valid_classes = unions > 0
    class_ious = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=np.float64),
        where=valid_classes,
    )

    return {
        "segment_correct": correct_segments,
        "segment_total": total_segments,
        "miou_sum": float(class_ious[valid_classes].sum()),
        "miou_classes": int(valid_classes.sum()),
    }


def _evaluate_dataset(benchmark_root, submission_root):
    totals = {
        "segment_correct": 0,
        "segment_total": 0,
        "miou_sum": 0.0,
        "miou_classes": 0,
        "frames_total": 0,
        "frames_evaluated": 0,
        "missing_predictions": [],
    }

    for sample_dir in sorted(p for p in benchmark_root.iterdir() if p.is_dir()):
        metadata = _load_sample_metadata(sample_dir)
        palette = _palette_from_metadata(metadata)
        if not palette:
            raise ValueError(f"No palette colors found in {sample_dir / 'metadata.json'}")
        segment_gt_labels = _segment_palette_labels(metadata, palette)

        gt_dir = sample_dir / "gt"
        seg_dir = sample_dir / "seg"
        for gt_path in sorted(gt_dir.glob("*.png")):
            totals["frames_total"] += 1
            pred_path = _find_prediction(submission_root, sample_dir.name, gt_path.name)
            seg_path = seg_dir / gt_path.name
            gt_rgb = _read_rgb(gt_path)
            seg_rgb = _read_rgb(seg_path)
            if pred_path is None:
                totals["missing_predictions"].append(f"{sample_dir.name}/{gt_path.name}")
                frame_metrics = _missing_frame_metrics(
                    gt_rgb=gt_rgb,
                    seg_rgb=seg_rgb,
                    palette=palette,
                    segment_gt_labels=segment_gt_labels,
                )
            else:
                frame_metrics = _compute_frame_metrics(
                    pred_rgb=_read_rgb(pred_path),
                    gt_rgb=gt_rgb,
                    seg_rgb=seg_rgb,
                    palette=palette,
                    segment_gt_labels=segment_gt_labels,
                )
                totals["frames_evaluated"] += 1
            totals["segment_correct"] += frame_metrics["segment_correct"]
            totals["segment_total"] += frame_metrics["segment_total"]
            totals["miou_sum"] += frame_metrics["miou_sum"]
            totals["miou_classes"] += frame_metrics["miou_classes"]

    segment_accuracy = (
        totals["segment_correct"] / totals["segment_total"] if totals["segment_total"] else 0.0
    )
    palette_miou = totals["miou_sum"] / totals["miou_classes"] if totals["miou_classes"] else 0.0
    score = (segment_accuracy + palette_miou) / 2.0

    return {
        "score": float(score),
        "segment_accuracy": float(segment_accuracy),
        "palette_miou": float(palette_miou),
        "frames_evaluated": int(totals["frames_evaluated"]),
        "frames_total": int(totals["frames_total"]),
        "segments_evaluated": int(totals["segment_total"]),
        "missing_predictions": totals["missing_predictions"],
    }


def evaluate(test_annotation_file, user_annotation_file, phase_name, **kwargs):
    tmp_dir = None
    try:
        benchmark_root = _resolve_benchmark_root(test_annotation_file)
        submission_root, tmp_dir = _prepare_submission_root(user_annotation_file)
        metrics = _evaluate_dataset(benchmark_root, submission_root)
        frame_coverage = (
            metrics["frames_evaluated"] / metrics["frames_total"]
            if metrics["frames_total"]
            else 0.0
        )
        leaderboard_metrics = {
            "Metric1": metrics["segment_accuracy"],
            "Metric2": metrics["palette_miou"],
            "Metric3": frame_coverage,
            "Total": metrics["score"],
        }

        if phase_name == "dev":
            result = [{"train_split": leaderboard_metrics}]
            submission_result = leaderboard_metrics
        else:
            result = [
                {"train_split": leaderboard_metrics},
                {"test_split": leaderboard_metrics},
            ]
            submission_result = result[0]

        return {
            "result": result,
            "submission_metadata": {
                "phase_name": phase_name,
                "benchmark_root": str(benchmark_root),
                "frames_evaluated": metrics["frames_evaluated"],
                "frames_total": metrics["frames_total"],
                "segments_evaluated": metrics["segments_evaluated"],
                "missing_predictions_count": len(metrics["missing_predictions"]),
                "missing_predictions": metrics["missing_predictions"][:20],
            },
            "submission_result": (
                f"score={metrics['score']:.6f}, "
                f"segment_accuracy={metrics['segment_accuracy']:.6f}, "
                f"palette_miou={metrics['palette_miou']:.6f}, "
                f"frames={metrics['frames_evaluated']}/{metrics['frames_total']}"
            ),
        }
    except Exception as e:
        sys.stderr.write(traceback.format_exc())
        return e
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()
