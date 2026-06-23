import json
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


BENCHMARK_ROOT = Path("/data01/sosui-ko/anime_colorization/benchmark/geometric_sequences")


def _read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _seg_ids(seg_rgb):
    return (
        (seg_rgb[:, :, 0].astype(np.int32) << 16)
        | (seg_rgb[:, :, 1].astype(np.int32) << 8)
        | seg_rgb[:, :, 2].astype(np.int32)
    )


def _load_json(path):
    with Path(path).open("r") as f:
        return json.load(f)


def _resolve_benchmark_root(test_annotation_file):
    annotation_path = Path(test_annotation_file)
    if annotation_path.is_dir():
        return annotation_path
    if annotation_path.is_file():
        try:
            data = json.loads(annotation_path.read_text())
            candidate = data.get("benchmark_root")
            if candidate and Path(candidate).is_dir():
                return Path(candidate)
        except json.JSONDecodeError:
            for line in annotation_path.read_text().splitlines():
                candidate = Path(line.strip())
                if candidate.is_dir():
                    return candidate
    return BENCHMARK_ROOT


def _load_submission_json(user_submission_file):
    submission_path = Path(user_submission_file)
    if submission_path.is_file():
        try:
            return _load_json(submission_path), None
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    if submission_path.is_file() and zipfile.is_zipfile(submission_path):
        tmp = tempfile.TemporaryDirectory(prefix="evalai_code_upload_")
        with zipfile.ZipFile(submission_path, "r") as zf:
            zf.extractall(tmp.name)
        extracted = Path(tmp.name)
        for candidate in [extracted / "submission.json", *extracted.rglob("submission.json")]:
            if candidate.is_file():
                return _load_json(candidate), tmp
        raise ValueError("submission.json was not found in the submitted zip file.")
    raise ValueError("Code upload evaluation expects a submission.json file.")


def _palette_from_metadata(metadata):
    palette = []
    for shape in metadata.get("shapes", []):
        color = tuple(int(v) for v in shape["color"])
        if color not in palette:
            palette.append(color)
    return palette


def _segment_gt_labels(metadata, palette):
    labels = {}
    for shape in metadata.get("shapes", []):
        seg_rgb = tuple(int(v) for v in shape["segment_rgb_id"])
        seg_id = (seg_rgb[0] << 16) | (seg_rgb[1] << 8) | seg_rgb[2]
        color = tuple(int(v) for v in shape["color"])
        labels[seg_id] = palette.index(color)
    return labels


def _normalize_segment_key(key):
    if isinstance(key, int):
        return key
    key = str(key).strip()
    if key.startswith("#") and len(key) == 7:
        return int(key[1:], 16)
    if "," in key:
        parts = [int(x.strip()) for x in key.split(",")]
        if len(parts) == 3:
            return (parts[0] << 16) | (parts[1] << 8) | parts[2]
    return int(key)


def _color_to_palette_label(value, palette):
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("#") and len(value) == 7:
            rgb = tuple(int(value[i : i + 2], 16) for i in (1, 3, 5))
        else:
            return int(value)
    else:
        rgb = tuple(int(v) for v in value)
    distances = [sum((rgb[i] - color[i]) ** 2 for i in range(3)) for color in palette]
    return int(np.argmin(np.asarray(distances)))


def _frame_predictions(predictions, sample_name, frame_name):
    sample_predictions = predictions.get(sample_name, {})
    frame_predictions = sample_predictions.get(frame_name)
    if frame_predictions is None:
        frame_predictions = sample_predictions.get(Path(frame_name).stem)
    if frame_predictions is None:
        frame_predictions = predictions.get(f"{sample_name}/{frame_name}")
    if frame_predictions is None:
        frame_predictions = predictions.get(f"{sample_name}/{Path(frame_name).stem}")
    if frame_predictions is None:
        return {}
    return frame_predictions.get("segments", frame_predictions)


def _evaluate_predictions(benchmark_root, submission):
    predictions = submission.get("predictions", submission)
    totals = {
        "segment_correct": 0,
        "segment_predicted": 0,
        "segment_total": 0,
        "frames_with_predictions": 0,
        "frames_total": 0,
    }

    for sample_dir in sorted(p for p in benchmark_root.iterdir() if p.is_dir()):
        metadata = _load_json(sample_dir / "metadata.json")
        palette = _palette_from_metadata(metadata)
        gt_labels = _segment_gt_labels(metadata, palette)
        seg_dir = sample_dir / "seg"

        for seg_path in sorted(seg_dir.glob("*.png")):
            totals["frames_total"] += 1
            seg_id_map = _seg_ids(_read_rgb(seg_path))
            visible_segment_ids = [
                seg_id for seg_id in gt_labels if np.any(seg_id_map == seg_id)
            ]
            frame_pred = _frame_predictions(predictions, sample_dir.name, seg_path.name)
            normalized_pred = {}
            for key, value in frame_pred.items():
                try:
                    normalized_pred[_normalize_segment_key(key)] = _color_to_palette_label(
                        value, palette
                    )
                except (TypeError, ValueError):
                    continue
            if normalized_pred:
                totals["frames_with_predictions"] += 1

            for seg_id in visible_segment_ids:
                totals["segment_total"] += 1
                if seg_id not in normalized_pred:
                    continue
                totals["segment_predicted"] += 1
                totals["segment_correct"] += int(normalized_pred[seg_id] == gt_labels[seg_id])

    segment_accuracy = (
        totals["segment_correct"] / totals["segment_total"] if totals["segment_total"] else 0.0
    )
    prediction_coverage = (
        totals["segment_predicted"] / totals["segment_total"] if totals["segment_total"] else 0.0
    )
    frame_coverage = (
        totals["frames_with_predictions"] / totals["frames_total"] if totals["frames_total"] else 0.0
    )
    score = (segment_accuracy + prediction_coverage) / 2.0
    return {
        "score": float(score),
        "segment_accuracy": float(segment_accuracy),
        "prediction_coverage": float(prediction_coverage),
        "frame_coverage": float(frame_coverage),
        **totals,
    }


def evaluate(test_annotation_file, user_submission_file, phase_name, **kwargs):
    tmp_dir = None
    try:
        benchmark_root = _resolve_benchmark_root(test_annotation_file)
        submission, tmp_dir = _load_submission_json(user_submission_file)
        metrics = _evaluate_predictions(benchmark_root, submission)
        leaderboard_metrics = {
            "score": metrics["score"],
            "segment_accuracy": metrics["segment_accuracy"],
            "prediction_coverage": metrics["prediction_coverage"],
            "frame_coverage": metrics["frame_coverage"],
        }
        return {
            "result": [{"test_split": leaderboard_metrics}],
            "submission_metadata": {
                "phase_name": phase_name,
                "benchmark_root": str(benchmark_root),
                "segment_correct": metrics["segment_correct"],
                "segment_predicted": metrics["segment_predicted"],
                "segment_total": metrics["segment_total"],
                "frames_with_predictions": metrics["frames_with_predictions"],
                "frames_total": metrics["frames_total"],
            },
            "submission_result": leaderboard_metrics,
        }
    except Exception as e:
        sys.stderr.write(traceback.format_exc())
        return e
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()
