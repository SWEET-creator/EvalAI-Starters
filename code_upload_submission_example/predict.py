import json
import os
from pathlib import Path


DATASET_ROOT = Path(os.getenv("DATASET_ROOT", "/dataset"))
SUBMISSION_PATH = Path(os.getenv("SUBMISSION_PATH", "/submission"))


def segment_id(segment_rgb_id):
    r, g, b = [int(v) for v in segment_rgb_id]
    return (r << 16) | (g << 8) | b


def main():
    predictions = {}
    for sample_dir in sorted(DATASET_ROOT.glob("geometric_*")):
        metadata_path = sample_dir / "metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text())
        sample_predictions = {}
        segment_predictions = {
            str(segment_id(shape["segment_rgb_id"])): shape["color"]
            for shape in metadata.get("shapes", [])
        }
        frame_names = sorted(path.name for path in (sample_dir / "line").glob("*.png"))
        if not frame_names:
            frame_names = [f"{idx:04d}.png" for idx in range(int(metadata.get("num_frames", 0)))]
        for frame_name in frame_names:
            sample_predictions[frame_name] = {"segments": segment_predictions}
        predictions[sample_dir.name] = sample_predictions

    SUBMISSION_PATH.mkdir(parents=True, exist_ok=True)
    (SUBMISSION_PATH / "submission.json").write_text(
        json.dumps({"predictions": predictions}, separators=(",", ":"))
    )


if __name__ == "__main__":
    main()
