import json
import tempfile
import unittest
from pathlib import Path

from loaders.hd_epic_loader import (
    load_mp4_to_vrs_mapping,
    resolve_hd_epic_slam_root,
    resolve_hd_epic_video_path,
)


class TestHDEpicLoaderHelpers(unittest.TestCase):
    def test_load_mp4_to_vrs_mapping_with_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sample.csv"
            csv_path.write_text("frame_idx,vrs_time_ns\n0,100\n1,200\n")

            frame_ids, timestamps = load_mp4_to_vrs_mapping(csv_path)

            self.assertEqual(frame_ids.tolist(), [0, 1])
            self.assertEqual(timestamps.tolist(), [100, 200])

    def test_resolve_hd_epic_video_and_slam_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "HD-EPIC"
            participant = "P08"
            video_id = "P08-20240614-085000"
            video_dir = root / "Videos" / participant
            slam_multi_dir = root / "SLAM-and-Gaze" / participant / "SLAM" / "multi"
            slam_root = slam_multi_dir / "3" / "slam"
            video_dir.mkdir(parents=True)
            slam_root.mkdir(parents=True)

            video_path = video_dir / f"{video_id}.mp4"
            video_path.write_bytes(b"")
            (video_dir / f"{video_id}_mp4_to_vrs_time_ns.csv").write_text("0,1\n")
            (slam_multi_dir / "vrs_to_multi_slam.json").write_text(
                json.dumps({f"/dataset/{video_id}_anonymized.vrs": "3"})
            )

            resolved_video_path, resolved_video_id = resolve_hd_epic_video_path(video_id, root)
            resolved_slam_root = resolve_hd_epic_slam_root(
                resolved_video_path, resolved_video_id, root
            )

            self.assertEqual(resolved_video_path, video_path.resolve())
            self.assertEqual(resolved_video_id, video_id)
            self.assertEqual(resolved_slam_root, slam_root.resolve())
