import json
import tempfile
import unittest
from pathlib import Path

from loaders.hd_epic_loader import (
    resolve_hd_epic_slam_root,
    resolve_hd_epic_vrs_path,
)


class TestHDEpicLoaderHelpers(unittest.TestCase):
    def test_resolve_hd_epic_vrs_and_slam_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "HD-EPIC"
            participant = "P08"
            video_id = "P08-20240614-085000"
            vrs_dir = root / "VRS" / participant
            slam_multi_dir = root / "SLAM-and-Gaze" / participant / "SLAM" / "multi"
            slam_root = slam_multi_dir / "3" / "slam"
            vrs_dir.mkdir(parents=True)
            slam_root.mkdir(parents=True)

            vrs_path = vrs_dir / f"{video_id}_anonymized.vrs"
            vrs_path.write_bytes(b"")
            (slam_multi_dir / "vrs_to_multi_slam.json").write_text(
                json.dumps({f"/dataset/{video_id}_anonymized.vrs": "3"})
            )

            resolved_vrs_path, resolved_video_id = resolve_hd_epic_vrs_path(video_id, root)
            resolved_slam_root = resolve_hd_epic_slam_root(
                resolved_vrs_path, resolved_video_id, root
            )

            self.assertEqual(resolved_vrs_path, vrs_path.resolve())
            self.assertEqual(resolved_video_id, video_id)
            self.assertEqual(resolved_slam_root, slam_root.resolve())
