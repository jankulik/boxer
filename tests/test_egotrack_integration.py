"""Exercise EgoTrack integration without datasets, model weights, or a GPU."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from boxernet.dinov3_wrapper import DINOV3_MODEL_FUNC, DinoV3Wrapper
from loaders.aria_loader import AriaLoader
from loaders.hd_epic_loader import (
    looks_like_hd_epic_input,
    resolve_hd_epic_slam_root,
    resolve_hd_epic_vrs_path,
)


class TestAriaTimestampSelection(unittest.TestCase):
    def setUp(self):
        self.loader = AriaLoader.__new__(AriaLoader)
        times = [100, 200, 300]
        self.loader.stream_id = ["rgb"]
        self.loader.provider = SimpleNamespace(
            get_num_data=lambda stream: len(times),
            get_image_data_by_index=lambda stream, index: (
                None,
                SimpleNamespace(capture_timestamp_ns=times[index]),
            ),
        )

    def test_nearest_frame_and_boundaries(self):
        for timestamp, expected in [
            (0, 0),
            (100, 0),
            (149, 0),
            (150, 0),
            (151, 1),
            (200, 1),
            (299, 2),
            (999, 2),
        ]:
            with self.subTest(timestamp=timestamp):
                self.assertEqual(
                    self.loader.find_frame_by_timestamp(timestamp), expected
                )
                self.assertEqual(
                    self.loader._find_frame_by_timestamp(timestamp), expected
                )

    def test_load_by_timestamp_uses_nearest_frame(self):
        self.loader.load = Mock(return_value="datum")
        self.assertEqual(self.loader.load_by_timestamp(149), "datum")
        self.loader.load.assert_called_once_with(0)


class TestHDEpicPaths(unittest.TestCase):
    def test_video_id_and_path_resolve_to_same_vrs(self):
        video_id = "P08-20240614-085000"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vrs = root / "VRS" / "P08" / f"{video_id}_anonymized.vrs"
            vrs.parent.mkdir(parents=True)
            vrs.touch()
            self.assertEqual(
                resolve_hd_epic_vrs_path(video_id, directory), (vrs, video_id)
            )
            self.assertEqual(resolve_hd_epic_vrs_path(str(vrs)), (vrs, video_id))
            self.assertTrue(looks_like_hd_epic_input(str(vrs)))
            self.assertFalse(looks_like_hd_epic_input("Apartment_release_golden"))

    def test_mapping_resolves_slam_and_reports_unextracted_archive(self):
        video_id = "P08-20240614-085000"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vrs = root / "VRS" / "P08" / f"{video_id}.vrs"
            vrs.parent.mkdir(parents=True)
            vrs.touch()
            multi = root / "SLAM-and-Gaze" / "P08" / "SLAM" / "multi"
            slam = multi / "7" / "slam"
            slam.mkdir(parents=True)
            (multi / "vrs_to_multi_slam.json").write_text(json.dumps({str(vrs): 7}))
            self.assertEqual(resolve_hd_epic_slam_root(vrs, video_id, directory), slam)
            slam.rmdir()
            (multi / "7.zip").touch()
            with self.assertRaisesRegex(FileNotFoundError, "has not been extracted"):
                resolve_hd_epic_slam_root(vrs, video_id, directory)

    def test_invalid_video_id_is_rejected(self):
        with self.assertRaisesRegex(FileNotFoundError, "Could not infer"):
            resolve_hd_epic_vrs_path("invalid-video-id")


class TestLocalDinoWeights(unittest.TestCase):
    def test_explicit_weights_are_used_without_download(self):
        factory = Mock(return_value=torch.nn.Identity())
        with patch.dict(DINOV3_MODEL_FUNC, {"dinov3_vits16plus": factory}):
            model = DinoV3Wrapper(weights_path="/models/local-dino.pth")
        factory.assert_called_once_with(
            pretrained=True, weights="/models/local-dino.pth"
        )
        self.assertFalse(model.model.training)


if __name__ == "__main__":
    unittest.main()
