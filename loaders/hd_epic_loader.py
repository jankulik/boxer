# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch

from loaders.base_loader import BaseLoader
from utils.file_io import load_closed_loop_trajectory, load_online_calib, load_semidense
from utils.tw.camera import CameraTW
from utils.tw.tensor_utils import find_nearest2

HD_EPIC_VIDEO_ID_RE = re.compile(r"^(P0[1-9]-2024\d{4}-\d{6})$")


def _create_vrs_provider(vrs_path: str):
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("VRS_LOG_LEVEL", "ERROR")
    from projectaria_tools.core import data_provider

    return data_provider.create_vrs_data_provider(vrs_path)


def looks_like_hd_epic_input(value: str) -> bool:
    if not value:
        return False
    norm = value.replace("\\", "/")
    base = os.path.basename(norm)
    stem = os.path.splitext(base)[0]
    stem = stem.removesuffix("_anonymized")
    if HD_EPIC_VIDEO_ID_RE.match(stem):
        return True
    return "/HD-EPIC/" in norm or "/VRS/P0" in norm


def _candidate_hd_epic_roots(input_path: str, hd_epic_root: str | None = None) -> list[Path]:
    roots = []
    if hd_epic_root:
        roots.append(Path(hd_epic_root).expanduser())

    p = Path(input_path).expanduser()
    if p.exists():
        cur = p if p.is_dir() else p.parent
        for parent in [cur, *cur.parents]:
            roots.append(parent)
            if parent.name == "HD-EPIC":
                roots.append(parent)
            if parent.name == "Videos":
                roots.append(parent.parent)
    else:
        roots.append(Path.cwd())
        roots.append(Path.cwd() / "HD-EPIC")
        roots.append(Path.home() / "HD-EPIC")
        roots.append(Path.home() / "datasets" / "HD-EPIC")

    uniq = []
    seen = set()
    for root in roots:
        try:
            key = str(root.resolve())
        except FileNotFoundError:
            key = str(root)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(root)
    return uniq


def _extract_hd_epic_video_id(path_or_id: str) -> str:
    stem = Path(path_or_id).stem.removesuffix("_anonymized")
    if not HD_EPIC_VIDEO_ID_RE.match(stem):
        raise FileNotFoundError(
            f"Could not infer an HD-EPIC video id from '{path_or_id}'. "
            "Pass a video id like P08-20240614-085000 or a matching .vrs path."
        )
    return stem


def resolve_hd_epic_vrs_path(
    input_path: str, hd_epic_root: str | None = None
) -> tuple[Path, str]:
    input_as_path = Path(input_path).expanduser()
    if input_as_path.exists():
        vrs_path = input_as_path.resolve()
        if vrs_path.suffix.lower() != ".vrs":
            raise FileNotFoundError(f"HD-EPIC input must point to a .vrs file, got: {vrs_path}")
        return vrs_path, _extract_hd_epic_video_id(vrs_path.name)

    video_id = _extract_hd_epic_video_id(input_path)

    participant = video_id.split("-")[0]
    for root in _candidate_hd_epic_roots(input_path, hd_epic_root):
        candidates = [
            root / "VRS" / participant / f"{video_id}_anonymized.vrs",
            root / "VRS" / participant / f"{video_id}.vrs",
            root / "HD-EPIC" / "VRS" / participant / f"{video_id}_anonymized.vrs",
            root / "HD-EPIC" / "VRS" / participant / f"{video_id}.vrs",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve(), video_id

    raise FileNotFoundError(
        f"Could not find HD-EPIC VRS {video_id}. "
        "Looked under typical HD-EPIC roots; try --hd_epic_root."
    )


def resolve_hd_epic_slam_root(
    vrs_path: str | os.PathLike, video_id: str, hd_epic_root: str | None = None
) -> Path:
    vrs_path = Path(vrs_path).expanduser().resolve()
    participant = video_id.split("-")[0]
    multi_roots = []
    for root in _candidate_hd_epic_roots(str(vrs_path), hd_epic_root):
        multi_roots.extend(
            [
                root / "SLAM-and-Gaze" / participant / "SLAM" / "multi",
                root / "HD-EPIC" / "SLAM-and-Gaze" / participant / "SLAM" / "multi",
            ]
        )

    checked = []
    for multi_root in multi_roots:
        mapping_path = multi_root / "vrs_to_multi_slam.json"
        checked.append(str(mapping_path))
        if not mapping_path.exists():
            continue

        with open(mapping_path, "r") as f:
            vrs_to_multi = json.load(f)

        folder_idx = None
        target_suffixes = [
            f"{video_id}_anonymized.vrs",
            f"{video_id}.vrs",
        ]
        for vrs_name, idx in vrs_to_multi.items():
            if any(str(vrs_name).endswith(suffix) for suffix in target_suffixes):
                folder_idx = str(idx)
                break
        if folder_idx is None:
            continue

        slam_root = multi_root / folder_idx / "slam"
        if slam_root.exists():
            return slam_root.resolve()

        zip_path = multi_root / f"{folder_idx}.zip"
        if zip_path.exists():
            raise FileNotFoundError(
                f"Found HD-EPIC SLAM archive {zip_path}, but it has not been extracted. "
                f"Please extract {zip_path} so that {multi_root / folder_idx / 'slam'} exists."
            )

    raise FileNotFoundError(
        f"Could not resolve SLAM outputs for {video_id}. "
        f"Checked: {checked}. Try --hd_epic_root or extract the matching multi-SLAM zip."
    )

class HDEpicLoader(BaseLoader):
    def __init__(
        self,
        input_path,
        hd_epic_root=None,
        camera="rgb",
        with_img=True,
        with_traj=True,
        with_sdp=True,
        with_obb=False,
        pinhole=False,
        pinhole_fxy=None,
        resize=None,
        skip_n=1,
        max_n=999999,
        start_n=0,
        force_reload=False,
        restrict_range=True,
        max_time_diff_s=0.05,
        prefer_sdp_aligned_frames=True,
        sdp_frame_alignment_s=0.015,
        use_global_sdp_for_rgb=True,
        global_sdp_max_points=50000,
    ):
        if camera != "rgb":
            raise ValueError("HD-EPIC video loader currently supports only --camera rgb")
        if with_obb:
            raise ValueError("HD-EPIC does not currently support --gt2d / ground-truth 2D boxes")

        self.vrs_path, self.video_id = resolve_hd_epic_vrs_path(input_path, hd_epic_root)
        self.slam_root = resolve_hd_epic_slam_root(self.vrs_path, self.video_id, hd_epic_root)

        self.camera = camera
        self.device_name = "HD-EPIC"
        self.with_img = with_img
        self.with_traj = with_traj
        self.with_sdp = with_sdp
        self.pinhole = pinhole
        self.pinhole_fxy = pinhole_fxy
        self.resize = resize
        self.skip_n = skip_n
        self.max_n = max_n
        self.force_reload = force_reload
        self.restrict_range = restrict_range
        self.max_time_diff_s = max_time_diff_s
        self.prefer_sdp_aligned_frames = prefer_sdp_aligned_frames
        self.sdp_frame_alignment_s = sdp_frame_alignment_s
        self.use_global_sdp_for_rgb = use_global_sdp_for_rgb
        self.global_sdp_max_points = global_sdp_max_points
        self._skip_debug_count = 0

        print("==> loading HDEpicLoader with the following settings:")
        print(f"vrs_path: {self.vrs_path}")
        print(f"slam_root: {self.slam_root}")
        print(f"camera: {camera}")
        print(f"with_img: {with_img}")
        print(f"with_traj: {with_traj}")
        print(f"with_sdp: {with_sdp}")
        print(f"pinhole: {pinhole}")
        print(f"resize: {resize}")
        print(f"skip_n: {skip_n}")
        print(f"max_n: {max_n}")
        print(f"start_n: {start_n}")
        print(f"restrict_range: {restrict_range}")
        print(f"max_time_diff_s: {max_time_diff_s}")
        print(f"prefer_sdp_aligned_frames: {prefer_sdp_aligned_frames}")
        print(f"sdp_frame_alignment_s: {sdp_frame_alignment_s}")
        print(f"use_global_sdp_for_rgb: {use_global_sdp_for_rgb}")
        print(f"global_sdp_max_points: {global_sdp_max_points}")

        print(f"==> HD-EPIC will use VRS images from {self.vrs_path}")
        self.provider = _create_vrs_provider(str(self.vrs_path))
        self.stream_id = self.provider.get_stream_id_from_label("camera-rgb")
        usable = int(self.provider.get_num_data(self.stream_id))
        if usable == 0:
            raise ValueError(f"No frames available for {self.video_id}")

        calib_path = self.slam_root / "online_calibration.jsonl"
        slaml_calibs, slamr_calibs, rgb_calibs, calib_ns = load_online_calib(str(calib_path))
        if rgb_calibs is None:
            raise ValueError(f"No RGB calibration found in {calib_path}")
        self.calibs = rgb_calibs
        self.calib_ts = calib_ns.numpy()

        if self.with_traj:
            traj_path = self.slam_root / "closed_loop_trajectory.csv"
            self.traj, pose_ts = load_closed_loop_trajectory(str(traj_path), subsample=5)
            self.pose_ts = pose_ts.numpy()

        if self.with_sdp:
            global_path = self.slam_root / "semidense_points.csv.gz"
            obs_path = self.slam_root / "semidense_observations.csv.gz"
            time_to_uids_slaml, time_to_uids_slamr, uid_to_p3 = load_semidense(
                str(global_path),
                str(obs_path),
                str(calib_path),
                force_reload=self.force_reload,
            )
            self.time_to_uids_slaml = time_to_uids_slaml
            self.time_to_uids_slamr = time_to_uids_slamr
            self.time_to_uids_combined = {}
            for k, v in time_to_uids_slaml.items():
                self.time_to_uids_combined[k] = list(v)
            for k, v in time_to_uids_slamr.items():
                if k in self.time_to_uids_combined:
                    self.time_to_uids_combined[k] = list(set(self.time_to_uids_combined[k] + v))
                else:
                    self.time_to_uids_combined[k] = list(v)
            self.sdp_times = np.array(sorted(self.time_to_uids_combined.keys()), dtype=np.int64)
            all_uids = list(uid_to_p3.keys())
            self.uid_to_idx = {uid: idx for idx, uid in enumerate(all_uids)}
            self.p3_array = np.array([uid_to_p3[uid][:3] for uid in all_uids], dtype=np.float32)
            if self.use_global_sdp_for_rgb:
                if len(self.p3_array) > self.global_sdp_max_points:
                    rng = np.random.default_rng(0)
                    keep = rng.choice(
                        len(self.p3_array), size=self.global_sdp_max_points, replace=False
                    )
                    keep.sort()
                    self.global_sdp_points = self.p3_array[keep]
                else:
                    self.global_sdp_points = self.p3_array
                print(
                    f"==> HD-EPIC using sampled global SDP for RGB: "
                    f"{len(self.global_sdp_points)}/{len(self.p3_array)} points"
                )

        valid_start = start_n
        valid_end = usable - 1
        if self.restrict_range:
            _, first_record = self.provider.get_image_data_by_index(self.stream_id, 0)
            _, last_record = self.provider.get_image_data_by_index(self.stream_id, usable - 1)
            image_start_ns = int(first_record.capture_timestamp_ns)
            image_end_ns = int(last_record.capture_timestamp_ns)
            modality_starts = [image_start_ns]
            modality_ends = [image_end_ns]
            if self.with_traj and len(self.pose_ts) > 0:
                modality_starts.append(int(self.pose_ts.min()))
                modality_ends.append(int(self.pose_ts.max()))
            if self.with_sdp and len(self.sdp_times) > 0:
                modality_starts.append(int(self.sdp_times.min()))
                modality_ends.append(int(self.sdp_times.max()))
            start_ns = max(modality_starts)
            end_ns = min(modality_ends)
            valid_start = max(valid_start, self._find_vrs_frame_by_timestamp(start_ns))
            valid_end = min(valid_end, self._find_vrs_frame_by_timestamp(end_ns))

        if valid_start > valid_end:
            raise ValueError(f"No overlapping HD-EPIC frames remain for {self.video_id}")

        self.sample_indices = list(range(valid_start, valid_end + 1, skip_n))
        if (
            self.with_sdp
            and self.prefer_sdp_aligned_frames
            and not self.use_global_sdp_for_rgb
            and len(self.sdp_times) > 0
            and len(self.sample_indices) > 0
        ):
            aligned_indices = []
            for frame_idx in self.sample_indices:
                ts_ns = self._frame_timestamp_ns(frame_idx)
                sdp_idx = find_nearest2(self.sdp_times, ts_ns)
                sdp_ns = int(self.sdp_times[sdp_idx])
                delta_s = abs(sdp_ns - ts_ns) / 1e9
                if delta_s <= self.sdp_frame_alignment_s:
                    aligned_indices.append(frame_idx)
            if aligned_indices:
                print(
                    f"==> HD-EPIC SDP alignment filter kept {len(aligned_indices)}/"
                    f"{len(self.sample_indices)} candidate frames "
                    f"(threshold={self.sdp_frame_alignment_s * 1000:.1f}ms)"
                )
                self.sample_indices = aligned_indices
            else:
                print(
                    "==> Warning: HD-EPIC SDP alignment filter would keep 0 frames; "
                    "falling back to unfiltered frame sampling"
                )
        self.sample_indices = self.sample_indices[:max_n]
        self.length = len(self.sample_indices)
        if self.length == 0:
            raise ValueError(f"HD-EPIC sampling produced 0 frames for {self.video_id}")
        self.index = 0

        print(f"==> Using HD-EPIC video '{self.video_id}' with {self.length} sampled frames")

        self._init_prefetch()

    def _find_vrs_frame_by_timestamp(self, target_ns: int) -> int:
        lo, hi = 0, self.provider.get_num_data(self.stream_id) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            _, record = self.provider.get_image_data_by_index(self.stream_id, mid)
            if int(record.capture_timestamp_ns) < target_ns:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _frame_timestamp_ns(self, frame_idx: int) -> int:
        _, record = self.provider.get_image_data_by_index(self.stream_id, frame_idx)
        return int(record.capture_timestamp_ns)

    def _read_vrs_rgb(self, frame_idx: int):
        data, record = self.provider.get_image_data_by_index(self.stream_id, frame_idx)
        if not data.is_valid():
            raise IOError(f"Invalid VRS image data at frame {frame_idx} in {self.vrs_path}")
        img = data.to_numpy_array()
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        return img, int(record.capture_timestamp_ns)

    def _build_camera(self, ts_ns: int, frame_w: int, frame_h: int):
        calib_idx = find_nearest2(self.calib_ts, ts_ns)
        cam_fish = self.calibs[calib_idx].float()
        cam = cam_fish.scale_to_size((frame_w, frame_h)).float()

        resize = self.resize
        if resize is not None:
            if isinstance(resize, tuple):
                resize_h, resize_w = resize
            else:
                resize_h = resize_w = resize
        else:
            resize_h, resize_w = frame_h, frame_w

        if not self.pinhole:
            if resize_h != frame_h or resize_w != frame_w:
                cam = cam.scale_to_size((resize_w, resize_h)).float()
            return cam, cam_fish, resize_h, resize_w, None

        cam_w = cam_fish.size[0].item()
        cam_h = cam_fish.size[1].item()
        w_ratio = resize_w / cam_w
        h_ratio = resize_h / cam_h
        cx = cam_fish.c[0] * w_ratio
        cy = cam_fish.c[1] * h_ratio
        fxy = self.pinhole_fxy if self.pinhole_fxy is not None else cam_fish.f[0] * 1.2
        fx = fxy * w_ratio
        fy = fxy * h_ratio
        intr_pin = [fx, fy, cx, cy]
        cam_pin = CameraTW.from_surreal(
            height=resize_h,
            width=resize_w,
            type_str="Pinhole",
            params=intr_pin,
            T_camera_rig=cam_fish.T_camera_rig,
        ).float()
        return cam_pin, cam_fish, resize_h, resize_w, cam_pin

    def load(self, idx):
        frame_idx = self.sample_indices[idx]
        img, ts_ns = self._read_vrs_rgb(frame_idx)
        rotated = torch.tensor([True])
        frame_h, frame_w = img.shape[:2]
        cam, cam_fish, resize_h, resize_w, pinhole_cam = self._build_camera(
            ts_ns, frame_w, frame_h
        )

        img_proc = img
        if resize_h != frame_h or resize_w != frame_w:
            img_proc = cv2.resize(img_proc, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        img_torch = torch.from_numpy(img_proc).permute(2, 0, 1)[None].float() / 255.0

        if self.pinhole:
            xx, yy = torch.meshgrid(
                torch.arange(resize_w), torch.arange(resize_h), indexing="ij"
            )
            xy = torch.stack([xx, yy], dim=-1).view(-1, 2).float()
            rays, valid = pinhole_cam.unproject(xy[None])
            xy_fish, valid2 = cam_fish.project(rays)
            xy_fish = xy_fish[0]
            valid = valid[0] & valid2[0]
            xy_fish[~valid] = -1
            xy_fish[:, 0] = (xy_fish[:, 0] / (frame_w - 1)) * 2 - 1
            xy_fish[:, 1] = (xy_fish[:, 1] / (frame_h - 1)) * 2 - 1
            uv = xy_fish.view(1, resize_w, resize_h, 2).permute(0, 2, 1, 3).float()
            src = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255.0
            img_torch = torch.nn.functional.grid_sample(
                src,
                uv,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )

        output = {
            "img0": img_torch.float(),
            "cam0": cam.float(),
            "rotated0": rotated,
            "time_ns0": ts_ns,
        }

        if self.with_traj:
            pose_idx = find_nearest2(self.pose_ts, ts_ns)
            pose_ns = self.pose_ts[pose_idx]
            delta_s = abs(int(pose_ns) - ts_ns) / 1e9
            if delta_s > self.max_time_diff_s:
                if self._skip_debug_count < 10:
                    print(
                        f"==> HD-EPIC skip (traj): frame_idx={frame_idx} ts_ns={ts_ns} "
                        f"pose_ns={int(pose_ns)} delta_ms={delta_s * 1000:.2f}"
                    )
                    self._skip_debug_count += 1
                return False
            output["T_world_rig0"] = self.traj[pose_idx].float()

        if self.with_sdp:
            if self.use_global_sdp_for_rgb:
                output["sdp_w"] = torch.from_numpy(self.global_sdp_points).float()
                return output
            sdp_idx = find_nearest2(self.sdp_times, ts_ns)
            sdp_ns = self.sdp_times[sdp_idx]
            delta_s = abs(int(sdp_ns) - ts_ns) / 1e9
            if delta_s > self.max_time_diff_s:
                if self._skip_debug_count < 10:
                    print(
                        f"==> HD-EPIC skip (sdp): frame_idx={frame_idx} ts_ns={ts_ns} "
                        f"sdp_ns={int(sdp_ns)} delta_ms={delta_s * 1000:.2f}"
                    )
                    self._skip_debug_count += 1
                return False
            uids = self.time_to_uids_combined[int(sdp_ns)]
            indices = [self.uid_to_idx[uid] for uid in uids]
            output["sdp_w"] = torch.from_numpy(self.p3_array[indices, :3]).float()

        return output
