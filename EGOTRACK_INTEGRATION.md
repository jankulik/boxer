# EgoTrack integration

The `egotrack-integration` branch builds on upstream Boxer commit
`d3647433dcb2f03abd35b1b89f5f332f6d0c93a7`. EgoTrack pins this branch's commit
through its `src/boxer` submodule; initializing that submodule supplies the
integration code without applying patches.

The integration adds:

- An explicit `dinov3_weights_path` argument to `BoxerNet.load_from_checkpoint`,
  so EgoTrack can keep DINOv3 weights in its configured checkpoint directory.
- Optional VRS, MPS, calibration, trajectory, semidense input, and cache paths
  in `AriaLoader`, plus public timestamp lookup and loading methods. Existing
  callers can continue using the original arguments and private lookup alias.
- An HD-EPIC loader that resolves video IDs, anonymized VRS recordings, and
  extracted multi-SLAM outputs. `run_boxer.py` accepts `--hd_epic_root`.
- An optional cache directory in `load_semidense`.

From the Boxer root, run the integration checks with the active Python
environment:

```bash
python -m unittest discover -s tests -p test_egotrack_integration.py -v
```

These checks cover timestamp selection, dataset path resolution, and local
DINOv3 weight selection without model downloads or a GPU. Full inference still
requires the original datasets, checkpoints, and runtime dependencies.

The upstream license and notices continue to apply; see `LICENSE` and `NOTICE`.
