# DFEW expression trajectory and frame-sampling study

This repository provides a local, shareable annotation platform for rating visible facial expressions in randomized single frames and continuous 16-frame sequences. The resulting ratings will later support a study of expression trajectories, frame sampling, and temporal learning.

The annotation platform is the current release focus. Analysis code is retained for the later study stage, but no analysis is required on an annotator's computer.

## Repository boundary

The public repository contains code, schemas, documentation, and synthetic faces only. DFEW material, task databases, ratings, exports, checkpoints, derived maps, and cached features are excluded by `.gitignore`.

The app can read the authorized official `clip_224x224_16f.zip` directly. Extracting thousands of small files is optional.

## Install

Use Python 3.11 or 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For feature caching and image/model analyses:

```bash
python -m pip install -r requirements-analysis.txt
```

Facial landmarks and pose are an optional extension because they are not required for the core registration and motion decomposition. Install them only on a machine where MediaPipe is supported:

```bash
python -m pip install -r requirements-landmarks.txt
```

On macOS, the annotator can double-click `scripts/start_mac.command`. On Windows, double-click `scripts\start_windows.bat`. Both launch the local browser app after the local configuration has been supplied.

The complete annotator handoff is described in `docs/ANNOTATOR_SETUP.md`.

## Try the synthetic demo

The demo contains only illustrated synthetic faces. On macOS, double-click `scripts/start_demo_mac.command`. On Windows, double-click `scripts\start_demo_windows.bat`. Sign in with `R01`, `R02`, or `R03`.

The demo shows both task types, keyboard controls, automatic draft saving, progress tracking, and resume behavior. It does not contain DFEW images, labels, model outputs, or real annotations.

## Configure authorized local data

Copy `config/project.example.toml` to `config/project.toml`. Supply local paths to the official encrypted 16-frame archive, full-length face-frame archives, fold-1 training and test CSV files, ten-rater annotation spreadsheet, and two fixed CNN checkpoints.

The local configuration is ignored by Git. The archive password is kept only there.

The repository and DFEW data can live side by side, but they do not need to be in the same directory. A convenient coordinator layout is:

```text
/your/workspace/
├── trajectory_sampling_study/       # GitHub clone
└── dfew_private/                    # local only; never commit this folder
    ├── clip_224x224_16f.zip         # official 16-frame archive
    ├── full_length_archives/        # optional original-length archives
    ├── train_set_1.csv
    ├── test_set_1.csv
    ├── annotation.xlsx              # the ten-rater DFEW spreadsheet
    └── checkpoints/
        ├── alexnet.pth
        └── resnet18.pth
```

You can instead point `config/project.toml` at the DFEW files wherever they already are; absolute paths are supported. The app accepts either the encrypted `clip_224x224_16f.zip` or an extracted directory with folders such as `00001/1.jpg` through `00001/16.jpg`. Do not copy DFEW images, passwords, private databases, or checkpoints into the Git clone.

### One-command setup

For the least manual work, put the private files under `local_data/dfew/` using the names in the tree above, then run:

```bash
python scripts/setup_local_config.py
```

The script checks the required files, asks for the archive password without echoing it, and writes the ignored `config/project.toml`. If your DFEW files already live elsewhere, keep them there and pass that folder instead:

```bash
python scripts/setup_local_config.py --dfew-root "/path/to/your/dfew_private"
```

The command also accepts `--archive`, `--annotation`, `--train-csv`, `--test-csv`, `--alexnet`, and `--resnet18` when those files are stored in different locations. After setup, the coordinator can initialize or open the study with:

```bash
python manage.py init-db
python manage.py prepare-study --config config/project.toml
python manage.py run --config config/project.toml
```

For annotation only, an annotator needs the Git clone, the private `study.sqlite` bundle supplied by the coordinator, and access to the authorized 16-frame archive. The CSV files, ten-rater spreadsheet, full-length archives, and checkpoints are coordinator-side inputs and are not required on an annotator's computer.

## Prepare stimulus variables and the master study

Recover original face-frame counts, compute DFEW vote clarity, and run the resumable CPU stimulus audit:

```bash
python -m analysis.full_length_index --config config/project.toml
python -m analysis.label_clarity --config config/project.toml
python -m analysis.stimulus_audit --config config/project.toml
python -m analysis.stimulus_position --frame-features artifacts/stimulus_audit/frame_features.csv.gz --transition-features artifacts/stimulus_audit/transition_features.csv.gz
python -m analysis.duplicate_audit --stimulus-features artifacts/stimulus_audit/stimulus_features.csv --config config/project.toml
```

`stimulus_audit` estimates image quality, repeated frames, face/crop stability, robust rigid motion, residual non-rigid optical flow, and photometric residual. Its SQLite cache permits resumption. `--with-landmarks` adds pose measures when the local MediaPipe runtime supports them; core motion decomposition requires neither MediaPipe nor a GPU.

After `stimulus_features.csv` exists, create a new master database. This makes the 2,396-clip training subset stratified by label clarity, original frame count, and motion.

```bash
python manage.py init-db
python manage.py prepare-study --config config/project.toml
python manage.py validate-study
```

The design contains all 37,456 test frames as randomized single-frame tasks, 10% independent repeats, 2% hidden within-rater repeats, two independent trajectories for all 2,341 test clips, two trajectories for 2,396 training clips, and rule-triggered third ratings.

## Calibration and annotation

A private calibration manifest contains only `clip_id`, `task_type`, and `frame_index`; no reference answer enters the task database or page. The selector uses four non-test clips per DFEW category, balancing clear and category-boundary examples and preferring clips outside the 2,396-clip human-rated training subset. Because that subset contains every available fold-1 training Disgust clip, the coordinator key explicitly flags the four reused calibration clips. The separate key is a discussion aid, not a gold-standard answer.

```bash
python manage.py select-calibration --config config/project.toml
python manage.py prepare-calibration --manifest local_data/calibration_manifest.csv
python manage.py make-annotator-bundle --annotator R01 --output local_data/bundles/R01
python manage.py make-annotator-bundle --annotator R02 --output local_data/bundles/R02
python manage.py make-annotator-bundle --annotator R03 --output local_data/bundles/R03
```

Each private bundle contains only one annotator’s assignments. The coordinator supplies the code folder, authorized archive, local configuration, and private database. Ratings auto-save and resume at the first unfinished task.

After the three exports are returned:

```bash
python manage.py merge exports/R01 exports/R02 exports/R03
python manage.py create-adjudications
```

Create and distribute fresh private bundles for pending third ratings, merge those exports, and lock the interfaces:

```bash
python manage.py export-consensus --output exports/consensus
python manage.py validate-study --require-complete
```

## Model and sampling analysis

Cache fixed CNN features and logits. These CPU commands resume from `progress.json`:

```bash
python -m analysis.cache_features --config config/project.toml --model alexnet --split train
python -m analysis.cache_features --config config/project.toml --model alexnet --split test
python -m analysis.cache_features --config config/project.toml --model resnet18 --split train
python -m analysis.cache_features --config config/project.toml --model resnet18 --split test
```

Then run the human-trajectory, sampling, motion, and spatial analyses:

```bash
python -m analysis.trajectory --ratings exports/consensus/trajectory_consensus.csv --label-clarity artifacts/label_clarity.csv
python -m analysis.annotation_agreement --frame-pairs exports/consensus/frame_reliability_pairs.csv --clip-ratings exports/consensus/clip_ratings.csv.gz
python -m analysis.human_label_comparison --test-csv local_data/dfew/test_set_1.csv --label-clarity artifacts/label_clarity.csv --frame-primary exports/consensus/frame_primary.csv --trajectory-consensus exports/consensus/trajectory_consensus.csv
python -m analysis.frame_subsets --frame-logits local_data/features/alexnet_test/frame_logits.csv.gz local_data/features/resnet18_test/frame_logits.csv.gz --training-frame-logits local_data/features/alexnet_train/frame_logits.csv.gz local_data/features/resnet18_train/frame_logits.csv.gz --trajectories artifacts/trajectory_consensus.csv --frame-ratings exports/consensus/frame_primary.csv
python -m analysis.subset_geometry --subset-results artifacts/frame_subsets/subset_results.csv.gz
python -m analysis.late_advantage --frame-logits local_data/features/alexnet_test/frame_logits.csv.gz local_data/features/resnet18_test/frame_logits.csv.gz --trajectories artifacts/trajectory_consensus.csv --stimulus-position artifacts/stimulus_position/stimulus_position_features.csv
python -m analysis.motion_evidence --frame-logits local_data/features/alexnet_test/frame_logits.csv.gz local_data/features/resnet18_test/frame_logits.csv.gz --transition-features artifacts/stimulus_audit/transition_features.csv.gz
python -m analysis.explain_sampling --outcomes artifacts/frame_subsets/sampling_outcomes.csv.gz --trajectory artifacts/trajectory_consensus.csv --label-clarity artifacts/label_clarity.csv --stimulus artifacts/stimulus_audit/stimulus_features.csv --stimulus-position artifacts/stimulus_position/stimulus_position_features.csv
python -m analysis.stimulus_structure --frame-features artifacts/stimulus_audit/frame_features.csv.gz --transition-features artifacts/stimulus_audit/transition_features.csv.gz --stimulus-features artifacts/stimulus_audit/stimulus_features.csv --trajectories artifacts/trajectory_consensus.csv --sampling-outcomes artifacts/frame_subsets/sampling_outcomes.csv.gz
python -m analysis.gradcam_motion --config config/project.toml --frame-logits local_data/features/alexnet_test/frame_logits.csv.gz local_data/features/resnet18_test/frame_logits.csv.gz --trajectories artifacts/trajectory_consensus.csv
```

Run the complete temporal training/evaluation matrix with one command:

```bash
python -m analysis.run_temporal_matrix --feature-root local_data/features --trajectories artifacts/trajectory_consensus.csv --selection artifacts/training_alignment_selection.csv
```

Training orders are natural, weak-to-strong, and strong-to-weak. Test orders are natural, reverse, weak-to-strong, strong-to-weak, and twenty saved permutations. Every test condition contains exactly the same frames. Average pooling verifies order invariance. The endpoint-only head measures how much of an apparent ordering gain can be obtained merely because sorting places a stronger or weaker frame last; GRU/TCN effects are interpreted relative to that control. The full-training reference is a natural-order unidirectional GRU by default and is reported separately from the matched 2,396-clip comparison.

After the matrix finishes, create paired natural–reverse, weak–strong, natural–shuffle, and matched-versus-opposite training/test direction contrasts. Human trajectory and original-length variables are merged so the effects are summarized by expression category, trajectory type, human direction, and source-video length.

```bash
python -m analysis.temporal_effects --predictions artifacts/temporal_models/*_test_predictions.csv.gz --trajectories artifacts/trajectory_consensus.csv --stimulus artifacts/stimulus_audit/stimulus_features.csv
python -m analysis.cross_temporal --backbone alexnet --feature-root local_data/features --output artifacts/cross_temporal_alexnet.csv
python -m analysis.cross_temporal --backbone resnet18 --feature-root local_data/features --output artifacts/cross_temporal_resnet18.csv
python -m analysis.validate_outputs --strict
```

## Safe synthetic trial

The demonstration contains no DFEW images:

```bash
python manage.py make-demo
python manage.py init-db --database local_data/demo.sqlite
python manage.py prepare-study --config demo_data/project.toml --database local_data/demo.sqlite --selection-output local_data/demo_training_selection.csv
python manage.py run --config demo_data/project.toml --database local_data/demo.sqlite
```

Demo access codes are `R01`, `R02`, and `R03`.

## Backups and exports

```bash
python manage.py backup
python manage.py status
python manage.py export --annotator R01 --output exports/R01
```

Every export first creates a timestamped SQLite backup. Re-importing the same export bundle is idempotent.

Before publishing the code repository, run:

```bash
python scripts/public_release_check.py
```

## Dataset citation

Jiang, X., Zong, Y., Zheng, W., Tang, C., Xia, W., Lu, C., and Liu, J. (2020). DFEW: A Large-Scale Database for Recognizing Dynamic Facial Expressions in the Wild. In *Proceedings of the 28th ACM International Conference on Multimedia*, 2881–2889.
