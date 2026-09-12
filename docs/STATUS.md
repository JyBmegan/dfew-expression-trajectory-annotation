# Implementation status and resolved issues

## Preserved work

The existing `analysis_and_writing` directory, manuscript files, Figure 1, and earlier analyses have not been changed. This project is isolated in `trajectory_sampling_study`.

## Completed implementation

- Local Flask and SQLite annotation platform with auto-save, resumption, keyboard controls, progress, backup, export, merge, private per-rater bundles, calibration tasks, and adjudication.
- Full assignment generator for 2,341 test clips and the 2,396-clip training alignment subset.
- Direct encrypted-archive frame access without extracting DFEW into the public project.
- Complete original-frame-count index across all 11 full-length archive parts.
- DFEW ten-rater label-clarity table.
- Complete CPU stimulus audit for 9,356 fold-1 training clips and 2,341 fold-1 test clips: 187,152 frame records and 175,455 adjacent-transition records.
- Final 2,396-clip training alignment selection with 380 clips in each non-Disgust class and all 116 Disgust clips.
- Formal master database with 46,688 main tasks and 51,425 main assignments, plus a 56-task-per-rater calibration block.
- Three portable private annotator databases containing only the assigned rater's queue.
- Resumable fixed-CNN feature/logit cache tested with both project checkpoints.
- Human-trajectory descriptors and clustering, late-advantage decomposition, all 1,820 four-frame combinations, stimulus-block explanation, temporal heads, cross-temporal generalization, motion-evidence analysis, and change-targeted Grad-CAM/deformation overlap.
- Synthetic-data and real-checkpoint smoke tests plus database design tests.

## Problems found and corrections

1. The existing extracted 16-frame directory contained only the 2,341 fold-1 test clips. The official encrypted 16-frame archive contains the complete input set. The platform and analyses now read that archive directly, preserving the official frame definition and avoiding slow creation of hundreds of thousands of small files on an external drive.
2. One locally extracted full-length part was incomplete. The indexer validates direct archives and transparently uses the intact nested source for missing or invalid parts.
3. The installed MediaPipe build attempted to create an OpenGL service even in its CPU graph. Core motion decomposition now uses CPU OpenCV feature registration and residual dense flow. MediaPipe landmarks are optional and cannot stop the analysis.
4. The initial dry-run training subset was stratified only by label clarity because stimulus features did not yet exist. It was retained only as a recoverable pre-balanced copy. The completed stimulus audit was then used to regenerate the formal master database, balancing the 2,396-clip selection by label clarity, original length, and registered motion within category.
5. Earlier consensus export names mixed raw ratings and consensus tables. Exports now preserve annotator-level records separately and produce one explicit `trajectory_consensus` table.
6. The cross-split perceptual-hash screen found 18 candidate pairs. Full 16-frame verification places 5 in the very-high-similarity tier and 6 in the high-similarity tier. Four of the five very-high pairs share the same DFEW label, while one very-high pair and two high-similarity pairs have different labels. Labels and similarity measures are stored beside every candidate. The eleven corresponding training clips are excluded before drawing the small human-rated training-alignment subset; test clips remain in the main test analysis and can be flagged for a small sensitivity analysis.
7. The local OpenCV build did not contain its Haar detector data, and the installed MediaPipe graph could not initialize its macOS display service. Frame quality, rigid registration, residual non-rigid flow, crop drift, duplicates, and photometric residual were still computed for every clip. Face-landmark pose fields remain optional; human ratings separately capture occlusion, crop/shot changes, speaking, and subject switches.
8. An initial AlexNet cache used the output after the second fully connected layer's ReLU, although the original model's `return_features=True` interface and earlier temporal scripts use that layer's linear output before ReLU. The incomplete test cache was stopped and both AlexNet splits are being regenerated through the model's public feature-return interface. Cache progress now records this feature definition, and final validation rejects caches made with another definition.
9. The first consensus exporter averaged the repeated isolated-frame judgments into the main per-frame table. That gave repeated frames a different measurement rule from singly rated frames. The main export now always contains the one prespecified primary rating for every frame; second-rater and hidden-repeat observations are exported only as paired reliability records.

## Data-dependent stages

Human-result analyses require completed ratings and do not fabricate placeholders. The fixed-CNN feature caches are being generated sequentially on CPU and resume from durable checkpoints. Human annotation has not started.
