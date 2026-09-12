# Analysis map

This document keeps each computation tied to the study question. It is an execution map, not a manuscript outline.

| Question | Human or stimulus evidence | Model evidence | Primary comparison | Output |
|---|---|---|---|---|
| Where is the main visible expression? | Two independent 16-point intensity curves; adjudicated when prespecified disagreement rules are met | Per-frame true-class logit margin | Human trajectory position versus model evidence position | `trajectory_consensus.csv`, trajectory clusters, position summaries |
| Are the human measurements reproducible enough for trajectory analysis? | Prespecified independent and hidden frame repeats; paired continuous-sequence ratings | None | Category agreement and intensity-curve gaps, kept separate from the primary frame values | `annotation_agreement` summaries and pair tables |
| What produces the average late-frame advantage? | Start–end direction, peak position/width, high-intensity coverage, DFEW vote clarity | Early–late paired margin differences for budgets 1 and 4 | Within each trajectory type, then weighted decomposition across types | `late_advantage` tables |
| Why does a four-frame sample succeed or fail? | Peak/high-intensity/expression-presence coverage, isolated-frame agreement | All 1,820 four-frame combinations; fixed and clip-specific summaries kept distinct | Error from the full 16-frame decision and prediction agreement | `subset_results` and grouped-fold explanation tables |
| Which temporal layouts make a fixed four-frame sample effective? | Human coverage is analyzed separately | Span, center, gaps, boundary inclusion, and temporal-quartile coverage for all 1,820 combinations | Deterministic subset ranks and geometry summaries, without treating the 1,820 overlapping subsets as independent observations | `subset_geometry` tables |
| Is apparent motion useful evidence? | Rigid movement, residual non-rigid deformation, photometric change, boundary flags | Adjacent absolute margin change and prediction switching | Grouped-fold incremental prediction, by component | `motion_evidence` tables |
| Could early–late image quality or clip boundaries mimic a position effect? | Within-clip early–late brightness, contrast, and sharpness differences; first/last transition deviations from the interior | These directional variables enter the sampling explanation separately from expression trajectories | Paired within-clip summaries and grouped-fold block removal | `stimulus_position` tables |
| Does expression-order alignment help temporal heads? | Training and test curves define weak-to-strong and strong-to-weak orders | Average pooling, endpoint-only control, uni-GRU, bi-GRU, and causal TCN on the same cached frames | Paired natural–reverse, weak–strong, natural–shuffle, and matched-versus-opposite training/test direction contrasts; recurrent/convolutional effects are compared with the endpoint-only control | `temporal_models` predictions and `temporal_effects` summaries |
| Does spatial attribution follow expression-related deformation? | Residual non-rigid deformation maps on evidence-changing transitions | True-class Grad-CAM change between the same adjacent frames | Spatial rank correlation and top-area overlap, balanced by class and trajectory | `gradcam_motion` maps and tables |
| Do representation positions generalize across time? | Human trajectory and stimulus variables provide the main account | Train-position by test-position classification matrix | Additional grouped-fold explanation beyond human/stimulus variables | `cross_temporal` matrices; promoted only if incrementally informative |

## Interpretation limits built into the computation

- The human curve describes the main visible expression in the supplied 16-frame sequence. It does not assume a neutral-to-expression direction.
- Neutral sequences are retained as a low-expression reference and receive no forced onset or apex.
- Rearranged temporal tests keep all 16 frames; only their order changes.
- The clip-specific best four-frame set is descriptive. Deployable sampling is evaluated with one fixed rule or a grouped-fold predictor that has no access to the held-out clip.
- Rigid movement, residual deformation, and photometric residual are separate variables. Total optical flow is not treated as facial-muscle movement.
