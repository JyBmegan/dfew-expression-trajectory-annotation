# Data dictionary

## Annotation outputs

- `frame_ratings.csv.gz`: annotator, role, clip, frame, visible category, intensity, repeat type, completion time, and item duration.
- `clip_ratings.csv.gz`: annotator, role, clip, split, main visible category, 16 intensity values, sequence-property flags, completion time, and item duration.
- `frame_primary.csv`: exactly one prespecified primary isolated-frame rating per test clip and position. This is the single-frame analysis table; repeated ratings never replace or average into it.
- `frame_reliability_pairs.csv`: primary–repeat pairs for the 10% independent second ratings and 2% hidden within-rater repeats, with category agreement and intensity gaps.
- `trajectory_consensus.csv`: one consensus 16-point curve per clip plus uniformly derived onset, apex, offset, peak width, peak count, high-intensity proportion, and adjacent differences. Neutral has no onset/apex/offset values.
- `adjudications.csv`: trigger reasons, third annotator result, and final category and curve.

Item duration accumulates visible-page interaction time and caps any single inactive gap at 60 seconds, so a browser left open during a break does not dominate workload estimates.

## Stimulus outputs

- `full_length_index.csv`: original face-frame count and a nominal uniform-sampling interval. The latter is a scale descriptor, not a claim that the 16 supplied frames have been matched to exact source indices.
- `stimulus_features.csv`: clip-level original frame count, image quality, crop/face stability, duplicates, and motion summaries.
- `frame_features.csv.gz`: position-level brightness, contrast, sharpness, image hash, face location/scale, and optional pose.
- `transition_features.csv.gz`: 15 adjacent transitions per clip, including rigid translation/rotation/scale, residual non-rigid flow, and photometric residual.
- `label_clarity.csv`: highest DFEW vote, top-two vote margin, normalized vote entropy, Neutral votes, and all seven vote counts.
- `stimulus_position_features.csv`: within-clip early–late and last–first image-quality differences plus first/last transition deviations from the clip interior.
- `duplicate_candidates.csv`: cross-split sequence pairs, full-16-frame similarity tier, both DFEW labels, and whether those labels agree.

## Model outputs

- `model_outputs`: backbone, temporal head, training order, test order, clip, target, logits, prediction, and true-class logit margin.
- `subset_results.csv.gz`: all 1,820 fixed four-position combinations per backbone, full-sequence margin difference, prediction agreement, classification metrics, and human-trajectory coverage.
- `subset_group_results.csv.gz`: the same fixed combinations summarized separately by DFEW category, human start–end direction, and trajectory cluster.
- `train_selected_test_results.csv`: fixed four-position sets chosen using official training clips and then evaluated on official test clips, kept separate from same-test descriptive optima.
- `geometry_correlations.csv`, `geometry_group_summaries.csv`, and `top_fixed_subsets.csv`: deterministic descriptions of how temporal span, center, gaps, and boundary inclusion vary with four-frame outcomes.
- `sampling_outcomes.csv.gz`: clip-level outcomes for the predefined early, uniform, and late four-frame methods.
- `per_clip_oracle.csv`: explicitly marked descriptive upper bound that uses each clip’s full-sequence result.

Clip ID is the linking key across annotation, stimulus, and model tables. Frame positions are one-based from 1 to 16. Model class indices are zero-based from 0 to 6 and follow Happiness, Sadness, Neutral, Anger, Surprise, Disgust, and Fear.
