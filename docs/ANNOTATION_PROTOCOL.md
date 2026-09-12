# Annotation protocol

## Target of judgment

Rate only the face shown in the same 16 frames received by the models. Do not infer emotion from dialogue, scene context, identity, or an assumed neutral-to-expression progression.

## Scale anchors

- 0: no visible non-neutral expression, or the face cannot be evaluated.
- 1: a very weak cue is visible.
- 2: weak but repeatably visible.
- 3: moderate.
- 4: clearly expressed.
- 5: strong.
- 6: very strong and visually unambiguous.

`Mixed` means that more than one expression is visibly present. `Unclear` means that the face is visible but a category cannot be judged reliably. `Face not visible` is reserved for frames that cannot support a facial judgment.

Neutral and `Face not visible` use intensity 0. The category retains the distinction between a visible neutral face and a frame that cannot be judged. For `Mixed` or `Unclear`, intensity records the overall strength of the visible non-neutral facial change even when one expression category cannot be isolated.

## Single-frame block

Judge each frame without guessing what occurs elsewhere in its clip. The task order separates repeated views of the same clip by at least 50 intervening tasks. Complete this block before continuous-sequence ratings so sequence context does not enter isolated judgments.

## Continuous-sequence block

Watch the full loop once, choose the main visible expression, then trace the strength of that same expression at all 16 positions. For `Mixed` or `Unclear`, trace overall visible non-neutral expression strength. A valid curve may rise, fall, remain stable, contain multiple peaks, or begin strongly. Onset, apex, offset, peak count, and duration are computed later from the curve and are never requested as duplicate judgments.

## Calibration

All three annotators first rate the same private calibration set independently. The coordinator reviews category disagreements and scale use after the completed calibration set and clarifies the written anchors before the main tasks begin. Main-study clips are not discussed between annotators.

## Quality and scheduling

Use short sessions with breaks. The platform records active item time so the coordinator can adjust scheduling from observed pace. The scientific coverage remains fixed: one rating per isolated test frame with planned repeats, and two independent continuous curves for every test and training-alignment clip.
