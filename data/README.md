# Data

This directory is intentionally empty. The **CaddieSet** dataset is **not** redistributed in this repository; it
remains the property of its original authors.

## How to obtain CaddieSet

Please see the original paper and follow the authors' release instructions:

> Jung, S., Hong, S., Jeong, J., Jeong, S., Choi, J., Kim, H., & Lee, W. (2025).
> *CaddieSet: A Golf Swing Dataset with Human Joint Features and Ball Information.*
> arXiv:2508.20491. <https://arxiv.org/abs/2508.20491>

Once you have `CaddieSet.csv`, place it in this directory:

```
data/CaddieSet.csv
```

The pipeline scripts in `../src/` assume this exact filename.

## Schema (quick reference)

Every row is one swing, one camera view. The 80 columns fall into four groups:

| Group | Columns | Notes |
|---|---|---|
| Identifiers | `View`, `ClubType`, `GolferId` | `View ∈ {FACEON, DTL}` |
| Ball flight | `Distance`, `Carry`, `LrDistanceOut`, `DirectionAngle`, `SpinBack`, `SpinSide`, `SpinAxis`, `BallSpeed` | Launch-monitor outputs |
| Pose features | `0-...` through `7-...` (68 columns) | Numeric prefix = swing phase (0=Address, 7=Finish) |

Roughly half of the pose columns are populated only in one of the two camera views. Our pipeline handles this by
restricting each analysis to features with ≥90% non-missing rate in the relevant view.

See the paper (`../paper/paper.pdf`, §3) for full details and summary statistics.
