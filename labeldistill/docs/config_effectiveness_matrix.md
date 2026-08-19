# LabelDistill configuration effectiveness matrix

Every source-config leaf must have one owner.  `YAML` values are consumed by a
builder or runtime object, `preset` values are versioned Python implementation
details, and `derived` values are computed and read-only.

| Config path | Owner / consumer | Contract test |
| --- | --- | --- |
| `experiment.*` | audit, seeding, output identity | `test_config_system.py` |
| `runtime.*` | `trainer_builder.py`, optimizer LR derivation | `test_config_system.py`, `test_j4_builder.py` |
| `ema.*`, `checkpoint.*` | `trainer_builder.py` | trainer builder tests |
| `data.root`, `data.*_info`, `data.split` | `J4Experiment` dataloaders | builder tests |
| `data.num_workers`, `data.use_cbgs`, `data.key_idxes` | `J4Experiment`, J4 preset | builder/parity tests |
| `geometry.*` | J4 preset, mask builder, derived geometry | config/builder tests |
| `classes.names`, `classes.tasks` | student/teacher presets, matcher, scaler | config/builder tests |
| `student.type` | preflight registry validation | config tests |
| `student.output_channels`, `student.temporal_kd_selection` | J4 preset / `LabelDistill` | parity/builder tests |
| `teacher.type`, `teacher.checkpoint`, `teacher.checkpoint_prefix` | builder / `LabelDistill` | builder tests |
| `teacher.proposal.*` | teacher preset and independent decoder | parity/builder tests |
| `distillation.feature_channels` | `LabelDistill` adaptor contract | builder tests |
| `matching.*` | `ProposalTargetLayer` | config, ROI and builder tests |
| `region.scaler.*` | `AdaptiveGTScalerV3` | scaler/builder tests |
| `region.mask.*` | `QualityAwareMaskGeneratorV3` | mask/builder tests |
| `loss.*` | `J4Experiment.training_step` | experiment loss tests |
| `optimizer.*`, `scheduler.*` | `J4Experiment.configure_optimizers` | parity tests |
| `derived.*` | validation only; source YAML must not set it | config tests |

Python presets own tensor contracts, box-code layout, structured result types,
student/teacher low-level topology, and the mathematical implementations of
decode, matching, scaling and masking.  Preset and builder source hashes are
included in the audit manifest.
