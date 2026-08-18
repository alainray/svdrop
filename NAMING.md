# Naming scheme for experiments

The historical run folders (`erm_dfr_gdro_new_0.5_95`, `gdro_gdro_1_95`,
`erm_gdro_frz_rst2_875`, ...) mix five independent axes into one `_`-separated
string, use opaque version tags (`_new`, `_2`, `_1`, `_old`), and let sweep
values collide with correlation levels. They cannot be parsed, so tables end up
being assembled from hand-written lists of folder names.

New runs use the grammar below. Old runs keep their folder names; the mapping to
the canonical name lives in `run_configs/runs_index.csv`.

## Grammar

```
results/<dataset>/<exp>/model_outputs_<seed>/
```

`<exp>` is a sequence of fields joined by `__`. Each field is `<key>-<value>`,
split on the **first** hyphen so that values like `1e-5` stay intact.

The first field is the method, from a closed set:

| method | meaning |
|---|---|
| `e2e.erm` / `e2e.gdro` / `e2e.rw` / `e2e.jtt` | trained end to end |
| `ft.<features>.<head>` | last-layer retraining; `<features>` is the run that produced the frozen backbone, `<head>` the objective the new head is trained under. Both range over `erm`, `gdro`, `rw`. |

So `ft.erm.gdro` is GDRO-FT, `ft.gdro.gdro` is a GDRO head on GDRO features.

Remaining keys, in this order (omit any that does not apply):

| key | values | meaning |
|---|---|---|
| `c` | `95`, `875`, `75`, `625`, `50` (Waterbirds), `0.0`–`1.0` (MNIST-CIFAR), `std` | correlation level of the **training data actually loaded** |
| `src` | `train`, `val` | which split feeds the head — the DFR axis |
| `bal` | `none`, `rw`, `sub`, `sub.rw` | group balancing: none / `WeightedRandomSampler` / group-balanced subsampled CSV / both |
| `frac` | `1.0`, `0.5`, ... | fraction of that split retained by `--fraction` |
| `wd` | `1.0`, `1e-4`, ... | weight decay on the trainable parameters (λ) |
| `lr` | `1e-5`, ... | learning rate |
| `ep` | `301`, ... | number of epochs |
| `bn` | `train`, `eval` | mode of the frozen part of the network (see below) |
| `unfreeze`, `restart` | integers | only when non-zero |
| `feat` | `oldgen`, ... | only when the frozen backbone comes from a different instantiation of the dataset than the head is trained on |

Examples:

```
ft.erm.gdro__c-95__src-train__bal-rw__frac-1.0__wd-1.0__lr-1e-5__ep-301__bn-eval
ft.erm.erm__c-95__src-val__bal-sub__frac-1.0__wd-1e-4__lr-1e-4__ep-301__bn-eval
e2e.gdro__c-95__bal-rw__frac-1.0__wd-1.0__lr-1e-5__ep-301
```

## Why `bn` is an axis

`run_epoch` calls `model.train()` on every training epoch, including runs where
every backbone parameter has `requires_grad=False`. The frozen part then keeps
behaving as if it were training:

- **ResNet** (Waterbirds, CelebA): BatchNorm normalises with batch statistics and
  keeps updating `running_mean`/`running_var`. Because `--reweight_groups`
  changes what a batch contains, they drift *towards the balanced distribution*
  and the inherited head stops matching the features it was fitted on.
- **BERT** (MultiNLI, CivilComments): the 38 dropout layers stay on, so the
  "frozen" representation of an example is different every time it is seen. No
  state is corrupted, but the features are not fixed.
- **SimpleCNN** (MNIST-CIFAR): no BatchNorm and no dropout, so the axis is moot
  and those runs are unaffected.

By default the frozen part now runs in eval mode; `--train_frozen` restores the
old behaviour. Every run made before that flag existed is `bn-train`.

## Reporting

`run_configs/runs_index.csv` (regenerate with `python3 run_configs/build_index.py`)
has one row per run and one column per axis. Group by columns, not by folder
name.
