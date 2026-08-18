# Dataset layout

Paths are relative to whatever is passed as `--root_dir` (the experiments use
`../datasets`). The directory names below are what the loaders in `data/` build
by hand, so a folder that is merely *named* differently will fail with
"does not exist yet" even when the data is there.

## Waterbirds (`-d CUB`)

```
<root>/<target_name>_<confounder_name>/metadata.csv
```

With `-t waterbird_complete95 -c forest2water2` that is
`waterbird_complete95_forest2water2/`. The distributed archive unpacks folders
named `waterbird_complete95.0_forest2water3`, so symlink them:

```bash
for pair in 95:95.0 875:87.5 75:75.0 625:62.5 50:50.0 100:100.0; do
  ln -sfn "waterbird_complete${pair##*:}_forest2water3" \
          "waterbird_complete${pair%%:*}_forest2water2"
done
```

**Each folder is an independent regeneration of the dataset**: the same image
filename gets a different background in `waterbird_complete50.0` than in
`waterbird_complete95.0`, so the `place` label — and therefore the group — is
folder-specific. Numbers from different folders are comparable in distribution
but are not the same sample. Test *membership* is stable across folders; only
train/val membership and the background assignment change.

`metadata_dfr.csv` (group-balanced subsample of the training split) has to live
*inside* each folder, since the loader joins it to the folder. The loose
`metadata_dfr.csv<token>` files at the dataset root are staging copies.

## CelebA (`-d CelebA`)

```
<root>/celeba/img_align_celeba/
<root>/celeba/list_attr_celeba.csv        (or metadata.csv)
<root>/celeba/list_eval_partition.csv
<root>/celeba/list_attr_celeba_dfr.csv    (--metadata_csv_name dfr)
<root>/celeba/list_eval_partition_dfr.csv
```

`--metadata_csv_name dfr` is a keyword, not a filename: `celebA_dataset.py`
rewrites it to the `_dfr` pair above, whose split 0 is a group-balanced
subsample of the training split (1387 per group).

## CivilComments (`-d jigsaw`)

```
<root>/jigsaw/data/all_data_with_identities.csv
```

Note `jigsaw/data/`, not `civilcomments/`. Groups come from
`-c identity_any -t toxicity`, so 2 identity values x 2 labels = 4 groups.

## MNIST-CIFAR (`-d MNISTCIFAR`)

```
<root>/MNISTCIFAR/MNIST_CIFAR_binary_<correlation>.pth
```

The correlation is passed as the confounder name, e.g. `-c 0.9`.

## MultiNLI (`-d MultiNLI`)

```
<root>/multinli/data/metadata.csv
<root>/multinli/glue_data/cached_train_bert-base-uncased_128_mnli
<root>/multinli/glue_data/cached_dev_bert-base-uncased_128_mnli
<root>/multinli/glue_data/cached_dev_bert-base-uncased_128_mnli-mm
```

The cached BERT features come from
`https://nlp.stanford.edu/data/dro/multinli_bert_features.tar.gz`. Note the
files sit directly in `glue_data/`, not in the `glue_data/MNLI/` subfolder the
group_DRO README describes.

The metadata is `dataset_metadata/multinli/metadata_random.csv` from
`github.com/kohpangwei/group_DRO`; install it as `metadata.csv` (the runs use
`-t gold_label_random`). It re-splits all 412349 examples into 206175 train /
82462 val / 123712 test, which is why the row count must match the three cached
feature files concatenated (392702 + 9815 + 9832 = 412349). If it does not, the
labels are silently misaligned with the features.
