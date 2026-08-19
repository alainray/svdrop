# Per-dataset configuration for GDRO, shared by the GDRO-FT and end-to-end
# launchers. Every value is what the paper's runs used, read off the argument
# dumps in run_configs/. The epoch budget matches the training of the base model
# in each case.
#
# Source this and call: configure <dataset>
# It sets DATASET_ARG, TARGET, CONF, METADATA, MODEL, LR, WD_FT, WD_E2E, BS,
# EPOCHS_FT, EPOCHS_E2E, RESULTS, PRETRAIN.

configure() {
  case "$1" in
    CUB)
      DATASET_ARG=CUB;         TARGET=waterbird_complete95; CONF=forest2water2
      METADATA="metadata.csv"; MODEL=resnet50
      LR=1e-05;   WD_FT=1.0;   WD_E2E=1.0
      BS=64;      EPOCHS_FT=301;  EPOCHS_E2E=301
      RESULTS=CUB
      # Backbone retrained on the copy of Waterbirds we actually have; the
      # inherited one came from a different generation of the dataset.
      PRETRAIN="pretrained_models/CUB/erm_95_clean_SEED.pth"
      ;;
    CelebA)
      DATASET_ARG=CelebA;      TARGET=Blond_Hair;  CONF=Male
      METADATA="list_attr_celeba.csv"; MODEL=resnet50
      LR=1e-05;   WD_FT=0.1;   WD_E2E=0.1
      BS=64;      EPOCHS_FT=51;   EPOCHS_E2E=51
      RESULTS=CelebA
      PRETRAIN="pretrained_models/CelebA/erm_SEED.pth"
      ;;
    MNISTCIFAR)
      DATASET_ARG=MNISTCIFAR;  TARGET=CIFAR;  CONF=0.9
      METADATA="metadata.csv"; MODEL=scnn
      LR=0.001;   WD_FT=1e-04; WD_E2E=1e-04
      BS=10000;   EPOCHS_FT=5001; EPOCHS_E2E=5001
      RESULTS=MNISTCIFAR
      PRETRAIN="pretrained_models/MNISTCIFAR/erm_0.9_SEED.pth"
      ;;
    MultiNLI)
      DATASET_ARG=MultiNLI;    TARGET=gold_label_random; CONF=sentence2_has_negation
      METADATA="metadata.csv"; MODEL=bert-base-uncased
      LR=2e-05;   WD_FT=0.1;   WD_E2E=0.0
      BS=32;      EPOCHS_FT=5;    EPOCHS_E2E=6
      RESULTS=MultiNLI
      PRETRAIN="pretrained_models/MultiNLI/erm_0.9_SEED.pth"
      ;;
    civilcomments)
      DATASET_ARG=jigsaw;      TARGET=toxicity;  CONF=identity_any
      METADATA="all_data_with_identities.csv"; MODEL=bert-base-uncased
      LR=1e-05;   WD_FT=0.01;  WD_E2E=0.01
      BS=16;      EPOCHS_FT=5;    EPOCHS_E2E=5
      RESULTS=civilcomments
      PRETRAIN="pretrained_models/civilcomments/erm_0.9_SEED.pth"
      ;;
    *)
      echo "dataset desconocido: $1 (CUB, CelebA, MNISTCIFAR, MultiNLI, civilcomments)" >&2
      return 1 ;;
  esac
}
