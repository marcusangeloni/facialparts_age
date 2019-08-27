# Train CNN

```bash
TASK='age' \
FOLD='0' \
MODEL_DIR=/home/marcus/experiments/CNN/trained_models/${TASK}/Fold_${FOLD}/${TASK}_Fold_${FOLD}_`date +%d%m%Y_%H%M%S` \
CONFIG='/home/marcus/git/facialparts_age/config/conf1.json' \
TRAIN_META='/home/marcus/tf-records/'${TASK}'/train/fold_'${FOLD}'_task_'${TASK}'_phase_train*' \
EVAL_META='/home/marcus/tf-records/'${TASK}'/val/fold_'${FOLD}'_task_'${TASK}'_phase_val*' \
STEPS=40000 \
BATCH_SIZE=32 \
LR=1e-5 \
REG_VAL=1e-5 \
DROPOUT=0.4 \
BP=5 \
N_CLASSES=8

python tf_train_network.py \
--model_dir  ${MODEL_DIR} \
--resize_config_file ${CONFIG} \
--train_metadata ${TRAIN_META} \
--eval_metadata ${EVAL_META} \
--total_steps ${STEPS} \
--batch_size ${BATCH_SIZE} \
--learning_rate ${LR} \
--reg_val ${REG_VAL} \
--dropout ${DROPOUT} \
--batch_prefetch ${BP} \
--n_classes ${N_CLASSES}

FOLD='1' \
MODEL_DIR=/home/marcus/experiments/CNN/trained_models/${TASK}/Fold_${FOLD}/${TASK}_Fold_${FOLD}_`date +%d%m%Y_%H%M%S` \
TRAIN_META='/home/marcus/tf-records/'${TASK}'/train/fold_'${FOLD}'_task_'${TASK}'_phase_train*' \
EVAL_META='/home/marcus/tf-records/'${TASK}'/val/fold_'${FOLD}'_task_'${TASK}'_phase_val*'

python tf_train_network.py \
--model_dir  ${MODEL_DIR} \
--resize_config_file ${CONFIG} \
--train_metadata ${TRAIN_META} \
--eval_metadata ${EVAL_META} \
--total_steps ${STEPS} \
--batch_size ${BATCH_SIZE} \
--learning_rate ${LR} \
--reg_val ${REG_VAL} \
--dropout ${DROPOUT} \
--batch_prefetch ${BP} \
--n_classes ${N_CLASSES}

FOLD='2' \
MODEL_DIR=/home/marcus/experiments/CNN/trained_models/${TASK}/Fold_${FOLD}/${TASK}_Fold_${FOLD}_`date +%d%m%Y_%H%M%S` \
TRAIN_META='/home/marcus/tf-records/'${TASK}'/train/fold_'${FOLD}'_task_'${TASK}'_phase_train*' \
EVAL_META='/home/marcus/tf-records/'${TASK}'/val/fold_'${FOLD}'_task_'${TASK}'_phase_val*'

python tf_train_network.py \
--model_dir  ${MODEL_DIR} \
--resize_config_file ${CONFIG} \
--train_metadata ${TRAIN_META} \
--eval_metadata ${EVAL_META} \
--total_steps ${STEPS} \
--batch_size ${BATCH_SIZE} \
--learning_rate ${LR} \
--reg_val ${REG_VAL} \
--dropout ${DROPOUT} \
--batch_prefetch ${BP} \
--n_classes ${N_CLASSES}

FOLD='3' \
MODEL_DIR=/home/marcus/experiments/CNN/trained_models/${TASK}/Fold_${FOLD}/${TASK}_Fold_${FOLD}_`date +%d%m%Y_%H%M%S` \
TRAIN_META='/home/marcus/tf-records/'${TASK}'/train/fold_'${FOLD}'_task_'${TASK}'_phase_train*' \
EVAL_META='/home/marcus/tf-records/'${TASK}'/val/fold_'${FOLD}'_task_'${TASK}'_phase_val*'

python tf_train_network.py \
--model_dir  ${MODEL_DIR} \
--resize_config_file ${CONFIG} \
--train_metadata ${TRAIN_META} \
--eval_metadata ${EVAL_META} \
--total_steps ${STEPS} \
--batch_size ${BATCH_SIZE} \
--learning_rate ${LR} \
--reg_val ${REG_VAL} \
--dropout ${DROPOUT} \
--batch_prefetch ${BP} \
--n_classes ${N_CLASSES}

FOLD='4' \
MODEL_DIR=/home/marcus/experiments/CNN/trained_models/${TASK}/Fold_${FOLD}/${TASK}_Fold_${FOLD}_`date +%d%m%Y_%H%M%S` \
TRAIN_META='/home/marcus/tf-records/'${TASK}'/train/fold_'${FOLD}'_task_'${TASK}'_phase_train*' \
EVAL_META='/home/marcus/tf-records/'${TASK}'/val/fold_'${FOLD}'_task_'${TASK}'_phase_val*'

python tf_train_network.py \
--model_dir  ${MODEL_DIR} \
--resize_config_file ${CONFIG} \
--train_metadata ${TRAIN_META} \
--eval_metadata ${EVAL_META} \
--total_steps ${STEPS} \
--batch_size ${BATCH_SIZE} \
--learning_rate ${LR} \
--reg_val ${REG_VAL} \
--dropout ${DROPOUT} \
--batch_prefetch ${BP} \
--n_classes ${N_CLASSES}
```

# Test CNN

```bash
python tf_test_network.py '/home/marcus/adience-parts-iccvw' '/home/marcus/facialparts_age/folds/train_val_txt_files_per_fold/' '/home/marcus/experiments/CNN/trained_models/age/Fold_0/age_Fold_0_05022019_225503/1549416002/' 0

python tf_test_network.py '/home/marcus/adience-parts-iccvw' '/home/marcus/facialparts_age/folds/train_val_txt_files_per_fold/' '/home/marcus/experiments/CNN/trained_models/age/Fold_1/age_Fold_1_05022019_232002/1549417504/' 1

python tf_test_network.py '/home/marcus/adience-parts-iccvw' '/home/marcus/facialparts_age/folds/train_val_txt_files_per_fold/' '/home/marcus/experiments/CNN/trained_models/age/Fold_2/age_Fold_2_05022019_234505/1549419001/' 2

python tf_test_network.py '/home/marcus/adience-parts-iccvw' '/home/marcus/facialparts_age/folds/train_val_txt_files_per_fold/' '/home/marcus/experiments/CNN/trained_models/age/Fold_3/age_Fold_3_06022019_001002/1549420504/' 3

python tf_test_network.py '/home/marcus/adience-parts-iccvw' '/home/marcus/facialparts_age/folds/train_val_txt_files_per_fold/' '/home/marcus/experiments/CNN/trained_models/age/Fold_4/age_Fold_4_06022019_003505/1549422002/' 4
```
