# Create TFRecords with train and val sets

```bash
python generate_tfrecords.py \
--protocol-file ../../folds/train_val_txt_files_per_fold/test_fold_is_0/age_train.txt \
--image-dir /home/marcus/adience-parts-iccvw \
--output-path /home/marcus/tf-records \
--resize-config-file ../../config/conf1.json \
--negative-list /home/marcus/adience_negative_list.txt

python generate_tfrecords.py \
--protocol-file ../../folds/train_val_txt_files_per_fold/test_fold_is_1/age_train.txt \
--image-dir /home/marcus/adience-parts-iccvw \
--output-path /home/marcus/tf-records \
--resize-config-file ../../config/conf1.json \
--negative-list /home/marcus/adience_negative_list.txt

python generate_tfrecords.py \
--protocol-file ../../folds/train_val_txt_files_per_fold/test_fold_is_2/age_train.txt \
--image-dir /home/marcus/adience-parts-iccvw \
--output-path /home/marcus/tf-records \
--resize-config-file ../../config/conf1.json \
--negative-list /home/marcus/adience_negative_list.txt

python generate_tfrecords.py \
--protocol-file ../../folds/train_val_txt_files_per_fold/test_fold_is_3/age_train.txt \
--image-dir /home/marcus/adience-parts-iccvw \
--output-path /home/marcus/tf-records \
--resize-config-file ../../config/conf1.json \
--negative-list /home/marcus/adience_negative_list.txt

python generate_tfrecords.py \
--protocol-file ../../folds/train_val_txt_files_per_fold/test_fold_is_4/age_train.txt \
--image-dir /home/marcus/adience-parts-iccvw \
--output-path /home/marcus/tf-records \
--resize-config-file ../../config/conf1.json \
--negative-list /home/marcus/adience_negative_list.txt
```

# Sanity test

```bash
python tfrecords_sanity_test.py --tfrecord-file ../../../tf-records/age/train/fold_0_task_age_phase_train-00000-of-00001.tfrecord.gz --output-dir ./

python tfrecords_sanity_test.py --tfrecord-file ../../../tf-records/age/train/fold_1_task_age_phase_train-00000-of-00001.tfrecord.gz --output-dir ./

python tfrecords_sanity_test.py --tfrecord-file ../../../tf-records/age/train/fold_2_task_age_phase_train-00000-of-00001.tfrecord.gz --output-dir ./

python tfrecords_sanity_test.py --tfrecord-file ../../../tf-records/age/train/fold_3_task_age_phase_train-00000-of-00001.tfrecord.gz --output-dir ./

python tfrecords_sanity_test.py --tfrecord-file ../../../tf-records/age/train/fold_4_task_age_phase_train-00000-of-00001.tfrecord.gz --output-dir ./

python tfrecords_sanity_test.py --tfrecord-file ../../../tf-records/age/val/fold_0_task_age_phase_val-00000-of-00001.tfrecord.gz --output-dir ./

python tfrecords_sanity_test.py --tfrecord-file ../../../tf-records/age/val/fold_1_task_age_phase_val-00000-of-00001.tfrecord.gz --output-dir ./

python tfrecords_sanity_test.py --tfrecord-file ../../../tf-records/age/val/fold_2_task_age_phase_val-00000-of-00001.tfrecord.gz --output-dir ./

python tfrecords_sanity_test.py --tfrecord-file ../../../tf-records/age/val/fold_3_task_age_phase_val-00000-of-00001.tfrecord.gz --output-dir ./

python tfrecords_sanity_test.py --tfrecord-file ../../../tf-records/age/val/fold_4_task_age_phase_val-00000-of-00001.tfrecord.gz --output-dir ./
```