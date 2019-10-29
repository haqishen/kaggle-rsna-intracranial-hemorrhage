mkdir -p cache model data/submission

# train
python -m src.preprocess.dicom_to_dataframe --input /data/data/RSNA/stage_1_train.csv --output /data/data/RSNA/train_raw.pkl --imgdir /data/data/RSNA/stage_1_train_images
python -m src.preprocess.create_dataset --input /data/data/RSNA/train_raw.pkl --output /data/data/RSNA/train.pkl
python -m src.preprocess.make_folds --input /data/data/RSNA/train.pkl --output /data/data/RSNA/train_folds.pkl --n-fold 5 --seed 42

# test
python -m src.preprocess.dicom_to_dataframe --input /data/data/RSNA/stage_1_sample_submission.csv --output /data/data/RSNA/test_raw.pkl --imgdir /data/data/RSNA/stage_1_test_images
python -m src.preprocess.create_dataset --input /data/data/RSNA/test_raw.pkl --output /data/data/RSNA/test.pkl
