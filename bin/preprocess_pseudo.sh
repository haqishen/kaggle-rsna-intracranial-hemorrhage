mkdir -p cache model data/submission

# pseudo
python -m src.postprocess.make_pseudo --input /data/src/kaggle-rsna-intracranial-hemorrhage/exp/effnet_b3_512_p2/effnet_b3_512_p2_fold0_ep2_test_tta5.csv --output /data/src/kaggle-rsna-intracranial-hemorrhage/exp/effnet_b3_512_p2/pseudo.csv
python -m src.preprocess.dicom_to_dataframe --input /data/src/kaggle-rsna-intracranial-hemorrhage/exp/effnet_b3_512_p2/pseudo.csv --output /data/data/RSNA/pseudo_v1_raw.pkl --imgdir /data/data/RSNA/stage_1_test_images
python -m src.preprocess.create_dataset --input /data/data/RSNA/pseudo_v1_raw.pkl --output /data/data/RSNA/pseudo_v1.pkl

# Then image in test set should be copied to training set folder.
