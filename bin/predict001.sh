model=model001
folder=seres50_2
gpu=0
fold=0
ep=2
tta=1
clip=1e-6
conf=./conf/${model}.py

snapshot=/data/src/kaggle-rsna-intracranial-hemorrhage/exp/${folder}/fold${fold}_ep${ep}.pt
valid=/data/src/kaggle-rsna-intracranial-hemorrhage/exp/${folder}/fold${fold}_ep${ep}_valid_tta${tta}.pkl
test=/data/src/kaggle-rsna-intracranial-hemorrhage/exp/${folder}/fold${fold}_ep${ep}_test_tta${tta}.pkl
sub=/data/src/kaggle-rsna-intracranial-hemorrhage/exp/${folder}/${model}_fold${fold}_ep${ep}_test_tta${tta}.csv

# python -m src.cnn.main test ${conf} --snapshot ${snapshot} --output ${test} --n-tta ${tta} --fold ${fold} --gpu ${gpu}
python -m src.postprocess.make_submission --input ${test} --output ${sub} --clip ${clip}
#kaggle competitions submit rsna-intracranial-hemorrhage-detection -m "" -f ./data/submission/${sub}

