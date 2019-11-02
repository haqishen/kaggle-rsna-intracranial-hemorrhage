model = seres50_mine
gpu = 2
fold = 0
conf = ./conf/${model}.py

ep = 2
folder = ${model}
tta = 5
clip = 1e-6
snapshot = /data/src/kaggle-rsna-intracranial-hemorrhage/exp/${folder}/fold${fold}_ep${ep}.pt
valid = /data/src/kaggle-rsna-intracranial-hemorrhage/exp/${folder}/fold${fold}_ep${ep}_valid_tta${tta}.pkl
test = /data/src/kaggle-rsna-intracranial-hemorrhage/exp/${folder}/fold${fold}_ep${ep}_test_tta${tta}.pkl
sub = /data/src/kaggle-rsna-intracranial-hemorrhage/exp/${folder}/${model}_fold${fold}_ep${ep}_test_tta${tta}.csv
message = ${sub}

waits = 0

wait:
	@python wait.py ${waits}

train: wait
	@python -m src.cnn.main train ${conf} --fold ${fold} --gpu ${gpu}

ptrain: wait
	@python -m src.cnn.main ptrain ${conf} --fold ${fold} --gpu ${gpu}

train14: wait
	@python -m src.cnn.main train ${conf} --fold 1 --gpu ${gpu}
	@python -m src.cnn.main train ${conf} --fold 2 --gpu ${gpu}
	@python -m src.cnn.main train ${conf} --fold 3 --gpu ${gpu}
	@python -m src.cnn.main train ${conf} --fold 4 --gpu ${gpu}

train04: wait
	@python -m src.cnn.main train ${conf} --fold 0 --gpu ${gpu}
	@python -m src.cnn.main train ${conf} --fold 1 --gpu ${gpu}
	@python -m src.cnn.main train ${conf} --fold 2 --gpu ${gpu}
	@python -m src.cnn.main train ${conf} --fold 3 --gpu ${gpu}
	@python -m src.cnn.main train ${conf} --fold 4 --gpu ${gpu}

ptrain04: wait
	@python -m src.cnn.main ptrain ${conf} --fold 0 --gpu ${gpu}
	@python -m src.cnn.main ptrain ${conf} --fold 1 --gpu ${gpu}
	@python -m src.cnn.main ptrain ${conf} --fold 2 --gpu ${gpu}
	@python -m src.cnn.main ptrain ${conf} --fold 3 --gpu ${gpu}
	@python -m src.cnn.main ptrain ${conf} --fold 4 --gpu ${gpu}

valid: wait
	@python -m src.cnn.main valid ${conf} --snapshot ${snapshot} --output ${valid} --n-tta 1 --fold ${fold} --gpu ${gpu}

predict: wait
	@python -m src.cnn.main test ${conf} --snapshot ${snapshot} --output ${test} --n-tta ${tta} --fold ${fold} --gpu ${gpu}

makecsv:
	@python -m src.postprocess.make_submission --inputs exp/${folder} --output exp/${folder}/${model}_5fold_5tta.csv
# 	@python -m src.postprocess.make_submission --input ${test} --output ${sub} --clip ${clip}
# 	kaggle competitions submit rsna-intracranial-hemorrhage-detection -m "${message}" -f ${sub}

# make model=effnet_b3_512_p3 gpu=1 fold=0 predict && make model=effnet_b3_512_p3 gpu=1 fold=1 predict && make model=effnet_b3_512_p3 gpu=0 fold=2 predict && make model=effnet_b3_512_p3 gpu=0 fold=3 predict && make model=effnet_b3_512_p3 gpu=0 fold=4 predict
# make model=effnet_b3_512_p3 gpu=1 fold=4 predict && make model=effnet_b3_512_p3 gpu=0 fold=3 predict 
