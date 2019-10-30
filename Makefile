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

predict: wait
	@python -m src.cnn.main test ${conf} --snapshot ${snapshot} --output ${test} --n-tta ${tta} --fold ${fold} --gpu ${gpu}
	@python -m src.postprocess.make_submission --input ${test} --output ${sub} --clip ${clip}
# 	kaggle competitions submit rsna-intracranial-hemorrhage-detection -m "${message}" -f ${sub}
