model = model001
gpu = 2
fold = 0
conf = ./conf/${model}.py

train:
	@python -m src.cnn.main train ${conf} --fold ${fold} --gpu ${gpu}
