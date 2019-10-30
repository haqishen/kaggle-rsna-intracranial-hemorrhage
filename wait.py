import sys
import time
from tqdm import tqdm

seconeds = int(sys.argv[1])
if seconeds > 0:
	print('Waiting...')
	for i in tqdm(range(seconeds)):
		time.sleep(1)
