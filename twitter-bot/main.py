import twitter
from vgg16 import model
from scipy import ndimage
from scipy import misc
import numpy as np

def crop_center(img,cropx,cropy):
	y,x,_ = img.shape
	startx = x//2-(cropx//2)
	starty = y//2-(cropy//2)    
	return img[starty:starty+cropy,startx:startx+cropx]

def resize(img):
	h,w,_ = img.shape
	if w > 224:
		if h > 224:
			result = crop_center(img, 224, 224)
		else:
			im = crop_center(img, 224, 224)
			d = im.shape[1] - 224
			p = d // 2
			result = np.pad(im, ((0,0),(p,d-p)), 'constant', constant_values=(0,0))
	else:
		if h > 224:
			im = crop_center(img, 224, 224)
			d = im.shape[0] - 224
			p = d // 2
			result = np.pad(im, ((p,d-p),(0,0)), 'constant', constant_values=(0,0))
		else:
			dh = 224 - img.shape[0]
			dw = 224 - img.shape[1]
			ph = dh // 2
			pw = dw // 2
			result = np.pad(img, ((ph,dh-ph,3),(pw,dw-pw,3)), 'constant', constant_values=(0,0))
	return result

with open('flowers.csv') as infile:
	rows = csv.DictReader(infile)
	classes = { row['wnid']: row['name'] for row in rows }

model.load_weights('models/finetune1.h5')

api = twitter.Api(consumer_key='', consumer_secret='', access_token_key='', access_token_secret='')

events = api.GetUserStream()

at_tweets = (e for e in events if 'event' not in e and e['in_reply_to_user_id'] and not e['in_reply_to_status_id'])
retweets = (e for e in events if 'event' not in e and e['retweeted_status'])
favorites = (e for e in events if 'event' in e and e['event'] == 'favorite')

for t in at_tweets:
	# Check if picture in tweet
	# Download picture
	# Read in picture
	# Resize picture
	probas = model.predict([img])
	idxs = np.argmax(probas, axis=1)
	idx = idxs[0]
	api.PostUpdate('Flower is %s. RT if correct, like otherwise' % classes[idx], in_reply_to_status_id=t['id'])

for rt in retweets:
	# Update model

for fav in favorites:
	# Update model



