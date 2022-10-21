import pandas as pd
import os
import random


def run(dataset_name):
	# random.seed(42)
	df = pd.read_csv(os.path.join('seer-ijcai2020', dataset_name, 'profile.csv'), sep=',')

	# reviewerID,asin,overall,unixReviewTime,aspect_pos,aspect,opinion_pos,opinion,sentence,sentence_len,sentence_count,sentiment

	ratings = df[['reviewerID', 'asin', 'overall', 'unixReviewTime']].drop_duplicates()

	review = df.groupby(['reviewerID', 'asin'], as_index=False).agg({'sentence': '.'.join})

	review['sentence'] = review['sentence'].apply(lambda x: x.replace('\t', ' '))

	df['aspect:opinion:sentiment'] = df.apply(lambda x: f"{x['aspect']}:{x['opinion']}:{x['sentiment']}", axis=1)

	aos = df.groupby(['reviewerID', 'asin'], as_index=False).agg({'aspect:opinion:sentiment': ','.join})

	ratings.to_csv(os.path.join('seer-ijcai2020', dataset_name, 'ratings.txt'), index=False, header=False, sep=',')
	review.to_csv(os.path.join('seer-ijcai2020', dataset_name, 'review.txt'), index=False, header=False, sep='\t')

	with open(os.path.join('seer-ijcai2020', dataset_name, 'sentiment.txt'), 'w') as f:
		for _, (r, a, t) in aos.iterrows():
			f.write(f'{r},{a},{t}\n')


if __name__ == '__main__':
	run('cellphone')