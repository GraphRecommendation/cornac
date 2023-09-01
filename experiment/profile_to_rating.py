import pandas as pd
import os
import random


def run(dataset_name):
	# random.seed(42)
	# df columns: reviewerID,asin,overall,unixReviewTime,sentire_aspect,aspect_pos,aspect,opinion_pos,opinion,sentence,sentence_len,sentence_count,sentiment
	print(f'--- Loading {dataset_name} ---')
	df = pd.read_csv(os.path.join('seer-ijcai2020', dataset_name, 'profile.csv'), sep=',')
	df.drop(columns=['sentire_aspect', 'aspect_pos', 'opinion_pos', 'sentence_len', 'sentence_count'], inplace=True)

	# reviewerID,asin,overall,unixReviewTime,aspect_pos,aspect,opinion_pos,opinion,sentence,sentence_len,sentence_count,sentiment

	# Creating rating file
	ra_path = os.path.join('seer-ijcai2020', dataset_name, 'ratings.txt')
	if not os.path.isfile(ra_path):
		print('Removing duplicates')
		ratings = df[['reviewerID', 'asin', 'overall', 'unixReviewTime']].drop_duplicates()

		print('Saving ratings')
		ratings.to_csv(ra_path, index=False, header=False, sep=',')
		del ratings
	else:
		print('Warning: ratings.txt already exists, skipping.')

	df.drop(columns=['overall', 'unixReviewTime'], inplace=True)  # space efficiency

	# Creating review file
	re_path = os.path.join('seer-ijcai2020', dataset_name, 'review.txt')
	if not os.path.isfile(re_path):
		print('Getting sentences')
		review = df.groupby(['reviewerID', 'asin'], as_index=False).agg({'sentence': '.'.join})

		print('Joining sentences')
		review['sentence'] = review['sentence'].apply(lambda x: x.replace('\t', ' '))

		print('Saving reviews')
		review.to_csv(re_path, index=False, header=False, sep='\t')
		del review
	else:
		print('Warning: review.txt already exists, skipping.')

	df.drop(columns=['sentence'], inplace=True)  # space efficiency

	# Creating sentiment file
	s_path = os.path.join('seer-ijcai2020', dataset_name, 'sentiment.txt')
	if not os.path.isfile(s_path):
		print('Joining aspects, opinions and sentiments')
		df['aspect:opinion:sentiment'] = df.apply(lambda x: f"{x['aspect']}:{x['opinion']}:{x['sentiment']}", axis=1)
		df.drop(['aspect', 'opinion', 'sentiment'], axis=1, inplace=True)

		print('Assigning aspects, opinions and sentiments to reviews')
		aos = df.groupby(['reviewerID', 'asin'], as_index=False).agg({'aspect:opinion:sentiment': ','.join})

		print('Saving sentiment')
		with open(s_path, 'w') as f:
			for _, (r, a, t) in aos.iterrows():
				f.write(f'{r},{a},{t}\n')
	else:
		print('Warning: sentiment.txt already exists, skipping.')


if __name__ == '__main__':
	run('cellphone')
	run('toy')
	run('camera')
	run('computer')
	run('book')