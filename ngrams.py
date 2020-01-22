import math

STOP_WORD = "<STOP>"
UNKOWN_WORD = "<UNK>"

def readFile(filename):
	file = open(filename)
	data = [sentence[:len(sentence) - 1] + " " + STOP_WORD for sentence in file]
	file.close()
	return data

def tokenize(sentences):
	return [sentence.split(" ") for sentence in sentences]

def createTokenBank(token_sents):
	token_bank = {}
	token_count = 0
	for token_sent in token_sents:
		for token in token_sent:
			if token not in token_bank:
				token_bank[token] = 1
			else:
				token_bank[token] += 1
			token_count += 1;
	return (token_bank, token_count)

def replaceWithUNK1(token_sents, token_bank):
	for i in range(0, len(token_sents)):
		for j in range(0, len(token_sents[i])):
			if token_bank[token_sents[i][j]] < 3:
				token_sents[i][j] = UNKOWN_WORD
	return token_sents

def replaceWithUnk2(token_sents, train_data, unknown_word):
	for i in range(0, len(token_sents)):
		for j in range(0, len(token_sents[i])):
			if token_sents[i][j] not in train_data:
				token_sents[i][j] = unknown_word
	return token_sents

def createNgrams(token_sents_unk, n):
	n_grams_sents = []
	for token_sent_unk in token_sents_unk:
		n_grams_sent = []
		if len(token_sent_unk) < n:
			continue
		for i in range(0, len(token_sent_unk) - n + 1):
			n_grams = tuple(token_sent_unk[j] for j in range(i, i + n))
			n_grams_sent.append(n_grams)
		n_grams_sents.append(n_grams_sent)
	return n_grams_sents

"""
Inputs:
  token_bank: dictionary to hold the frequency of a certain token.
  cond_token_bank: dictionary to hold the frequency of the previous
  n - 1 words.
  n_grams: the data being calculated on.
  M: number of words in the testing data.
  N: number of words in the training data.
  n: gram.
"""
def computePerplexity(token_bank, cond_token_bank, n_grams, M, N, n):
	log_lik = 0
	for n_gram in n_grams:
		log_lik_s = 0
		for token in n_gram:
			"""
			If encountered the stop word, probability is 1.
			"""
			if token[n - 1] == STOP_WORD:
				log_lik_s += 0
				continue
			"""
			If it's unigram, denominator should be number of
			words in train data. Otherwise, its the frequency
			of the previous n-1 words.
			"""
			if n > 1:
				N = cond_token_bank[token[:n - 1]]
			pr_token = token_bank[token] / N
			log_lik_s += math.log(pr_token, 2)
		log_lik += log_lik_s
	l = log_lik / M
	perplexity = math.pow(2, -1 * l)
	return perplexity

def test(train_data, testing_data, word_count_test, word_count_train, n):
	result = []
	for i in range(n):
		perplexity = computePerplexity(train_data[i + 1], train_data[i], testing_data[i], word_count_test, word_count_train, i + 1)
		result.append(perplexity)
	return result

def main():
	"""
	Getting training, testing, and development result all requires reading data and
	preprocessing it. Preprocessing involves tokenizing the sentences and replacing
	some words with UNK's. More preprocessing will need to be done for each n-gram.
	"""

	"""
	Token banks used to store all the n-gram tokens from training data.
	"""
	n = 3
	token_banks = {0 : {}}

	"""
	There are 26602 unique words including STOP and UNK.
	There are 1622905 words in training data.
	"""
	sentences = readFile("A1-Data/1b_benchmark.train.tokens")
	token_sents = tokenize(sentenes)
	token_bank, token_count = createTokenBank(token_sents)
	token_sents_unk = replaceWithUNK1(token_sents, token_bank)

	"""
	Keep track of training data acnd number of words in training data
	for test and dev.
	"""
	train_data, __ = createTokenBank(token_sents_unk) # training data for word frequency
	train_data_words = token_count # word count for training data

	print("Words in training data: ", token_count)
	print("Unique words including <STOP> and <UNK>", len(train_data))

	"""
	Training result:
	Using the tokenized sentences that contains UNK's, generate n-grams for it
	and store it. Create a token bank from the n-gram, which are used as part
	of the training data for the test and dev.
	"""
	n_grams = []
	for i in range(n):
		n_gram = createNgrams(token_sents_unk, i + 1)
		token_banks[i + 1], __ = createTokenBank(n_gram)
		n_grams.append(n_gram)
	train_result = test(token_banks, n_grams, token_count, train_data_words, n)
	print("Train data:")
	for i in range(n):
		print(i + 1, "-gram: ", train_result[i])

	"""
	Testing result:
	"""
	sentences = readFile("A1-Data/1b_benchmark.test.tokens")
	token_sents = tokenize(sentences)
	token_bank, token_count = createTokenBank(token_sents)
	token_sents_unk = replaceWithUNK1(token_sents, token_bank)
	token_sents_unk = replaceWithUnk2(token_sents_unk, train_data, UNKOWN_WORD)

	n_grams = []
	for i in range(n):
		n_gram = createNgrams(token_sents_unk, i + 1)
		unknown_word = tuple(UNKOWN_WORD for j in range(i + 1))
		replaceWithUnk2(n_gram, token_banks[i + 1], unknown_word)
		n_grams.append(n_gram)
	test_result = test(token_banks, n_grams, token_count, train_data_words, n)
	print("Test data:")
	for i in range(n):
		print(i + 1, "-gram: ", test_result[i])

	"""
	Dev result
	"""
	sentences = readFile("A1-Data/1b_benchmark.dev.tokens")
	token_sents = tokenize(sentences)
	token_bank, token_count = createTokenBank(token_sents)
	token_sents_unk = replaceWithUNK1(token_sents, token_bank)
	token_sents_unk = replaceWithUnk2(token_sents_unk, train_data, UNKOWN_WORD)

	n_grams = []
	for i in range(n):
		n_gram = createNgrams(token_sents_unk, i + 1)
		unknown_word = tuple(UNKOWN_WORD for j in range(i + 1))
		replaceWithUnk2(n_gram, token_banks[i + 1], unknown_word)
		n_grams.append(n_gram)
	dev_result = test(token_banks, n_grams, token_count, train_data_words, n)
	print("Dev data:")
	for i in range(n):
		print(i + 1, "-gram: ", dev_result[i])

main()