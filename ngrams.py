import math

STOP_WORD = "<STOP>"
UNKOWN_WORD = "<UNK>"

def readFile(filename):
	file = open(filename)
	data = [sentence[:len(sentence) - 1] + " " + STOP_WORD for sentence in file]
	file.close()
	return data

def tokenize(sentences):
	return [sentences.split(" ") for sentence in sentences]

def createTokenBank(token_sents):
	token_bank = {}
	token_count
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
			if token_bank[i] < 3:
				token_sents[i][j] = UNKOWN_WORD
	return token_sents

def replaceWithUnk2(token_sents, train_data):
	for i in range(0, len(token_sents)):
		for j in range(0, len(token_sents[i])):
			if token_bank[i] not in train_data:
				token_sents[i][j] = UNKOWN_WORD
	return token_sents

def createNgrams(token_sents_unk, n):
	n_grams_sents = []
	for token_sent_unk in token_sents_unk:
		n_gram_sent = []
		if len(token_sent) < n:
			continue
		for i in range(0, len(token_sent) - n + 1):
			n_gram = tuple(token_sent_unk[j] for j in range(i, i + n))
			n_grams_sent.append(n_gram)
		n_grams_sents.append(n_grams_sent)
	return n_grams_sents

def computePerplexity(token_bank, cond_token_bank, n_grams, num_instances, word_count, n):
	log_lik = 0
	for n_gram in n_grams:
		log_lik_s = 0
		for token in n_gram:
			if token[n - 1] == STOP_WORD:
				log_lik_s += 0
				continue
			if n > 1:
				token_count = cond_token_bank[token[:n - 1]]
			log_lik_s += math.log(token_bank[token] / token_count, 2)
		log_lik += log_lik_s
	l = log_lik / num_instances
	perplexity = math.pow(2, -1 * l)
	return perplexity





















