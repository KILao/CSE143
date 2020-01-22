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

def replaceWithUnk(token_sents, token_bank):
	for i in range(0, len(token_sents)):
		for j in range(0, len(token_sents[i])):
			if token_bank[i] < 3:
				token_sents[i][j] = UNKOWN_WORD
	return token_sents
