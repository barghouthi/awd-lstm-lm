import argparse
import data
import pprint
import torch
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='Sentence Probability Evaluation')

parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN)')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--inf', type=str, default='./eval.txt',
                    help='file to evaluate')
args = parser.parse_args()

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f, map_location=lambda storage, loc: storage)[0]

model.eval()
if args.model == 'QRNN':
    model.reset()

if args.cuda:
    model.cuda()
else:
    model.cpu()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)
hidden = model.init_hidden(1)
input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
if args.cuda:
    input.data = input.data.cuda()

all_words = []

# Tokenize file content
with open(args.inf, 'r') as f:
    # TODO - maybe we should count the words first so this can be dynamically allocated,
    # currently this limits the file length to 4096 words
    ids = torch.LongTensor(4096)
    token = 0
    for line in f:
        words = line.split() + ['<eos>']
        for word in words:
            ids[token] = corpus.dictionary.word2idx[word]
            token += 1
            all_words.append(word)

probs = []

for word_idx in ids[:len(all_words)]:
    input.data.fill_(word_idx)
    output, hidden = model(input, hidden)
    word_weights = model.decoder(output).squeeze().cpu()
    softmax_output_flat = torch.nn.functional.softmax(word_weights)
    probs.append(softmax_output_flat[word_idx])

pprint.pprint(list(zip(["%.4f" % float(p) for p in probs], all_words[1:])))
