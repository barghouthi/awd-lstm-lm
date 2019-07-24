###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch
from torch.autograd import Variable

import data

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN)')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model, _, _  = torch.load(f)

print ("Loaded Torch model from, ", f)
model.eval()

print ("Evaluated model")

if args.model == 'QRNN':
    model.reset()

if args.cuda:
    model.cuda()
else:
    model.cpu()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)

print ("Loading data, ", args.data)
print ("Number of tokens, ",  ntokens)

while True:
    uin = input("Enter priming text:\n")
    if uin == "":
        input_text = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
        hidden = model.init_hidden(1)
        all_text = ""
    else:
        all_text = uin.split()
        hidden = model.init_hidden(1)
        input_text = corpus.dictionary.word2idx[all_text[0]] 
        input_text = Variable(torch.tensor([[input_text]]).long(), volatile=True)
        print ("Processing ", all_text[0])
    
    if args.cuda:
        input_text.data = input_text.data.cuda()

    full_out = ""
    with open(args.outf, 'w') as outf:

        for i in range(1,len(all_text)):
            output, hidden = model(input_text, hidden)
            #word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
            word_weights = model.decoder(output).squeeze().data.div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            
            # reinitialize input with predicted word
            input_text.data.fill_(corpus.dictionary.word2idx[all_text[i]])

            print ("Processing ", all_text[i])

        for i in range(args.words):
            output, hidden = model(input_text, hidden)
            #word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
            word_weights = model.decoder(output).squeeze().data.div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            
            # reinitialize input with predicted word
            input_text.data.fill_(word_idx)
            
            word = corpus.dictionary.idx2word[word_idx]
            if word == "<eos>": 
                output = "\n"
            else: 
                output = word + (' ')
            
            outf.write(output)
            
            full_out = full_out + output
                        
            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))
        print ('-'*89)
        print (full_out)
        print ('-'*89)

