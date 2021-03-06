import torch
from torch import nn, optim
import argparse
import numpy as np
from model import Model
from dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


### function: train
def train(dataset, model, args):
    model.train()

    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in tqdm(range(args.max_epochs), total=args.max_epochs):
        state_h, state_c = model.init_state(args.sequence_length)

        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        # for batch, (x, y) in enumerate(dataloader):
        for batch, (x, y) in pbar:
            optimizer.zero_grad()

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()

            print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })


### function: predict
def predict(dataset, model, text, next_words=100):
    model.eval()

    words = text.split(' ')
    state_h, state_c = model.init_state(len(words))

    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])

    return words



def main():
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    print(device)

    ## Set up args
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--sequence-length', type=int, default=4)
    args = parser.parse_args()
    print(type(args))


    ## Set up dataset
    dataset = Dataset(args)

    ## Set up model
    model = Model(dataset)
    # model = model.to(device)

    ## Train model
    train(dataset, model, args)

    ## Generate text
    prediction = " ".join(predict(dataset, model, text='his feet are dry'))
    print(prediction)

if __name__ == "__main__":
    main()