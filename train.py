import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

from model import LSTMSpeakerEncoder
from feature_extraction import NUM_MFCC
import data
from loss import TripletCosineLoss

# Model hyperparameters
HIDDEN_SIZE = 20
NUM_LAYERS = 1
LEARNING_RATE = 0.003

def train_model(model, train_loader, optimizer, cost_function, device, data_store):
    losses = []
    model.train()

    for step, (anchor_batch, pos_batch, neg_batch) in enumerate(train_loader):
        loss = 0
        
        # Iterate through each example in the batch
        for i in range(len(anchor_batch)):
            # Get input
            anchor = data_store[anchor_batch[i]][0]
            pos = data_store[pos_batch[i]][0]
            neg = data_store[neg_batch[i]][0]

            # Convert to tensor
            anchor = torch.tensor(anchor, dtype=torch.float32).unsqueeze(0)
            pos = torch.tensor(pos, dtype=torch.float32).unsqueeze(0)
            neg = torch.tensor(neg, dtype=torch.float32).unsqueeze(0)

            # Get embedding
            anchor_embedding = model(anchor.to(device))
            pos_embedding = model(pos.to(device))
            neg_embedding = model(neg.to(device))

            # Get loss for this example
            loss += cost_function(anchor_embedding, pos_embedding, neg_embedding)

        # Vectorized (efficient version)

        # anchors = [data_store[i][0] for i in anchor_batch]
        # anchors = np.concatenate(anchors, axis=0)
        # anchors = torch.tensor(anchors, dtype=torch.float32).unsqueeze(0)

        # positives = [data_store[i][0] for i in pos_batch]
        # positives = np.concatenate(positives, axis=0)
        # positives = torch.tensor(positives, dtype=torch.float32).unsqueeze(0)

        # negatives = [data_store[i][0] for i in neg_batch]
        # negatives = np.concatenate(negatives, axis=0)
        # negatives = torch.tensor(negatives, dtype=torch.float32).unsqueeze(0)

        # anchor_embedding = model(anchors.to(device))
        # positive_embedding = model(positives.to(device))
        # negative_embedding = model(negatives.to(device))
            
        # loss = cost_function(anchor_embedding, positive_embedding, negative_embedding)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Neural Network Training Script')
    parser.add_argument('--epochs', type=int, default=50, help='Number of iterations of training loop')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--data', type=str, required=True, help='Path to the training data')
    parser.add_argument('--embedding_size', type=int, default=32, help='Size of the output embedding vector')
    args = parser.parse_args()

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device set to: ", device)

    # Create an instance of LSTM model
    model = LSTMSpeakerEncoder(NUM_MFCC, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, embedding_size=args.embedding_size)
    model.to(device)
    print("Model loaded")

    # Create an instance of triplet dataset
    dataset = data.TripletDataset(args.data)
    data_store = dataset.data
    print("Data loaded")

    # Create a data loader for the dataset
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Define loss function and optimizer
    loss = TripletCosineLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Accumulate losses for each epoch
    train_losses = []

    # Train the model
    for epoch in range(args.epochs):  # Adjust the number of epochs as needed
        print(f"\nEpoch {epoch + 1}/{args.epochs} | ", end="")
        train_loss = train_model(model, train_loader, optimizer, loss, device, data_store)
        train_losses.append(train_loss)
        print(f"Loss: ", train_loss)

    print("Training complete!")

    # Save the model
    torch.save(model.state_dict(), 'my_lstm_model.pth')

if __name__ == '__main__':
    main()