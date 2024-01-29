import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import model
from feature_extraction import NUM_MFCC
import load_data
from loss import TripletCosineLoss

# Model hyperparameters
HIDDEN_SIZE = 20
NUM_LAYERS = 2
LEARNING_RATE = 0.003

def train_model(model, train_loader, optimizer, loss, device):

    losses = []
    model.train()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Neural Network Training Script')
    parser.add_argument('--epochs', type=int, default=50, help='Number of iterations of training loop')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the training data')
    parser.add_argument('--embedding_size', type=int, default=32, help='Size of the output embedding vector')
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create an instance of LSTM model
    model = model.LSTMSpeakerEncoder(NUM_MFCC, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, embedding_size=args.embedding_size)
    model.to(device)

    # Create an instance of triplet dataset
    dataset = load_data.TripletDataset(args.dataset_path)

    # Create a data loader for the dataset
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Define loss function and optimizer
    loss = TripletCosineLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    for epoch in range(args.epochs):  # Adjust the number of epochs as needed
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_model(model, train_loader, optimizer, loss, device)

    print("Training complete!")

if __name__ == '__main__':
    main()