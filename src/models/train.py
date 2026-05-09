import torch
import torch.nn as nn


def train_model(model, X_train, y_train, epochs=50, lr=0.001):

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):

        model.train()

        optimizer.zero_grad()

        outputs = model(X_train)

        loss = criterion(outputs, y_train)

        loss.backward()

        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")