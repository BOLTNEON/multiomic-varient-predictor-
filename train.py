import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
from model import MultiOmicsNN

def train_model(expr_data, prot_data, labels, epochs=10):
    expr_tensor = torch.tensor(expr_data, dtype=torch.float32)
    prot_tensor = torch.tensor(prot_data, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(expr_tensor, prot_tensor, label_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = MultiOmicsNN(expr_tensor.shape[1], prot_tensor.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0
        for expr_batch, prot_batch, label_batch in loader:
            optimizer.zero_grad()
            outputs = model(expr_batch, prot_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    return model
