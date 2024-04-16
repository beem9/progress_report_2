import torch
import torch.nn as nn
import numpy as np


num_cities = 50
num_examples = 10
input_dim = 2
hidden_dim = 128
sequence_len = num_cities
batch_size = 2
learning_rate = 0.0005
num_epochs = 100

# generate random tsp instances
def generate_tsp_data(num_examples, num_cities):
    data = []
    for _ in range(num_examples):
        coordinates = np.random.rand(num_cities, 2)
        tour = np.argsort(coordinates[:, 0])
        data.append((coordinates, tour))
    return data


def prepare_data(data):
    inputs = np.array([item[0] for item in data])
    targets = np.array([item[1] for item in data])

    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.long)
    return inputs_tensor, targets_tensor



class LSTMTSPSolver(nn.Module):
    def __init__(self, input_dim, hidden_dim, sequence_len, num_layers=2):
        super(LSTMTSPSolver, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, sequence_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out



model = LSTMTSPSolver(input_dim, hidden_dim, sequence_len)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


data = generate_tsp_data(num_examples, num_cities)
inputs, targets = prepare_data(data)


for epoch in range(num_epochs):
    total_loss = 0
    for i in range(0, len(inputs), batch_size):
        batch_inputs = inputs[i:i+batch_size]
        batch_targets = targets[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = 0
        for j in range(sequence_len):
            loss += criterion(outputs, batch_targets[:, j])
        loss /= sequence_len  # Average loss per step in the sequence

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/(len(inputs)/batch_size):.4f}')
