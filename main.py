# Library import
from utils.import_lib import *
from utils.datasets import get_train_dataloader, get_test_dataloader
from model import NeuralNet

# Hyperparameters
epochs = 10
batch_size = 64
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# DataLoader
train_dataloader = get_train_dataloader()
test_dataloader = get_test_dataloader()

# First batch extract
imgs, labels = next(iter(train_dataloader))
print(f"batch imags shape: {imgs.shape}\n")
print(f"batch labels shape: {labels.shape}\n")

# Model
model = NeuralNet()
model.to(device)
# loss & optimizer
loss_fn = nn.CrossEntropyLoss() # Softmax + NLLloss
optimizer = torch.optim.SGD(model.parameters(), lr = lr)

# Train
def train(dataloader, model, loss_fn, optimizer):
    
    # total loss & iteration counter
    total_loss = 0
    cnt = 0

    # train mode
    model.train()

    for imgs, labels in tqdm(dataloader, desc = "Training"):

        # make all grad of model param to 0
        optimizer.zero_grad()
        # forward
        pred = model(imgs)
        # loss & backward
        loss = loss_fn(pred, labels)
        loss.backward()
        # parameter update
        optimizer.step()

        total_loss += loss.item()
        cnt += 1
    
    avg_loss = total_loss / cnt

    return avg_loss # Returns epoch avg loss

# Test 
def test(dataloader, model, loss_fn):
    # eval mode
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    test_loss, correct = 0, 0

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc = "Evaluation"):

            pred = model(imgs)
            loss = loss_fn(pred, labels)
            test_loss += loss.item()
            correct += (torch.argmax(pred, dim = 1) == labels).type(torch.float).sum().item()

    test_loss = test_loss / num_batches
    accuracy = correct / size

    print(f"test loss : {test_loss}, accuracy : {accuracy * 100}%\n")


if __name__ == "__main__":
    for epoch in range(epochs):
        print(f"================== Epoch {epoch + 1} ===================")
        # Train
        avg_loss = train(train_dataloader, model, loss_fn, optimizer)
        print(f"epoch {epoch+1} loss : {avg_loss}\n")
        # Validation
        test(test_dataloader, model, loss_fn)
    
    # Save model state (save model weights)
    torch.save(model.state_dict(), './model_weights.pth')

    # Save model (save weights + model structure)
    torch.save(model, './model.pth')