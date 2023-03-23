import torch

from utils import get_dataset, get_model, progress_bar


def training_step(model, batch, optimizer, scaler):
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        outputs = model(
            input_ids=torch.tensor(batch['input_ids']),
            attention_mask=torch.tensor(batch['attention_mask']),
            labels=torch.tensor(batch['labels']),
        )
        step_loss = outputs[0].to('cuda')
    scaler.scale(step_loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return step_loss.detach()


def main():
    n_epochs = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    learning_rate = 3e-5

    # Data
    train_loader = get_dataset('./data/train.csv')

    # Model
    peft_model = get_model()
    peft_model.to(device)

    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(peft_model.parameters(), lr=learning_rate)

    scaler = torch.cuda.amp.GradScaler()

    # Train
    peft_model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        train_loss = torch.tensor(0.).to(device)
        for batch_idx, batch in enumerate(train_loader, start=1):
            step_loss = training_step(peft_model, batch, optimizer, scaler)
            train_loss += step_loss

            msg = f"Loss: {train_loss/(batch_idx+1):.3f}"
            progress_bar(batch_idx, len(train_loader), msg)

            train_loss = torch.tensor(0.).to(device)
            total_loss += train_loss


if __name__ == '__main__':
    main()
