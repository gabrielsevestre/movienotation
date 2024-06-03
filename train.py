import torch
import pandas as pd
from utilities import NoteDataset, NotePredictionModel, collate_fn, EarlyStoppingCallback
from tqdm import tqdm


df = pd.read_csv('./files/preprocessed_df.csv')

train_df = df.sample(frac=0.91, ignore_index=True)
eval_df = df.drop(train_df.index).reset_index()
eval_df = eval_df.drop(columns='index')
train_dataset = NoteDataset(train_df)
eval_dataset = NoteDataset(eval_df)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn, shuffle=True)
eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=8, collate_fn=collate_fn, shuffle=True)

device = torch.device('mps')  # change to your personal device !
model = NotePredictionModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
loss_fn = torch.nn.MSELoss()

earlystopping = EarlyStoppingCallback()


def evaluate():
    model.eval()
    val_loss = 0
    pbar = tqdm(eval_dataloader, total=len(eval_dataloader))
    for batch, inputs in enumerate(eval_dataloader):
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)
        cat_features = inputs['cat_features'].to(device)
        notes = inputs['notes'].to(device)

        output = model.forward(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               cat_features=cat_features)

        loss = loss_fn(output, notes)
        val_loss += loss.item()
        pbar.update(1)

    val_loss /= len(eval_dataloader)
    print(f'Validation Loss: {val_loss}')
    pbar.close()
    return val_loss


def train():

    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0
        for batch, inputs in enumerate(train_dataloader):
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            token_type_ids = inputs['token_type_ids'].to(device)
            cat_features = inputs['cat_features'].to(device)
            notes = inputs['notes'].to(device)

            output = model.forward(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   cat_features=cat_features)

            loss = loss_fn(output, notes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'loss': loss.item()})
            pbar.update(1)

        v = evaluate()
        if earlystopping.on_epoch_end(epoch, logs={'val_loss': v}):
            break
        pbar.close()


if __name__ == '__main__':
    train()
