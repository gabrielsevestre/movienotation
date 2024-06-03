import re
import torch
from transformers import BertModel, BertTokenizer
from peft import LoraConfig, get_peft_model


list_of_genres = ['Crime', 'Comedy', 'Adventure', 'Action', 'Science Fiction',
                  'Animation', 'Family', 'Drama', 'Romance', 'Music', 'Fantasy',
                  'Thriller', 'War', 'Western', 'Mystery', 'History', 'Horror',
                  'Documentary', 'Foreign', 'TV Movie']


class NotePredictionModel(torch.nn.Module):

    def __init__(self,
                 text_model='bert-base-uncased',
                 r=32,
                 lora_alpha=16,
                 lora_dropout=0.1,
                 num_cats=20,
                 hidden_dim=128
                 ):

        super(NotePredictionModel, self).__init__()

        text_model = BertModel.from_pretrained(text_model)
        config = LoraConfig(r=r,
                            lora_alpha=lora_alpha,
                            lora_dropout=lora_dropout)
        self.text_model = get_peft_model(text_model, config)
        self.combined_fc1 = torch.nn.Linear(text_model.config.hidden_size + num_cats, hidden_dim)
        self.combined_fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask, token_type_ids, cat_features):

        bert_output = self.text_model(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids).pooler_output

        combined_features = torch.cat((bert_output, cat_features), dim=-1)

        x = self.combined_fc1(combined_features)
        x = torch.relu(x)
        x = self.combined_fc2(x)

        return x


class NoteDataset(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        title = self.df.loc[idx]['title']
        overview = self.df.loc[idx]['overview']
        tokenized_text = self.tokenizer(title, overview, padding='max_length', return_tensors='pt')
        cat_features = torch.tensor(self.df.loc[idx][list_of_genres])
        label = torch.tensor(self.df.loc[idx]['vote_average'], dtype=torch.float32)
        return tokenized_text, cat_features, label


def collate_fn(batch):
    input_ids = torch.cat([f[0]['input_ids'] for f in batch], dim=0)
    token_type_ids = torch.cat([f[0]['token_type_ids'] for f in batch], dim=0)
    attention_mask = torch.cat([f[0]['attention_mask'] for f in batch], dim=0)
    cat_features = torch.stack([f[1] for f in batch], dim=0)
    notes = torch.stack([f[2] for f in batch])
    return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask,
            'cat_features': cat_features, 'notes': notes.unsqueeze(dim=1)}


class EarlyStoppingCallback:
    def __init__(self, patience=3):
        self.patience = patience
        self.best_loss = None
        self.wait = 0
        self.best_epoch = 1

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs['val_loss']
        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_epoch = epoch + 1
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                return True
        return False


def process_text_for_inference(title, overview):
    title, overview = str(title), str(overview)
    title, overview = title.lower(), overview.lower()
    title = re.sub(r'\s+', ' ', title)
    title = re.sub(r'[^a-zA-Z0-9\s]', '', title)
    overview = re.sub(r'\s+', ' ', overview)
    overview = re.sub(r'[^a-zA-Z0-9\s]', '', overview)
    return title, overview


def inference(model, tokenizer, title, overview, genres):
    title, overview = process_text_for_inference(title, overview)
    tok_text = tokenizer(title, overview,
                         padding='max_length',
                         truncation=True,
                         return_tensors='pt')
    input_ids = tok_text['input_ids']
    token_type_ids = tok_text['token_type_ids']
    attention_mask = tok_text['attention_mask']
    cat_features = [0 if list_of_genres[i] not in genres else 1 for i in range(len(list_of_genres))]
    cat_features = torch.reshape(torch.tensor(cat_features), (1, 20))
    output = model(input_ids=input_ids,
                   attention_mask=attention_mask,
                   token_type_ids=token_type_ids,
                   cat_features=cat_features)
    output = output.item()
    return output
