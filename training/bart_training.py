from transformers import BartTokenizer, BartForConditionalGeneration, AdamW
import torch
from torch.utils.data import TensorDataset, DataLoader
from utils.split import split_data

def train_bart(df, mutated_claim_col, model_name="facebook/bart-base"):
    # Load the BART tokenizer and model
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # Split the data
    train, val, test = split_data(df, 'text')

    # Prepare the data for training
    train_encodings = tokenizer((train[mutated_claim_col] + tokenizer.sep_token + train['text']).tolist(), truncation=True, max_length=512,padding="max_length")['input_ids']
    val_encodings = tokenizer((val[mutated_claim_col] + tokenizer.sep_token + val['text']).tolist(), truncation=True, max_length=512,padding="max_length")['input_ids']
    test_encodings = tokenizer((test[mutated_claim_col] + tokenizer.sep_token + test['text']).tolist(), truncation=True, max_length=512,padding="max_length")['input_ids']
    train_encodings = torch.tensor(train_encodings)
    val_encodings = torch.tensor(val_encodings)
    test_encodings = torch.tensor(test_encodings)

    train_labels = tokenizer(train['claim'].tolist(), truncation=True, padding=True)['input_ids']
    val_labels = tokenizer(val['claim'].tolist(), truncation=True, padding=True)['input_ids']
    test_labels = tokenizer(test['claim'].tolist(), truncation=True, padding=True)['input_ids']
    train_labels = torch.tensor(train_labels)
    val_labels = torch.tensor(val_labels)
    test_labels = torch.tensor(test_labels)

    # Convert to PyTorch Datasets
    train_dataset = TensorDataset(train_encodings, train_labels)
    val_dataset = TensorDataset(val_encodings, val_labels)

    print("Training set size:", len(train_dataset))
    print("Validation set size:", len(val_dataset))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    batch_size=64

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_encodings, batch_size=batch_size, shuffle=False)

    num_epochs = 3

    train_loss_list = []
    val_loss_list = []

    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(num_epochs):
        # Switch to training mode
        model.train()
        train_loss = 0
        for input_batch, claim_batch in train_loader:
            # move batch to device
            input_batch = input_batch.to(device)
            claim_batch = claim_batch.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_batch, labels=claim_batch)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_loss_list.append(train_loss)
        
        # Switch to evaluation mode
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_batch, claim_batch in val_loader:
                # move batch to device
                input_batch = input_batch.to(device)
                claim_batch = claim_batch.to(device)
                outputs = model(input_ids=input_batch, labels=claim_batch)
                loss = outputs[0]
                val_loss += loss.item()
            val_loss /= len(val_loader)
            val_loss_list.append(val_loss)
    
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # Switch to evaluation mode
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for input_batch, claim_batch in test_loader:
            # move batch to device
            input_batch = input_batch.to(device)
            claim_batch = claim_batch.to(device)
            outputs = model(input_ids=input_batch, labels=claim_batch)
            loss = outputs[0]
            test_loss += loss.item()
        test_loss /= len(test_loader)
    print(f"Test Loss = {test_loss:.4f}")

    # Return the model
    return model