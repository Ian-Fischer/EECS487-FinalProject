import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
from transformers import AutoTokenizer, BertModel, MobileBertModel
import matplotlib.pyplot as plt
import copy

class MediaBiasDataset:
  def __init__(self, train_df, valid_df, test_df, baseline=True, batch_size=16):
    self.train_df = train_df
    self.valid_df = valid_df
    self.test_df = test_df

    if baseline:
      self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    else:
      self.tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
    
    max_len = 0
    for paragraph in [self.train_df.text.values,self.valid_df.text.values,self.test_df.text.values]:
      # For every sentence...
      for para in paragraph:
          # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
          input_ids = self.tokenizer.encode(para, add_special_tokens=True)

          # Update the maximum sentence length.
          max_len = max(max_len, len(input_ids))

    self.train_dataset = self.build_dataset(train_df.text.values,train_df.label.values, max_len)
    self.valid_dataset = self.build_dataset(valid_df.text.values,valid_df.label.values, max_len)
    self.test_dataset = self.build_dataset(test_df.text.values,test_df.label.values, max_len)
    
    self.train_dataloader = DataLoader(
            self.train_dataset,  # The training samples.
            sampler = RandomSampler(self.train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

    self.validation_dataloader = DataLoader(
            self.valid_dataset, # The validation samples.
            sampler = SequentialSampler(self.valid_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )
    
    self.test_dataloader = DataLoader(
            self.test_dataset, # The validation samples.
            sampler = SequentialSampler(self.test_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

  def build_dataset(self, sentences, labels, max_length):
    input_ids = []
    attention_masks = []

    for sentence in sentences:
      encoded_dict = self.tokenizer.encode_plus(
                        sentence,                      
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = max_length,           # Pad & truncate all sentences.
                        padding="max_length",
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt'
                        )
            
      # Add the encoded sentence to the list.    
      input_ids.append(encoded_dict['input_ids'])
            
      # And its attention mask (simply differentiates padding from non-padding).
      attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    return TensorDataset(input_ids, attention_masks, labels.float())

class BaselineClassifier(nn.Module):
  def __init__(self, drop_rate=0.2):
      super(BaselineClassifier, self).__init__()
      D_in, D_out = 768, 1
      self.bert = BertModel.from_pretrained("bert-base-uncased")
      self.classifier = nn.Sequential(
          nn.Dropout(drop_rate),
          nn.Linear(D_in, D_out))
        
  def forward(self, input_ids, attention_masks):
      outputs = self.bert(input_ids, attention_masks)
      pooled_output = outputs[1]
      outputs = self.classifier(pooled_output)
      return outputs.squeeze()
  
class MobileBertClassifier(nn.Module):
  def __init__(self, drop_rate=0.2):
      super(MobileBertClassifier, self).__init__()
      D_in, D_out = 512, 1
      self.bert = MobileBertModel.from_pretrained("google/mobilebert-uncased")
      self.classifier = nn.Sequential(
          nn.Dropout(drop_rate),
          nn.Linear(D_in, D_out))
        
  def forward(self, input_ids, attention_masks):
      outputs = self.bert(input_ids, attention_masks)
      pooled_output = outputs[1]
      outputs = self.classifier(pooled_output)
      return outputs.squeeze()

def get_predictions(scores, threshold=.5):
    scores = torch.sigmoid(scores)
    predictions = torch.where(scores >= threshold, 1.0, 0.0)
    return predictions

def finetune_model(net, loaders, optim, loss_function, num_epoch=5, device='cpu', verbose=False):
    """
    Train the model.

    Input:
        - net: model
        - trn_loader: dataloader for training data
        - val_loader: dataloader for validation data
        - optim: optimizer
        - num_epoch: number of epochs to train
        - collect_cycle: how many iterations to collect training statistics
        - device: device to use
        - verbose: whether to print training details
    Return:
        - best_model: the model that has the best performance on validation data
        - stats: training statistics
    """
    train_loss, train_loss_ind, val_loss, val_acc, val_loss_ind = [], [], [], [], []
    num_itr = 0
    best_model, best_loss, best_acc, best_epoch = None, 10**10, 0, 0
    if verbose:
        print('------------------------ Start Training ------------------------')
    for epoch in range(num_epoch):
        # Training:
        net.train()
        total_loss = [] 
          
        for b_input_ids, b_input_mask, b_labels  in loaders.train_dataloader:
            b_input_ids = b_input_ids.to(device)
            b_input_mask = b_input_mask.to(device)
            b_labels = b_labels.to(device)
            num_itr += 1
            
            optim.zero_grad()
            scores = net(b_input_ids,b_input_mask)
            loss = loss_function(scores, b_labels)
            total_loss.append(loss.item())

            loss.backward()
            optim.step()
        
        total_loss = sum(total_loss) / len(total_loss)
        train_loss.append(total_loss)
        train_loss_ind.append(epoch)

        if verbose:
            print('Epoch No. {0} training loss: {1:.4f}'.format(
                epoch + 1,
                total_loss
                ))

        # Validation:
        net.eval()
        total_loss = [] # loss for each batch
        preds = []
        labels = []

        with torch.no_grad():
            for b_input_ids, b_input_mask, b_labels  in loaders.validation_dataloader:
                b_input_ids = b_input_ids.to(device)
                b_input_mask = b_input_mask.to(device)
                b_labels = b_labels.to(device)

                scores = net(b_input_ids,b_input_mask)
                loss = loss_function(scores, b_labels)
                total_loss.append(loss.item())
                temp = get_predictions(scores)
                preds.append(temp.cpu())
                labels.append(b_labels.cpu())
  
        y_true = torch.cat(preds)
        y_pred = torch.cat(labels)
        accuracy = (y_true == y_pred).sum() / y_pred.shape[0]

        total_loss = sum(total_loss) / len(total_loss)
        val_loss.append(total_loss)
        val_acc.append(accuracy)
        val_loss_ind.append(epoch)
        if verbose:
            print("Validation Loss: {:.4f}".format(loss))
            print("Validation Acc: {:.4f}".format(accuracy))
        if accuracy > best_acc:
            best_model = copy.deepcopy(net)
            best_epoch = epoch
            best_acc = accuracy
    if verbose:
        print('------------------------ Training Done ------------------------')
    
    stats = {'train_loss': train_loss,
             'train_loss_ind': train_loss_ind,
             'val_loss': val_loss,
             'val_acc': val_acc,
             'val_loss_ind': val_loss_ind,
             'accuracy': best_acc,
    }

    return best_model, stats

def plot_loss(stats):
    """Plot training loss and validation loss."""
    plt.plot(stats['train_loss_ind'], stats['train_loss'], label='Training loss')
    plt.plot(stats['val_loss_ind'], stats['val_loss'], label='Validation loss')
    plt.legend()
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss')
    plt.show()

