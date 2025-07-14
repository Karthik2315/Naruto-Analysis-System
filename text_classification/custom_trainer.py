import torch
from torch import nn
from transformers import Trainer

class CustomTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False):
    labels = inputs.get("labels").to(self.device)
    class_weights = torch.tensor(self.class_weights,dtype=torch.float)
    class_weights = class_weights.to(self.device)
    outputs = model(**inputs)
    logits = outputs.get("logits")
    logits = logits.float()
    loss_fct = nn.CrossEntropyLoss(weight=class_weights)
    loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
    return (loss, outputs) if return_outputs else loss
  
  def set_class_weights(self,class_weights):
    self.class_weights = class_weights
    
  def set_device(self,device):
    self.device = device