import torch
import huggingface_hub
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          DataCollatorWithPadding,
                          TrainingArguments,
                          pipeline)
import pandas as pd
from .cleaner import Cleaner 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from datasets import Dataset
from .training_utils import get_class_weights,compute_metrics
from .custom_trainer import CustomTrainer
import gc

class JutsuClassifier():
  def __init__(self,model_path,
               data_path=None,
               text_column_name='text',
               label_column_name='jutsus',
               model_name = "distilbert/distilbert-base-uncased",
               test_size=0.2,
               num_labels=3,
               huggingface_token=None):
    self.model_path = model_path
    self.data_path =data_path
    self.text_column_name = text_column_name
    self.label_column_name = label_column_name
    self.model_name = model_name
    self.test_size = test_size
    self.num_labels = num_labels
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.huggingface_token = huggingface_token
    
    if self.huggingface_token is not None:
      huggingface_hub.login(self.huggingface_token)
    
    self.tokenizer = self.load_tokenizer()
    
    if not huggingface_hub.repo_exists(self.model_path):
      if data_path is None:
        raise ValueError("Data path is None")
      train_data,test_data = self.load_data(self.data_path)
      train_data_df = train_data.to_pandas()      
      class_weights = get_class_weights(train_data_df)
      self.train_model(train_data,test_data,class_weights)
      
    self.model = self.load_model(self.model_path)
    
  def load_model(self,model_path):
    model = pipeline('text-classification',model=model_path,return_all_scores=True)
    return model
    
  def train_model(self,train_data,test_data,class_weights):
    model = AutoModelForSequenceClassification.from_pretrained(self.model_name,
                                                               num_labels=self.num_labels,
                                                               id2label=self.label_dict)
    data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
    training_args = TrainingArguments(
      output_dir=self.model_path,    
      learning_rate=2e-4,            
      per_device_train_batch_size=8,   
      per_device_eval_batch_size=8,   
      num_train_epochs=5,             
      weight_decay=0.01,              
      evaluation_strategy="epoch",   
      logging_strategy="epoch",     
      push_to_hub=True                
    )
    
    trainer = CustomTrainer(
      model=model,
      args=training_args,
      train_dataset = train_data,
      eval_dataset = test_data,
      tokenizer = self.tokenizer,
      data_collator=data_collator,
      compute_metrics = compute_metrics
    )

    trainer.set_device(self.device)
    trainer.set_class_weights(class_weights)

    trainer.train()
    trainer.push_to_hub()  # 🔁 Ensures complete model gets pushed
    self.tokenizer.push_to_hub(self.model_path)  # 🔁 Ensures tokenizer is pushed
    
    del trainer,model
    gc.collect()
    
    if self.device == 'cuda':
      torch.cuda.empty_cache()
  
  def simplify_jutsu(self,t):
    if "Ninjutsu" in t:
      return "Ninjutsu"
    if "Taijutsu" in t:
      return "Taijutsu"
    if "Genjutsu" in t:
      return "Genjutsu"
  
  def preprocess_function(self,tokenizer,examples):
    return tokenizer(examples['text_cleaned'],truncation=True)
  
  def load_data(self,data_path):
    df = pd.read_json(data_path,lines=True)
    df["jutsu_type_simplified"] = df.jutsu_type.apply(self.simplify_jutsu)
    df['text'] = df["jutsu_name"] + ". " + df['jutsu_description']
    df['jutsus'] = df['jutsu_type_simplified']
    df = df[['text','jutsus']]
    df = df.dropna()
    
    cleaner = Cleaner()
    df["text_cleaned"] = df[self.text_column_name].apply(cleaner.clean)
    
    le = preprocessing.LabelEncoder()
    le.fit(df[self.label_column_name].tolist())
    label_dict = {index:label_name for index,label_name in enumerate(le.__dict__["classes_"].tolist())}
    self.label_dict = label_dict
    df['labels'] = le.transform(df[self.label_column_name].tolist())
    df_train,df_test = train_test_split(df,test_size=0.2,stratify=df['labels'])
    train_dataset = Dataset.from_pandas(df_train)
    test_dataset = Dataset.from_pandas(df_test)
    tokenized_train = train_dataset.map(lambda examples:self.preprocess_function(self.tokenizer,examples),batched=True)
    tokenized_test = test_dataset.map(lambda examples:self.preprocess_function(self.tokenizer,examples),batched=True)
    return tokenized_train,tokenized_test
    
  def load_tokenizer(self):
    if huggingface_hub.repo_exists(self.model_path):
      tokenizer = AutoTokenizer.from_pretrained(self.model_path)
    else:
      tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    return tokenizer
  
  def postprocess(self,model_output):
    output = []
    for pred in model_output:
      label = max(pred,key=lambda x:x['score'])['label']
      output.append(label)
    return output
  
  def classify_jutsu(self,text):
    model_output = self.model(text)
    predictions = self.postprocess(model_output)
    return predictions
      
    