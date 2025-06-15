import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import torch
from transformers import pipeline
import numpy as np
from nltk import sent_tokenize
import nltk
import pandas as pd
import sys
import pathlib
folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(folder_path.parent))
from utils import load_subtitles_dataset
nltk.download('punkt')
nltk.download('punkt_tab')


class ThemeClassifier():
  def __init__(self,theme_list):
    self.model_name = "facebook/bart-large-mnli"
    self.device = 0 if torch.cuda.is_available() else -1
    self.theme_list = theme_list
    
    self.theme_classifier = self.load_model(self.device,self.model_name)
    
  def load_model(self,device,model_name):
    theme_classifier = pipeline(
      "zero-shot-classification",
      model=model_name,
      framework = "pt",
      device = device
    )
    
    return theme_classifier
  
  def get_themes_interface(self,script):
    script_sentences = sent_tokenize(script)
    sentence_batch_size = 20
    script_batches = []
    for index in range(0,len(script_sentences),sentence_batch_size):
      sent = " ".join(script_sentences[index:index+sentence_batch_size])
      script_batches.append(sent)
      
    theme_output = self.theme_classifier(
      script_batches,
      self.theme_list,
      multi_label=True
    )
    
    themes = {}
    for theme in theme_output:
      for label,score in zip(theme['labels'],theme['scores']):
        if label not in themes:
          themes[label] = []
        themes[label].append(score)
    
    themes = {key:np.mean(np.array(value)) for key,value in themes.items()}
    return themes 
  
  def get_themes(self,dtset_path,save_path=None):
    
    if save_path is not None and os.path.exists(save_path):
      return pd.read_csv(save_path)
    df = load_subtitles_dataset(dtset_path)
    output_themes = df['Script'].apply(self.get_themes_interface)
    themes_df = pd.DataFrame(output_themes.tolist())
    df[themes_df.columns] = themes_df
    
    if save_path is not None:
      df.to_csv(save_path,index=False)
    return df
      