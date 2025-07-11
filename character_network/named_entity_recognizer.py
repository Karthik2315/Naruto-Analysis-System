import spacy
from nltk import sent_tokenize
import os 
import sys
import pathlib
folder_path = pathlib.Path().parent.resolve()
sys.path.append(str(folder_path))
from utils import load_subtitles_dataset
import pandas as pd
from ast import literal_eval
import torch
class NamedEntityRecognizer:
  def __init__(self):
    self.nlp_model = self.load_model()
    
  def load_model(self):
    if torch.cuda.is_available():
      spacy.require_gpu()
    nlp = spacy.load("en_core_web_trf")
    return nlp
  
  def get_ners_inference(self,script):
    script_sentence = sent_tokenize(script)
    result = []
    for script in script_sentence:
      doc = self.nlp_model(script)
      ner = set()
      for ent in doc.ents:
        if ent.label_ == "PERSON":
          full_name = ent.text
          first_name = full_name.split(" ")[0].strip()
          ner.add(first_name)
      result.append(ner)
    return result
  
  def get_ners(self,dataset_path,save_path=None):
    if save_path is not None and os.path.exists(save_path):
      df = pd.read_csv(save_path)
      df['ners'] = df['ners'].apply(lambda x: literal_eval(x) if isinstance(x,str) else x)
      return df
    
    df = load_subtitles_dataset(dataset_path)    
    df['ners'] = df.Script.apply(self.get_ners_inference)
    if save_path is not None:
      df.to_csv(save_path,index=False)
      
    return df
    
    
  
          