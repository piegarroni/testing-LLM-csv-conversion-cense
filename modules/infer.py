import pandas as pd
import json
import os 
import re 
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch


class GenerateDataset():

    def __init__(self, file_path, model, tokenizer):
        self.file_path = file_path 
        self.tokenizer = tokenizer
        self.model = model

    def extract_file_name(self, dictionary):
        return dictionary.get('file_name')


    
    def preprocess_inference_data(self):
        df = pd.read_csv(self.file_path)
        df['file_name'] = self.file_path.split("/")[-1].split(".")[0]
        data = [{"file name": self.file_path.split("/")[-1].split(".")[0], "source" : df.loc[i].to_dict(), 'target': "xxx"} for i in range(len(df))]
        return data


    def infer(self, row):

        text = str(row)
        #print("Original row: \n", text)
        text = text.split(" 'target'")[0]
        #print("Truncated version: \n",text)
        batch = self.tokenizer(text, return_tensors='pt')
        input_ids = batch.to('cuda')

        with torch.cuda.amp.autocast():
            output_tokens = self.model.generate(**input_ids, max_new_tokens=250, num_beams=1)#, eos_token_id=eos_token_id)
        
        result = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True).split('\n')[0]
        #print("Generated text: \n", result)

        return result


    def generate_dataframe(self):

        # load data
        data = self.preprocess_inference_data()
  
        # generate data row by row
        results = [self.infer(i) for i in data]

        # pattern to extract target dictionary (generated part)
        pattern_target = r"'target': {(.*?)}"
        generated_dicts = []

        for generated in results:

            matches = re.findall(pattern_target, generated, re.DOTALL)

            for match in matches: # THERE SHOULD BE ONLY ONE SO CHANGE THIS CODE!
                match = match.replace("'", '"').replace(" nan", ' "nan"')
                string = '{' + match + '}'
                generated_dict = json.loads(string)

            generated_dicts.append(generated_dict)

        generated_dataset = pd.DataFrame(generated_dicts)
        
        return generated_dataset

    def save_dataframe(self):
        generated_dataset = self.generate_dataframe()
        generated_dataset.to_csv(f"data/{self.file_path.split('/')[-1].replace('.csv', '') + '-generated.csv'}")

        return generated_dataset
    



