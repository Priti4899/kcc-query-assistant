import pandas as pd
import re

def clean_text(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text

def preprocess_kcc_csv(input_path,output_path):
    df = pd.read_csv("D:\Mypython\kcc_query_assistant\data\questionsv4.csv")
    df.dropna(subset=["questions", "answers"], inplace=True)
    df["Question"] = df["questions"].apply(clean_text)
    df["Answer"] = df["answers"].apply(clean_text)
    
    df["pair"] = df.apply(lambda x: f"Q: {x['questions']}\nA: {x['answers']}", axis=1)
    df[["Question", "answers", "pair"]].to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

# Run manually
if __name__ == "__main__":
    preprocess_kcc_csv("D:\Mypython\kcc_query_assistant\data\questionsv4.csv", "D:\Mypython\kcc_query_assistant\data\cleaned_kcc1.csv")
