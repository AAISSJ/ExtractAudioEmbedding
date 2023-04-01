import pandas as pd 
import torch
from huggingsound import SpeechRecognitionModel
import librosa

def main():
    df=pd.read_json('/PATH/to/230330_basic_info.json')
 
    tran_list = []
    pre_lan = None
    for i,row in df.iterrows():
        if row.lan !=pre_lan :
            pre_lan = row.lan
            if row.lan == 'en':
                model_id = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
            else : 
                model_id = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
            device = f"cuda:{1}" if torch.cuda.is_available() else "cpu"
            model = SpeechRecognitionModel(model_id, device=device)
        with torch.no_grad():
            try:
                transcription = model.transcribe([row.path], batch_size=1)
                tran_list.append(transcription[0]['transcription'])
            except : 
                tran_list.append(None)
    
    df['tran']=tran_list
    total = df.dropna().reset_index(drop=True)
    total.to_json("/PATH/to/final_data.json",orient='records')

if __name__=="__main__":
    main()
