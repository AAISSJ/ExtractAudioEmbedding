import torchaudio
import torch
import pandas as pd

def extract_audio_embeds(bundle, audio_files, target_layer=-1):

    # Build the model and load pretrained weight.
    model = bundle.get_model()
    embeddings=[]
    
    for audio_path in audio_files:
        waveform, sample_rate = torchaudio.load(audio_path)

        if sample_rate != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

        # Extract acoustic features
        # The returned features is a list of tensors. Each tensor is the output of a transformer layer.
        with torch.inference_mode():
            features, _ = model.extract_features(waveform)
            
        embeddings.append(features[target_layer].numpy())
    
    return embeddings


DATA_DIR='path/to/data'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_json('../total_asr_data.json')

bundle_list=[
        # Wav2Vec2Bundle instantiates models that generate acoustic features that can be used for downstream inference and fine-tuning.
        torchaudio.pipelines.HUBERT_BASE,torchaudio.pipelines.WAV2VEC2_XLSR53, torchaudio.pipelines.WAV2VEC2_BASE
    ]

for i,bundle in enumerate(bundle_list):
    extracted_embeddings = extract_audio_embeds(bundle, path, -1)
    df[str(i)]=extracted_embeddings
    print("***"+ bundle+"done***")

df.to_csv('./embeddings.csv',index=False)
