import pandas as pd 
import opensmile 
import os 

def extract_feature(smile, path, name_list):
    feature_list=[]
    for name in name_list : 
        y = smile.process_file(path+name)
        feature_list.append(y)
    feature_df=pd.concat(feature_list)
    return feature_df


AUDIO_DIR='/path/to/dir'

os.chdir(AUDIO_DIR)
audio_list=os.listdir()

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

result_df=extract_feature(smile, AUDIO_DIR, audio_list)

result_df.to_csv('./opensmile_feature.csv',index=False)
