DATA_DIR="/path/to/dir"

cd $DATA_DIR
mkdir audio
cd audio

for file in $DATA_DIR/*.mp4
do 
    filepath=${file}
    filename=$(basename -z -s ".mp4" $file)
    new_file="${DATA_DIR}/audio/${filename}.mp3"
    ffmpeg -i $filepath $DATA_DIR/audio/$filename.mp3
done 
