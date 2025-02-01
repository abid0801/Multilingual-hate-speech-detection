from huggingface_hub import upload_folder

upload_folder(
    folder_path="multilingual_hate_speech_model",  
    repo_id="abid0801/multilingual-hate-speech-model", 
    repo_type="model"
)
