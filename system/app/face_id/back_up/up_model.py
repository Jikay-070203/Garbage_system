from huggingface_hub import upload_folder
# Thư mục chứa model
upload_folder(
    folder_path=r"D:\SourceCode\ProjectOJT\complete\OJT_TASK3_LOCAL\FASTAPI_V8\model",  
    path_in_repo="",  #upload all
    repo_id="hoanguyenthanh07/Face_onnx", 
    repo_type="model",  
    token="" #token
)



