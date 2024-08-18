import os
from huggingface_hub import hf_hub_download

def download_model():
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    save_directory = "/models"
    hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        print("Error: Hugging Face token not found in environment variables.")
        return

    try:
        files_to_download = [
            "config.json",
            "generation_config.json",
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
            "model.safetensors.index.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json"
        ]

        for file in files_to_download:
            print(f"Downloading file: {file}")
            hf_hub_download(repo_id=model_name, filename=file, local_dir=save_directory, token=hf_token)

        print(f"Model files downloaded successfully to {save_directory}")
    except Exception as e:
        print(f"Error downloading model: {str(e)}")

if __name__ == "__main__":
    download_model()
