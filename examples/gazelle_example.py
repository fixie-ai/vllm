import argparse
import math
import os
#import subprocess
import torchaudio
from vllm.model_executor.models import gazelle
import transformers
#import torch

from vllm import LLM, SamplingParams
from vllm.sequence import MultiModalData

# The assets are located at `s3://air-example-data-2/vllm_opensource_llava/`.
transformers.AutoConfig.register("gazelle", gazelle.GazelleConfig)
transformers.AutoModel.register(gazelle.GazelleConfig, gazelle.GazelleForConditionalGeneration)


def run_gazelle():
    llm = LLM(
        model="tincans-ai/gazelle-v0.1",
        #image_input_type="pixel_values",
        audio_token_id=32000,
        #image_input_shape="1,3,336,336",
        #image_feature_size=576,
        trust_remote_code=True
    )
    
    # This should be provided by another online or offline component.
    #audio = torch.load("audio/test6.pt")
    audio, sr = torchaudio.load("audio/test6.wav")
    transform = torchaudio.transforms.Resample(sr, 16000)
    audio = transform(audio)
    audio_num_tokens = math.ceil(audio.shape[1] / 16000 * 6.2375)    
    prompt = '[INST] Transcribe this audio exactly: ' + '<|audio|>' * audio_num_tokens + ' [/INST]'
    print(f"Loaded audio with len {audio.shape} and dtype {audio.dtype}")

    sampling_params = SamplingParams(max_tokens=100)        
    outputs = llm.generate(prompt, sampling_params=sampling_params,
                           multi_modal_data=MultiModalData(
                               type=MultiModalData.Type.AUDIO, data=audio))
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)



def main(args):
    run_gazelle()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo on Gazelle")
    args = parser.parse_args()
    # Download from s3
    s3_bucket_path = "s3://air-example-data-2/vllm_opensource_gazelle/"
    local_directory = "audio"

    # Make sure the local directory exists or create it
    os.makedirs(local_directory, exist_ok=True)

    # Use AWS CLI to sync the directory, assume anonymous access
    """
    subprocess.check_call([
        "aws",
        "s3",
        "sync",
        s3_bucket_path,
        local_directory,
        "--no-sign-request",
    ])
    """
    main(args)
