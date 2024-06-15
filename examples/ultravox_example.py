import argparse
import math
import os
#import subprocess
import librosa
from vllm.model_executor.models.ultravox import UltravoxConfig, UltravoxModel
import transformers
import torch

from vllm import LLM, SamplingParams
from vllm.sequence import MultiModalData


# The assets are located at `s3://air-example-data-2/vllm_opensource_llava/`.
transformers.AutoConfig.register("ultravox", UltravoxConfig)
transformers.AutoModel.register(UltravoxConfig, UltravoxModel)


def run_infer(llm, prompt, sampling_params, multi_modal_data):
    outputs = llm.generate(prompt, sampling_params=sampling_params, multi_modal_data=multi_modal_data)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

def run_ultravox():
    # This should be provided by another online or offline component.
    #audio = torch.load("audio/test6.pt")
    sampling_params = SamplingParams(max_tokens=100, temperature=0.0)      
    audio, sample_rate = librosa.load("audio/test6.wav", sr=16000)
    audio = librosa.resample(
        audio, orig_sr=sample_rate, target_sr=16000
    )
    audio = torch.from_numpy(audio)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained("fixie-ai/ultravox-v0.2")    
    audio_processor = transformers.AutoProcessor.from_pretrained("openai/whisper-small")
    processed = audio_processor(audio=audio, sampling_rate=16000, return_tensors="pt", padding="longest")  
    audio_values = processed.get("input_features")    

    # Shape is [1, 80, M] where M is the number of 10ms frames.
    # These frames will get stacked 8 high with a 2-frame stride, so we divide by 16
    # to get the number of audio embeddings.
    audio_num_tokens = math.ceil(audio_values.shape[2] / 16)
    prompt = 'You are Albert Einstein. Respond to the following audio:\n' + '<|audio|>'
    prompt = prompt.replace("<|audio|>", '<|reserved_special_token_0|>' * audio_num_tokens)
    text_input = tokenizer.apply_chat_template(conversation=[{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)    
    
    multi_modal_data=MultiModalData(type=MultiModalData.Type.AUDIO, data=audio_values)
    print(f"Loaded audio with len {audio_values.shape} and dtype {audio_values.dtype}")
        
    llm = LLM(
        model="fixie-ai/ultravox-v0.2",
        #image_input_type="pixel_values",
        audio_token_id=128002,
        #image_input_shape="1,3,336,336",
        #image_feature_size=576,
        trust_remote_code=True
    )

    run_infer(llm, text_input, sampling_params, multi_modal_data)    


def main(args):
    run_ultravox()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo on Ultravox")
    args = parser.parse_args()
    # Download from s3
    s3_bucket_path = "s3://air-example-data-2/vllm_opensource_ultravox/"
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
