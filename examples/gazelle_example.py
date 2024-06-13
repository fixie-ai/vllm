import argparse
import math
import os
#import subprocess
import torchaudio
from vllm.model_executor.models import gazelle
import transformers
from transformers import AutoProcessor, AutoModel
import torch
import gc
from vllm import LLM, SamplingParams
from vllm.sequence import MultiModalData

from gazelle_hf import GazelleConfig, GazelleForConditionalGeneration

# The assets are located at `s3://air-example-data-2/vllm_opensource_llava/`.
transformers.AutoConfig.register("gazelle", GazelleConfig)
transformers.AutoModel.register(GazelleConfig, GazelleForConditionalGeneration)

torch.set_printoptions(edgeitems=8, sci_mode=True)




def run_infer(llm, prompt, sampling_params, multi_modal_data):
    outputs = llm.generate(prompt, sampling_params=sampling_params, multi_modal_data=multi_modal_data)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

def run_gazelle():

    
    # This should be provided by another online or offline component.
    #audio = torch.load("audio/test6.pt")
    sampling_params = SamplingParams(max_tokens=100, temperature=0.5)      
  
    #audio, sr = torchaudio.load("audio/test6.wav")
    #print("sr", sr)
    import librosa
    import numpy as np
    torch.set_printoptions(edgeitems=3, sci_mode=True)

    audio, sample_rate = librosa.load("audio/test6.wav", sr=16000)
    #audio = librosa.resample(
    #    audio, orig_sr=sample_rate, target_sr=16000
    #)
    #audio = np.expand_dims(audio, 0)
    audio = torch.from_numpy(audio)
    #transform = torchaudio.transforms.Resample(sr, 16000)
    #audio = transform(audio).to(torch.float16)
    print("input audio", audio, audio.dtype)
    dtype = torch.float32
    audio_processor = AutoProcessor.from_pretrained(
        "facebook/wav2vec2-base-960h"
    )
    x = audio_processor(audio, sampling_rate=16000, padding="longest", return_tensors="pt")
    audio = x.get("input_features") or x.get("input_values")    
    print("presqueeze audio", audio, audio.shape)
    #audio = audio.squeeze(0).to(torch.bfloat16).to("cuda")
    audio = audio.to(dtype).to("cuda")
    print("proc audio", audio, audio.shape, audio.dtype)

    audio_num_tokens = math.ceil(audio.shape[1] / 16000 * 6.2375)    
    prompt = '[INST] Respond to the following audio: ' + '<|audio|>' * audio_num_tokens + ' [/INST]'
    multi_modal_data=MultiModalData(type=MultiModalData.Type.AUDIO, data=audio)
    print(f"Loaded audio with len {audio.shape} and dtype {audio.dtype}")

    
    
    
    #audio_encoder = gazelle.audio_tower
    #audio_encoder2 = AutoModel.from_pretrained("facebook/wav2vec2-base-960h").to(dtype).to("cuda")
        #y = audio_encoder.feature_extractor.forward(audio) #conv_layers[0].conv.forward(audio)
        #yp = [(n, p) for n, p in audio_encoder.named_parameters()]
        #zp = [(n, p) for n, p in audio_encoder2.named_parameters()]
        # use torch.allclose to compare yp and zp
        #for i in range(len(yp)):
        #    print(i, yp[i][0], zp[i][0], yp[i][1][0], zp[i][1][0])
        #print("gz params", audio_encoder.feature_extractor.projector.named_parameters())
        #print("fb params", audio_encoder2.feature_extractor.projector.named_parameters())
        #with torch.inference_mode():
        #    y = audio_encoder.forward(audio).last_hidden_state
        #    z = audio_encoder2.forward(audio).last_hidden_state

        #y = audio_encoder(audio).last_hidden_state
        #print("gz encoded audio", y, y.shape, y.dtype)
        #print("fb encoded audio", z, z.shape, z.dtype)
    if True:
        gazelle = GazelleForConditionalGeneration.from_pretrained("tincans-ai/gazelle-v0.2", torch_dtype=dtype).to(dtype).to("cuda")
        gazelle.generate(prompt, sampling_params=sampling_params, multi_modal_data=multi_modal_data)
    
        
    llm = LLM(
        model="tincans-ai/gazelle-v0.2",
        #image_input_type="pixel_values",
        audio_token_id=32000,
        #image_input_shape="1,3,336,336",
        #image_feature_size=576,
        trust_remote_code=True
    )

    run_infer(llm, prompt, sampling_params, multi_modal_data)
    
    prompt = "[INST] What's your greatest accomplishment? [/INST]"
    multi_modal_data = None
    run_infer(llm, prompt, sampling_params, multi_modal_data)


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
