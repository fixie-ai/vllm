nohup python -m vllm.entrypoints.openai.api_server --model=tincans-ai/gazelle-v0.2 --audio-token-id=32000 --api-key=89d313170fb06cc6b9e5933c29ea9353 --served-model-name=fixie-ai/ultravox-v0.1 > vllm.log 2>&1 &

