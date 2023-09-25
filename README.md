# fast-embeddings-api

An OpenAI-like API for Massive Text Embeddings using FastAPI.

## How to run
A CUDA GPU is mandatory. Install the requirements:
```bash
pip install -r requirements
```

Start the server by defining a model with **MODEL** variable:
```bash
MODEL=BAAI/bge-base-en-v1.5 python -m src.server
```

By default, it runs in `0.0.0.0:8000`. You can change this by defining variables **HOST** and/or **PORT**.

## Docker image
You can run the application using Docker with the following command:
```bash
docker run -it --gpus all -p 8000:8000 -e MODEL=BAAI/bge-base-en-v1.5 vokturz/fast-embeddings-api
```

## MTEB Benchmark
You can check the best embeddings model on https://huggingface.co/spaces/mteb/leaderboard

## Credits

Mainly based in [Optimum-Benchmark x MTEB by HuggingFace](https://github.com/huggingface/optimum-benchmark/tree/main/examples/fast-mteb) and [limcheekin/open-text-embeddings](https://github.com/limcheekin/open-text-embeddings).