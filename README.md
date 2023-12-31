# fast-embeddings-api

An OpenAI-like API for Massive Text Embeddings using FastAPI.

## How to run
Install the requirements:
```bash
pip install -r requirements.txt
```

Start the server by defining a model with **MODEL** variable. You can also add a **DEVICE** variable which can be `cuda` or `cpu`:
```bash
MODEL=BAAI/bge-base-en-v1.5 DEVICE=cuda python -m src.server
```

By default, it runs in `0.0.0.0:8000`. You can change this by defining variables **HOST** and/or **PORT**.

The first time you run the server it will create a `model_auto_opt_OX` folder, where `X=3` or `4` depending on the device. This folder contains the optimized ONNX version of your model. For the next runs, it will use that optimized model. If you want to regenerate the folder, you can use the variable **RELOAD=True**
## Docker image
You can run the application using Docker with the following command:
```bash
docker run -it --gpus all -p 8000:8000 -e MODEL=BAAI/bge-base-en-v1.5 -e DEVICE=cuda vokturz/fast-embeddings-api
```

## MTEB Benchmark
You can check the best embeddings model on https://huggingface.co/spaces/mteb/leaderboard

## Credits

Mainly based in [Optimum-Benchmark x MTEB by HuggingFace](https://github.com/huggingface/optimum-benchmark/tree/main/examples/fast-mteb) and [limcheekin/open-text-embeddings](https://github.com/limcheekin/open-text-embeddings).