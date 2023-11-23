import uvicorn
import os
from src.server.app import create_app, load_model

if __name__ == "__main__":
    try:
        model_name = os.environ["MODEL"]
    except Exception as e:
        raise Exception("MODEL environment variable not set", e)
    
    device = os.getenv("DEVICE", "cpu")
    _reload = os.getenv("RELOAD", False)
    load_model(model_name, device, _reload)

    
    print("\nStarting server...")
    app = create_app()

    uvicorn.run(
        app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8000))
    )




