from challenge.api import app
import uvicorn
import os

application = app

if __name__ == "__main__":
    uvicorn.run(application, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
