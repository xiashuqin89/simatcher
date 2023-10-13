import uvicorn

from simatcher.cli.api import fastapp


uvicorn.run(fastapp, host="0.0.0.0", port=8030)
