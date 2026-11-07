# One image, two targets. `server` holds the world and runs the tool server (the compose service `tools`); the
# default target is the agent's container, which holds only the client: the `tools` CLI and its MCP door.
FROM python:3.12-slim AS base
COPY runtime.py /opt/task/runtime.py
RUN printf '#!/bin/sh\nexec python3 /opt/task/runtime.py tools "$@"\n' > /usr/local/bin/tools \
 && chmod 755 /usr/local/bin/tools

FROM base AS server
RUN pip install --no-cache-dir pydantic==2.13.4
COPY world/ /opt/task/world/
EXPOSE 8765
CMD ["python3", "/opt/task/runtime.py", "serve", "--world", "/opt/task/world", "--state", "/srv/state", \
     "--port", "8765"]

FROM base AS agent
ENV TOOLS_URL=http://tools:8765
WORKDIR /app
