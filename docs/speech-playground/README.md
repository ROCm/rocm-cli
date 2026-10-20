<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Speech API playground

A single-page reference for the speech endpoints `rocm serve` exposes, with a
"try it" form for each: type text, press **Speak**, and the browser plays the
reply.

```bash
rocm serve kokoro                              # or: rocm serve tts (GPU)
python3 -m http.server 8080 -d docs/speech-playground
# open http://localhost:8080/?server=http://127.0.0.1:11435/v1
```

The page talks to the server directly from the browser; the server allows
requests from `localhost` origins. Serve the page over HTTP rather than opening
the file, so the browser sends a `localhost` origin.

[openapi.yaml](openapi.yaml) describes the same endpoints for other OpenAPI
tools.
