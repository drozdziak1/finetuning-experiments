FROM rocm/pytorch:latest

RUN pip install -q pandas peft==0.9.0 transformers==4.31.0 trl==0.4.7 accelerate scipy tensorboardX

# RUN pip install -q https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-0.45.1.dev0-py3-none-manylinux_2_24_x86_64.whl

RUN pip install https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-0.45.3.dev271-py3-none-manylinux_2_24_x86_64.whl

ADD ./precache_models.py .

RUN --mount=type=cache,id=model-cache,target=/root/.cache/huggingface \
    python ./precache_models.py && \
    cp -r /root/.cache/huggingface saved_cache

RUN mv saved_cache /root/.cache/huggingface

RUN mkdir /src
