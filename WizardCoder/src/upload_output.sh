#!/bin/bash

ossutil64 cp -r /workspace/storage/output/ oss://aly-de-fra-001/yr-data/human-eval/storage-now/output/ \
  --access-key-id="${ALY_STS_ID}" \
  --access-key-secret="${ALY_STS_SECRET}" \
  --sts-token="${ALY_STS_TOKEN}" \
  --endpoint="${ALY_ENDPOINT}";
