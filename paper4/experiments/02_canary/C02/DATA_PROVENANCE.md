# RefCOCOg Data Provenance

- Dataset: `jxu124/refcocog`, split `train`
- Source: https://huggingface.co/datasets/jxu124/refcocog
- Annotation URL: `https://huggingface.co/datasets/jxu124/refcocog/resolve/main/data/train-00000-of-00001-4fe3e6340cfb69ed.parquet`
- Local path: `paper4/cache/refcocog/train.parquet`
- SHA-256: `6ab9631fb9353e94f440832f272ab15bae5726b62168e930a922aa687ed62b96`
- Schema sample verified via public datasets-server: `image_id`, `sentences`, `ann_id`, `ref_id`, `bbox`, `raw_image_info`, `category_id`.
- Records parsed: 5,000 real RefCOCOg train records through the public datasets-server API. The local parquet is retained for provenance; the parser used the returned records directly.
