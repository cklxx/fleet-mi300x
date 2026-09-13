# Vendored reference implementation

These files are **not mine**. They are copied verbatim from the model
repository so the correctness tests have a fixed golden implementation to
compare against, and so results do not change when the upstream repo does.

| File | Source |
|---|---|
| `modeling_deepseek.py` | [deepseek-ai/DeepSeek-Coder-V2-Lite-Base](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Base/blob/main/modeling_deepseek.py) |
| `configuration_deepseek.py` | [same repo](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Base/blob/main/configuration_deepseek.py) |
| `dsv2lite_config.json` | `config.json` from the same repo |

Licensed under the terms of that repository; see its `LICENSE` and
`LICENSE-MODEL`. Retained here only as a test oracle.

## Why vendored rather than loaded with `trust_remote_code`

`tests/test_reference_vs_hf.py` has to run without the 31 GB checkpoint, on a
tiny random config, so it imports these modules directly rather than going
through `AutoModel`. Pinning the copy also means a change upstream cannot
silently move the reference that every boundary tolerance in `docs/design.md`
§6 is stated against.

## Known incompatibility

The file targets transformers 4.x and imports `is_torch_fx_available`, removed
in transformers 5. Locally the tests stub that symbol; on the GPU box
`scripts/setup_env.sh` pins `transformers>=4.39,<5` instead, which is the
configuration results should be reproduced under.
