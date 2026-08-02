# GPT-OSS long-prompt candidates

Token counts are exact `o200k_harmony` counts produced by OpenAI's
`openai-harmony` renderer. All prompts use GPT-OSS native Harmony formatting,
reasoning effort `high`, conversation date `2026-08-01`, and the default
knowledge cutoff and channel configuration from `openai-harmony==0.0.8`.

## Results

- `gpt_oss_120b_9k_11k/`: LongBench-Chat scan. None of its 50 prompts fell
  within 9,000-11,000 tokens. The nearest prompt was 11,447 tokens.
- `gpt_oss_120b_9k_11k_leval/`: L-Eval scan. Eight prompts fell within the
  target window. The five candidates with the longest reference outputs are
  marked `selected_for_pilot: true`.

Each selected case contains its source record, logical messages, fully rendered
Harmony prompt, metadata, reference output, and exact token count.

## Reproduce

```powershell
python -m pip install -r requirements-tokenizer.txt
python scripts/filter_longbench_chat.py --replace-output
python scripts/filter_leval.py --replace-output
```

Keep the reasoning effort and conversation date identical when running remote
inference; changing either changes the rendered input and may change its count.
