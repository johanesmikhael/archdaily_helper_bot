﻿# archdaily_helper_bot

This is a telegram bot designed for summarizing articles in Archdaily. However, it can be also used for many webpages such as Wikipedia.
You need to get token for your telegram bot, openai and huggingface API to run the repository (create your own keys.py and store the API keys).

Telegram bot: https://t.me/dobido_bot

```bash
# create virtual environment with conda
conda create -n summarizer python=3.11

# activate virtual environment
conda activate summarizer

# install dependencies
(summarizer) pip install -r requirements.txt
```

The project is mainly inspired from Lauzhack mini-hackathon held on 11th November 2023 at EPFL.
https://github.com/LauzHack/apis-telegram
