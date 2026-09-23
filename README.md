# AquaMapKG MVP

## Local launch

```bash
source .venv/bin/activate
uv pip install -r requirements.txt
streamlit run app.py
```

The first prediction downloads the public
FathomNet marine trash model from Hugging Face.
The prototype stores findings only in the current
browser session and uses manually entered coordinates.
