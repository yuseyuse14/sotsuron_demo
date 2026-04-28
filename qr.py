from pathlib import Path
import qrcode

URL = "https://context-abbr.streamlit.app"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

img = qrcode.make(URL)
img.save(OUT_DIR / "qrcode.png")