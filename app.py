import streamlit as st
from google import genai
from google.genai import types
import wave
import io
import re
import time

st.set_page_config(page_title="Malayalam Text to Speech", page_icon="🔊")
st.title("🔊 Malayalam Text to Speech")
st.caption("Powered by Gemini API — Leda voice")

# --- API Key ---
# Streamlit Cloud-ൽ deploy ചെയ്യുമ്പോൾ Settings > Secrets ഇൽ GEMINI_API_KEY വെക്കുക.
# Local test ചെയ്യുമ്പോൾ box-ൽ key type ചെയ്യാം.
api_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
if not api_key:
    api_key = st.text_input("Gemini API Key", type="password", help="Google AI Studio-യിൽ നിന്ന് സൗജന്യമായി എടുക്കാം")

text = st.text_area("Malayalam Text ഇവിടെ paste ചെയ്യുക", height=300, placeholder="ഇവിടെ എത്ര വലിയ text ഉം ഇടാം...")

MAX_CHARS = 1500  # ഒരു API call-ൽ അയക്കുന്ന max characters (safe limit)

VOICE_NAME = "Leda"
MODEL_NAME = "gemini-2.5-flash-preview-tts"


def split_text(full_text, max_chars=MAX_CHARS):
    """വാക്യങ്ങൾ മുറിയാതെ text-നെ ചെറിയ chunks ആക്കുന്നു."""
    sentences = re.split(r'(?<=[.!?॥।])\s+|(?<=\n)\n+', full_text.strip())
    chunks = []
    current = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(current) + len(s) + 1 <= max_chars:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks


def generate_audio_chunk(client, text_chunk):
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=text_chunk,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE_NAME)
                )
            ),
        ),
    )
    return response.candidates[0].content.parts[0].inline_data.data


def pcm_to_wav_bytes(pcm_data, channels=1, rate=24000, sample_width=2):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


if st.button("🎙️ ശബ്ദം ഉണ്ടാക്കുക", type="primary"):
    if not api_key:
        st.error("ദയവായി API Key നൽകുക.")
    elif not text.strip():
        st.error("ദയവായി text ടൈപ്പ് ചെയ്യുക അല്ലെങ്കിൽ paste ചെയ്യുക.")
    else:
        client = genai.Client(api_key=api_key)
        chunks = split_text(text)
        st.info(f"Text {len(chunks)} ഭാഗങ്ങളായി (chunks) തിരിച്ചു. Audio generate ചെയ്യുന്നു...")

        # 300ms silence between chunks so words don't merge together
        silence_gap = b"\x00\x00" * int(24000 * 0.3)

        all_pcm = b""
        progress = st.progress(0)
        status = st.empty()
        error_occurred = False

        for i, chunk in enumerate(chunks):
            status.text(f"Chunk {i + 1}/{len(chunks)} generate ചെയ്യുന്നു...")
            try:
                pcm = generate_audio_chunk(client, chunk)
                all_pcm += pcm
                if i < len(chunks) - 1:
                    all_pcm += silence_gap
            except Exception as e:
                st.error(f"Chunk {i + 1} പരാജയപ്പെട്ടു: {e}")
                error_occurred = True
                break
            progress.progress((i + 1) / len(chunks))
            time.sleep(1)  # free-tier rate limit-ൽ തട്ടാതിരിക്കാൻ ചെറിയ delay

        if all_pcm and not error_occurred:
            wav_bytes = pcm_to_wav_bytes(all_pcm)
            st.success("Audio തയ്യാറായി!")
            st.audio(wav_bytes, format="audio/wav")
            st.download_button(
                "⬇️ Download WAV",
                data=wav_bytes,
                file_name="malayalam_speech.wav",
                mime="audio/wav",
            )

