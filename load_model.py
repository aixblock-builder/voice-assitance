import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import builtins
from TTS.api import TTS

# Auto accept Coqui ToS
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

builtins.input = lambda _: "y"

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")


ASR_MODEL_ID = "openai/whisper-large-v3"
GEN_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"

def get_device_and_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "cuda", torch.bfloat16
        else:
            return "cuda", torch.float16
    else:
        return "cpu", torch.float32  # dùng float32 cho CPU để tránh lỗi dtype

device, torch_dtype = get_device_and_dtype()

# === LOAD ASR PIPELINE ===
asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    ASR_MODEL_ID,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
)

asr_processor = AutoProcessor.from_pretrained(ASR_MODEL_ID)

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=asr_model,
    tokenizer=asr_processor.tokenizer,
    feature_extractor=asr_processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=0 if device == "cuda" else -1,
)

# === LOAD TEXT GENERATION PIPELINE ===
def load_text_generation_model():
    print(f"Loading text generation model on {device.upper()}...")
    return pipeline(
        "text-generation",
        model=GEN_MODEL_ID,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else "cpu",
        max_new_tokens=256,
    )

text_gen_pipeline = load_text_generation_model()