# from load_model import load_model
# from load_model import load_model
# load_model()

import os
import uuid

import torch
from aixblock_ml.model import AIxBlockMLBase
import torch
import subprocess
import json
import threading
import requests
from loguru import logger
import numpy as np
from function_ml import connect_project, download_dataset, upload_checkpoint
from logging_class import start_queue, write_log
import time
from mcp.server.fastmcp import FastMCP
import zipfile
from huggingface_hub import (
    HfFolder, 
    login,
    whoami,
    ModelCard,
    upload_file,
    upload_folder,
    create_repo
)
from TTS.api import TTS
import io
from scipy.io.wavfile import write
import base64
import numpy as np
from tqdm import tqdm
import tarfile
import shutil
import yaml
import gradio as gr
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer, AutoModelForCausalLM
import torchaudio
import gc

hf_token = os.getenv("HF_TOKEN", "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")
HfFolder.save_token("hf_"+"bjIxyaTXDGqlUa"+"HjvuhcpfVkPjcvjitRsY")
login(token = "hf_"+"bjIxyaTXDGqlUa"+"HjvuhcpfVkPjcvjitRsY")

# Global variables for model caching
pipe_prediction = None
tokenizer = None
model_predict = None
speech_model = None
speech_processor = None
speech_pipe = None
current_speech_model_id = None
tts_model = None


hf_access_token = "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"
# login(token=hf_access_token)
CUDA_VISIBLE_DEVICES = []
for i in range(torch.cuda.device_count()):
    CUDA_VISIBLE_DEVICES.append(i)
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    f"{i}" for i in range(len(CUDA_VISIBLE_DEVICES))
)
print(os.environ["CUDA_VISIBLE_DEVICES"])


HOST_NAME = os.environ.get("HOST_NAME", "https://dev-us-west-1.aixblock.io")
TYPE_ENV = os.environ.get("TYPE_ENV", "DETECTION")


mcp = FastMCP("aixblock-mcp")

CHANNEL_STATUS = {}

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    dtype = torch.float16
else:
    device = torch.device("cpu")
    dtype = torch.float32

def load_speech_model(model_id):
    """Load speech-to-text model once and cache it globally"""
    global speech_model, speech_processor, speech_pipe, current_speech_model_id
    
    if current_speech_model_id == model_id and speech_pipe is not None:
        return speech_pipe
    
    logger.info(f"Loading speech model: {model_id}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    speech_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    speech_model.to(device)

    speech_processor = AutoProcessor.from_pretrained(model_id)

    speech_pipe = pipeline(
        "automatic-speech-recognition",
        model=speech_model,
        tokenizer=speech_processor.tokenizer,
        feature_extractor=speech_processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    current_speech_model_id = model_id
    return speech_pipe

def load_text_generation_model(model_id, token, local_dir="./data/checkpoint"):
    """Load text generation model once and cache it globally"""
    global pipe_prediction, tokenizer, model_predict
    
    if model_predict == model_id and pipe_prediction is not None:
        return pipe_prediction, tokenizer
    
    logger.info(f"Loading text generation model: {model_id}")
    gc.collect()
    torch.cuda.empty_cache()
    
    try:
        model_name = model_id.split("/")[-1]
        local_model_dir = os.path.join(local_dir, model_name)
        if os.path.exists(local_model_dir) and os.path.exists(
            os.path.join(local_model_dir, "config.json")
        ):
            print(f"✅ Loading model from local: {local_model_dir}")
            model_source = local_model_dir
        else:
            print(f"☁️ Loading model from HuggingFace Hub: {model_id}")
            model_source = model_id
    except Exception as e:
        print(e)
        print(f"☁️ Loading model from HuggingFace Hub: {model_id}")
        model_source = model_id

    print("model_source", model_source)
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

        print("Using CUDA.")
        pipe_prediction = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=dtype,
            device_map="auto"
        )
    else:
        print("Using CPU.")
        pipe_prediction = AutoModelForCausalLM.from_pretrained(
            model_source,
            device_map="cpu",
        )
    
    model_predict = model_id
    return pipe_prediction, tokenizer

def load_tts_model():
    """Load TTS model once and cache it globally"""
    global tts_model
    
    if tts_model is not None:
        return tts_model
    
    logger.info("Loading TTS model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True).to(device)
    return tts_model

class MyModel(AIxBlockMLBase):
    @mcp.tool()
    def action(self, command, **kwargs):
        logger.info(f"Received command: {command} with args: {kwargs}")
        if command.lower() == "train":
            try:
                model_id = kwargs.get("model_id", "meta-llama/Llama-3.2-1B-Instruct")
                dataset_id = kwargs.get(
                    "dataset_id", "autoprogrammer/Qwen2.5-Coder-7B-Instruct-codeguardplus"
                )

                push_to_hub = kwargs.get("push_to_hub", True)
                hf_model_id = kwargs.get(
                    "hf_model_id", "meta-llama/Llama-3.2-1B-Instruct"
                )
                push_to_hub_token = kwargs.get(
                    "push_to_hub_token", "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"
                )
                framework = kwargs.get("framework", "huggingface")
                task = kwargs.get("task", "text-generation")
                prompt = kwargs.get("prompt", "")
                trainingArguments = kwargs.get("TrainingArguments", None)
                cuda_debug = kwargs.get("cuda_debug", False)

                json_file = "training_args.json"
                absolute_path = os.path.abspath(json_file)

                with open(absolute_path, "w") as f:
                    json.dump(trainingArguments, f)
                logger.info(f"Training arguments: {trainingArguments}")

                if cuda_debug == True:
                    os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
                    os.environ["NCCL_DEBUG"] = "INFO"

                os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
                os.environ["TORCH_USE_CUDA_DSA"] = "0"
                clone_dir = os.path.join(os.getcwd())
                project_id = kwargs.get("project_id", 0)
                token = kwargs.get("token", "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")
                checkpoint_version = kwargs.get("checkpoint_version")
                checkpoint_id = kwargs.get("checkpoint")
                dataset_version = kwargs.get("dataset_version")
                dataset = kwargs.get("dataset")
                channel_log = kwargs.get("channel_log", "training_logs")
                world_size = kwargs.get("world_size", 1)
                rank = kwargs.get("rank", 0)
                master_add = kwargs.get("master_add", "127.0.0.1")
                master_port = kwargs.get("master_port", "23456")
                host_name = kwargs.get("host_name", HOST_NAME)
                instruction_field = kwargs.get("prompt_field", "prompt")
                input_field = kwargs.get("input_field", "task_description")
                output_field = kwargs.get("output_field", "response")
                log_queue, logging_thread = start_queue(channel_log)
                epoch = kwargs.get("epoch", 5)
                batch_size = kwargs.get("batch_size", 1)
                write_log(log_queue)
                channel_name = f"{hf_model_id}_{str(uuid.uuid4())[:8]}"
                username = ""
                hf_model_name = ""

                try:
                    user = whoami(token=push_to_hub_token)['name']
                    hf_model_name = f"{user}/{hf_model_id}"
                except Exception as e:
                    hf_model_name = "Token not correct"
                    print(e)
                    
                CHANNEL_STATUS[channel_name] = {
                    "status": "training",
                    "hf_model_id": hf_model_name,
                    "command": command,
                    "created_at": time.time(),
                }
                print(f"🚀 Đã bắt đầu training kênh: {channel_name}")

                def func_train_model(
                    clone_dir,
                    project_id,
                    token,
                    checkpoint_version,
                    checkpoint_id,
                    dataset_version,
                    dataset_id,
                    model_id,
                    world_size,
                    rank,
                    master_add,
                    master_port,
                    prompt,
                    json_file,
                    channel_log,
                    hf_model_id,
                    push_to_hub,
                    push_to_hub_token,
                    host_name,
                    epoch,
                    batch_size
                ):

                    dataset_path = None
                    project = connect_project(host_name, token, project_id)

                    if dataset_version and dataset_id and project:
                        dataset_path = os.path.join(
                            clone_dir, f"datasets/{dataset_version}"
                        )

                        if not os.path.exists(dataset_path):
                            data_path = os.path.join(clone_dir, "data_zip")
                            os.makedirs(data_path, exist_ok=True)

                            dataset_name = download_dataset(project, dataset_id, data_path)
                            print(dataset_name)
                            if dataset_name:
                                data_zip_dir = os.path.join(data_path, dataset_name)

                                with zipfile.ZipFile(data_zip_dir, "r") as zip_ref:
                                    zip_ref.extractall(dataset_path)

                                extracted_files = os.listdir(dataset_path)
                                zip_files = [
                                    f for f in extracted_files if f.endswith(".zip")
                                ]

                                if len(zip_files) == 1:
                                    inner_zip_path = os.path.join(
                                        dataset_path, zip_files[0]
                                    )
                                    print(
                                        f"🔁 Found inner zip file: {inner_zip_path}, extracting..."
                                    )
                                    with zipfile.ZipFile(inner_zip_path, "r") as inner_zip:
                                        inner_zip.extractall(dataset_path)
                                    os.remove(inner_zip_path)

                        train_dir = os.path.join(dataset_path, "train/train_manifest.json")
                        validation_dir = os.path.join(dataset_path, "validation/validation_manifest.json")
                    
                    else:
                        url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
                        filename = "LJSpeech-1.1.tar.bz2"

                        response = requests.get(url, stream=True)
                        total_size = int(response.headers.get('content-length', 0))
                        block_size = 1024  # 1 Kibibyte

                        with open(filename, 'wb') as file, tqdm(
                            desc=filename,
                            total=total_size,
                            unit='iB',
                            unit_scale=True,
                            unit_divisor=1024,
                        ) as bar:
                            for data in response.iter_content(block_size):
                                file.write(data)
                                bar.update(len(data))
                        
                        with tarfile.open("LJSpeech-1.1.tar.bz2", "r:bz2") as tar:
                            tar.extractall("LJSpeech-1.1")
                        
                        target_dir = "Data"
                        wavs_src = os.path.join("LJSpeech-1.1", "LJSpeech-1.1", "wavs")
                        wavs_dst = os.path.join(target_dir, "wavs")

                        os.makedirs(target_dir, exist_ok=True)

                        # Di chuyển thư mục wavs
                        if os.path.exists(wavs_dst):
                            shutil.rmtree(wavs_dst)
                        
                        logger.info(f"Đang di chuyển thư mục wavs từ {wavs_src} đến {wavs_dst}")
                        shutil.move(wavs_src, wavs_dst)
                        logger.info(f"Đã di chuyển thư mục wavs từ {wavs_src} đến {wavs_dst}")


                    make_dir = os.path.join(os.getcwd(), "checkpoint")
                    os.makedirs(make_dir, exist_ok=True)
                    # subprocess.run(
                    #     ("whereis accelerate"),
                    #     shell=True,
                    # )
                    config_path = "Configs/config_ft.yml"
                    config = yaml.safe_load(open(config_path))

                    logger.info("===Train===")

                    config['batch_size'] = batch_size
                    config['max_len'] = 100 # not enough RAM
                    config['loss_params']['joint_epoch'] = 110 
                    config['epochs'] = epoch

                    logger.info(f"Config: {config}")
                    try:
                        subprocess.run([
                            "venv/bin/python", "train_finetune.py",
                            "--config_path", "Configs/config_ft.yml"
                        ], check=True)
                    except Exception as e:
                        logger.error(f"Lỗi khi train model: {e}")

                    user = whoami(token=push_to_hub_token)['name']
                    repo_id = f"{user}/{hf_model_id}"

                    # Nếu repo chưa tồn tại, tạo mới
                    create_repo(repo_id=repo_id, token=push_to_hub_token, exist_ok=True)

                    # ✅ Upload checkpoint (đặt tên chuẩn theo Transformers nếu có thể)
                    upload_folder(
                        folder_path=make_dir,
                        path_in_repo="checkpoint",
                        repo_id=repo_id,
                        token=push_to_hub_token,
                        commit_message="Upload fine-tuned checkpoint"
                    )

                    yaml_metadata = (
                        "---\n"
                        "license: apache-2.0\n"
                        "language: en\n"
                        "tags:\n"
                        "  - speech\n"
                        "  - translation\n"
                        f"model_name: {hf_model_id}\n"
                        "---\n\n"
                    )

                    # Nội dung phần mô tả
                    description = (
                        "# Model Overview\n\n"
                        "This model was trained/fine-tuned for speech translation tasks. "
                        "It is based on a pretrained backbone and optimized using custom datasets.\n\n"
                    )

                    # Nội dung phần trích dẫn
                    citations = (
                        "## Citations\n\n"
                        "This model was fine-tuned using custom data and training scripts.\n\n"
                        "© 2025 YourTeamName. All rights reserved.\n"
                    )

                    # Tạo nội dung README
                    readme_content = yaml_metadata + description + citations

                    # Ghi ra file README.md
                    readme_path = "README.md"
                    with open(readme_path, "w", encoding="utf-8") as f:
                        f.write(readme_content)

                    # Upload lên Hugging Face Hub
                    upload_file(
                        path_or_fileobj=readme_path,
                        path_in_repo="README.md",
                        repo_id=repo_id,
                        token=push_to_hub_token,
                        commit_message="Upload new README.md"
                    )

                    print("✅ Đã upload README.md và checkpoint lên Hugging Face Hub!")

                    CHANNEL_STATUS[channel_name]["status"] = "done"
                    output_dir = "./data/checkpoint"
                    print(push_to_hub)
                    if push_to_hub:
                        import datetime

                        output_dir = "./data/checkpoint"
                        now = datetime.datetime.now()
                        date_str = now.strftime("%Y%m%d")
                        time_str = now.strftime("%H%M%S")
                        version = f"{date_str}-{time_str}"

                        upload_checkpoint(project, version, output_dir)

                train_thread = threading.Thread(
                    target=func_train_model,
                    args=(
                        clone_dir,
                        project_id,
                        token,
                        checkpoint_version,
                        checkpoint_id,
                        dataset_version,
                        dataset_id,
                        model_id,
                        world_size,
                        rank,
                        master_add,
                        master_port,
                        prompt,
                        absolute_path,
                        channel_log,
                        hf_model_id,
                        push_to_hub,
                        push_to_hub_token,
                        host_name,
                        epoch,
                        batch_size
                    ),
                )
                train_thread.start()

                return {
                    "message": "train completed successfully",
                    "channel_name": channel_name,
                }
            except Exception as e:
                return {"message": f"train failed: {e}"}

        elif command.lower() == "tensorboard":
            def run_tensorboard():
                # train_dir = os.path.join(os.getcwd(), "{project_id}")
                # log_dir = os.path.join(os.getcwd(), "logs")
                p = subprocess.Popen(f"tensorboard --logdir ./logs --host 0.0.0.0 --port=6006", stdout=subprocess.PIPE, stderr=None, shell=True)
                out = p.communicate()
                print(out)

            tensorboard_thread = threading.Thread(target=run_tensorboard)
            tensorboard_thread.start()
            return {"message": "tensorboardx started successfully"}
        
        # elif command.lower() == "dashboard":
        #     link = promethus_grafana.generate_link_public("ml_00")
        #     return {"Share_url": link}
          
        elif command.lower() == "predict":
            prompt = kwargs.get("prompt", "")
            model_id = kwargs.get("model_id", "Qwen/Qwen3-1.7B")
            token = kwargs.get("token")
            raw_input = kwargs.get("input", None)

            print(raw_input)

            def is_base64(s):
                try:
                    # Kiểm tra nếu có tiền tố "data:audio", loại bỏ phần đầu
                    if s.startswith("data:audio"):
                        s = s.split(",")[1]
                    # Thử decode base64
                    base64.b64decode(s, validate=True)
                    return True
                except Exception:
                    return False

            def handle_audio_input(audio_input, save_dir="audios"):
                os.makedirs(save_dir, exist_ok=True)
                
                if is_base64(audio_input):
                    # Nếu là base64
                    if audio_input.startswith("data:audio"):
                        audio_input = audio_input.split(",")[1]
                    audio_bytes = base64.b64decode(audio_input)
                    audio_path = os.path.join(save_dir, "input_audio.wav")
                    with open(audio_path, "wb") as f:
                        f.write(audio_bytes)
                else:
                    # Nếu là URL
                    response = requests.get(audio_input)
                    audio_path = os.path.join(save_dir, "input_audio.wav")
                    with open(audio_path, "wb") as f:
                        f.write(response.content)

                return audio_path

            def decode_base64_to_audio(base64_audio, output_file="output.wav"):
                if "," in base64_audio:
                    base64_audio = base64_audio.split(",", 1)[1]

                missing_padding = len(base64_audio) % 4
                if missing_padding:
                    base64_audio += "=" * (4 - missing_padding)

                file_path = os.path.join(os.path.dirname(__file__), output_file)

                audio_data = base64.b64decode(base64_audio)
                with open(file_path, "wb") as audio_file:
                    audio_file.write(audio_data)

                return file_path

            def download_audio(audio_url, save_path):
                # Tạo request để tải video từ URL
                response = requests.get(audio_url, stream=True)
                
                # Kiểm tra nếu request thành công
                if response.status_code == 200:
                    with open(save_path, 'wb') as audio_file:
                        for chunk in response.iter_content(chunk_size=1024):
                            if chunk:
                                audio_file.write(chunk)
                    print(f"audio has been downloaded and saved to {save_path}")
                    return save_path  # Trả về đường dẫn đến video đã tải về
                else:
                    print(f"Failed to download audio. Status code: {response.status_code}")
                    return None
            
            input_datas = json.loads(raw_input)
            print(input_datas)

            if not input_datas:
                input_datas = {
                    "name": "Result",
                    "data": prompt
                }

            # Load speech-to-text model
            pipe = load_speech_model(model_id="openai/whisper-large-v3")

            predictions = []
            input_data = input_datas[0]
            print(input_data)
            question = ""

            try:
                audio_path = handle_audio_input(input_data["data"])
                print("Audio saved at:", audio_path)

                waveform, sample_rate = torchaudio.load(audio_path)

                # Mono hóa nếu cần
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                sample = {
                    "array": waveform.squeeze().numpy(),
                    "sampling_rate": sample_rate
                }

                result = pipe(sample, return_timestamps=True)

                question = result["text"]

                print("question", question)

            except Exception as e:
                question = "I don't understand your question"
                print(e)

            # Load text generation model
            with torch.no_grad():
                pipe_prediction, tokenizer = load_text_generation_model(model_id, hf_access_token)

            messages = [
                {
                    "role": "system",
                    "content": "Respond with a concise list or bullet points. Do not write full paragraphs."
                },
                {
                    "role": "user",
                    "content": str(question)
                }
            ]

            print("messages", messages)

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(pipe_prediction.device)

            # conduct text completion
            generated_ids = pipe_prediction.generate(
                **model_inputs,
                max_new_tokens=32768
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

            # parsing thinking content
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            # thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            prompt = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

            print("prompt", prompt)
            
            # Load TTS model
            tts = load_tts_model()

            predictions = []
            result = []
            generated_url=""

            if audio_path: 
                print("===input_audio", audio_path)
            
                tts.tts_to_file(text=prompt,
                    file_path = "output.wav",
                    speaker_wav=audio_path,
                    language="en"
                )

                print("===file_path", "output.wav")

                with open("output.wav", "rb") as f:
                    audio_base64 = base64.b64encode(f.read()).decode("utf-8") 

                result.append({
                    "data": f"data:audio/wav;base64,{audio_base64}"
                })
                generated_url = f"/static/{os.path.basename('output.wav')}"
                print(result)

            else:
                tts.tts_to_file(text=prompt,
                    file_path="output.wav",
                    speaker_wav="male.wav",
                    language="en")
                
                print("===wav", "output.wav")

                with open("output.wav", "rb") as f:
                    audio_base64 = base64.b64encode(f.read()).decode("utf-8")

                generated_url = f"/static/{os.path.basename('output.wav')}"
                
                result.append({
                    "data": f"data:audio/wav;base64,{audio_base64}"
                })

                print(result)

            predictions.append({
                'result': [{
                    'from_name': "generated_text",
                    'to_name': "text_output", #audio
                    'type': 'textarea',
                    'value': {
                        'data': result,
                        "url": generated_url, 
                        'text': result
                    }
                }],
                'model_version': ""
            })
            print("predictions", predictions)
            return {"message": "predict completed successfully", "result": predictions}

        
        elif command.lower() == "prompt_sample":
                task = kwargs.get("task", "")
                if task == "question-answering":
                    prompt_text = f"""
                   Here is the context: 
                    {{context}}

                    Based on the above context, provide an answer to the following question using only a single word or phrase from the context without repeating the question or adding any extra explanation: 
                    {{question}}

                    Answer:
                    """
                elif task == "text-classification":
                   prompt_text = f"""
                    Summarize the following text into a single, concise paragraph focusing on the key ideas and important points:

                    Text: 
                    {{context}}

                    Summary:
                    """
                
                elif task == "summarization":
                    prompt_text = f"""
                    Summarize the following text into a single, concise paragraph focusing on the key ideas and important points:

                    Text: 
                    {{context}}

                    Summary:
                    """
                return {"message": "prompt_sample completed successfully", "result":prompt_text}
        
        elif command.lower() == "stop":
            subprocess.run(["pkill", "-9", "-f", "llama_recipes/finetuning.py"])
            return {"message": "Done", "result": "Done"}
        
        elif command.lower() == "action-example":
            return {"message": "Done", "result": "Done"}
        
        elif command == "status":
            channel = kwargs.get("channel", None)

            if channel:
                # Nếu có truyền kênh cụ thể
                status_info = CHANNEL_STATUS.get(channel)
                if status_info is None:
                    return {"channel": channel, "status": "not_found"}
                elif isinstance(status_info, dict):
                    return {"channel": channel, **status_info}
                else:
                    return {"channel": channel, "status": status_info}
            else:
                # Lấy tất cả kênh
                if not CHANNEL_STATUS:
                    return {"message": "No channels available"}

                channels = []
                for ch, info in CHANNEL_STATUS.items():
                    if isinstance(info, dict):
                        channels.append({"channel": ch, **info})
                    else:
                        channels.append({"channel": ch, "status": info})

                return {"channels": channels}
        else:
            return {"message": "command not supported", "result": None}
        
    @mcp.tool()
    def model(self, **kwargs):
        global model_demo, tokenizer_demo, model_loaded_demo, model_id_demo

        model_id_demo = kwargs.get("model_id", "google/gemma-3-4b-it")
        project_id = kwargs.get("project_id", 0)

        print(
            f"""\
        Project ID: {project_id}
        """
        )
        
        MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

        DESCRIPTION = """\
        # Gemma-3
        """

        if not torch.cuda.is_available():
            DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"
        if torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16

        with gr.Blocks(css="style.css") as demo:
            gr.Markdown(DESCRIPTION)
            with gr.Row():
                load_btn = gr.Button("Load Model")
                status_text = gr.Textbox(label="Model Status", interactive=False)


        gradio_app, local_url, share_url = demo.launch(
            share=True,
            quiet=True,
            prevent_thread_lock=True,
            server_name="0.0.0.0",
            show_error=True,
        )
        return {"share_url": share_url, "local_url": local_url}
    
    @mcp.tool()
    def model_trial(self, project, **kwargs):
        import gradio as gr 
        return {"message": "Done", "result": "Done"}


        css = """
        .feedback .tab-nav {
            justify-content: center;
        }

        .feedback button.selected{
            background-color:rgb(115,0,254); !important;
            color: #ffff !important;
        }

        .feedback button{
            font-size: 16px !important;
            color: black !important;
            border-radius: 12px !important;
            display: block !important;
            margin-right: 17px !important;
            border: 1px solid var(--border-color-primary);
        }

        .feedback div {
            border: none !important;
            justify-content: center;
            margin-bottom: 5px;
        }

        .feedback .panel{
            background: none !important;
        }


        .feedback .unpadded_box{
            border-style: groove !important;
            width: 500px;
            height: 345px;
            margin: auto;
        }

        .feedback .secondary{
            background: rgb(225,0,170);
            color: #ffff !important;
        }

        .feedback .primary{
            background: rgb(115,0,254);
            color: #ffff !important;
        }

        .upload_image button{
            border: 1px var(--border-color-primary) !important;
        }
        .upload_image {
            align-items: center !important;
            justify-content: center !important;
            border-style: dashed !important;
            width: 500px;
            height: 345px;
            padding: 10px 10px 10px 10px
        }
        .upload_image .wrap{
            align-items: center !important;
            justify-content: center !important;
            border-style: dashed !important;
            width: 500px;
            height: 345px;
            padding: 10px 10px 10px 10px
        }

        .webcam_style .wrap{
            border: none !important;
            align-items: center !important;
            justify-content: center !important;
            height: 345px;
        }

        .webcam_style .feedback button{
            border: none !important;
            height: 345px;
        }

        .webcam_style .unpadded_box {
            all: unset !important;
        }

        .btn-custom {
            background: rgb(0,0,0) !important;
            color: #ffff !important;
            width: 200px;
        }

        .title1 {
            margin-right: 90px !important;
        }

        .title1 block{
            margin-right: 90px !important;
        }

        """

        with gr.Blocks(css=css) as demo:
            with gr.Row():
                with gr.Column(scale=10):
                    gr.Markdown(
                        """
                        # Theme preview: `AIxBlock`
                        """
                    )

           
            def predict(input_img):
            
                # result = self.action(project, "predict",collection="",data={"img":input_img})
                # print(result)
                # if result['result']:
                #     boxes = result['result']['boxes']
                #     names = result['result']['names']
                #     labels = result['result']['labels']
                    
                #     for box, label in zip(boxes, labels):
                #         box = [int(i) for i in box]
                #         label = int(label)
                #         input_img = cv2.rectangle(input_img, box, color=(255, 0, 0), thickness=2)
                #         # input_img = cv2.(input_img, names[label], (box[0], box[1]), color=(255, 0, 0), size=1)
                #         input_img = cv2.putText(input_img, names[label], (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                return input_img
            
            def download_btn(evt: gr.SelectData):
                print(f"Downloading {dataset_choosen}")
                return f'<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"><a href="/my_ml_backend/datasets/{evt.value}" style="font-size:50px"> <i class="fa fa-download"></i> Download this dataset</a>'
                
            def trial_training(dataset_choosen):
                print(f"Training with {dataset_choosen}")
                result = self.action(project, "train",collection="",data=dataset_choosen)
                return result['message']

            def get_checkpoint_list(project):
                print("GETTING CHECKPOINT LIST")
                print(f"Proejct: {project}")
                # import os
                checkpoint_list = [i for i in os.listdir("my_ml_backend/models") if i.endswith(".pt")]
                checkpoint_list = [f"<a href='./my_ml_backend/checkpoints/{i}' download>{i}</a>" for i in checkpoint_list]
                if os.path.exists(f"my_ml_backend/{project}"):
                    for folder in os.listdir(f"my_ml_backend/{project}"):
                        if "train" in folder:
                            project_checkpoint_list = [i for i in os.listdir(f"my_ml_backend/{project}/{folder}/weights") if i.endswith(".pt")]
                            project_checkpoint_list = [f"<a href='./my_ml_backend/{project}/{folder}/weights/{i}' download>{folder}-{i}</a>" for i in project_checkpoint_list]
                            checkpoint_list.extend(project_checkpoint_list)
                
                return "<br>".join(checkpoint_list)

            def tab_changed(tab):
                if tab == "Download":
                    get_checkpoint_list(project=project)
            
            def upload_file(file):
                return "File uploaded!"
            
            with gr.Tabs(elem_classes=["feedback"]) as parent_tabs:
                with gr.TabItem("Image", id=0):   
                    with gr.Row():
                        gr.Markdown("## Input", elem_classes=["title1"])
                        gr.Markdown("## Output", elem_classes=["title1"])
                    
                    gr.Interface(predict, gr.Image(elem_classes=["upload_image"], sources="upload", container = False, height = 345,show_label = False), 
                                gr.Image(elem_classes=["upload_image"],container = False, height = 345,show_label = False), allow_flagging = False             
                    )


                # with gr.TabItem("Webcam", id=1):    
                #     gr.Image(elem_classes=["webcam_style"], sources="webcam", container = False, show_label = False, height = 450)

                # with gr.TabItem("Video", id=2):    
                #     gr.Image(elem_classes=["upload_image"], sources="clipboard", height = 345,container = False, show_label = False)

                # with gr.TabItem("About", id=3):  
                #     gr.Label("About Page")

                with gr.TabItem("Trial Train", id=2):
                    gr.Markdown("# Trial Train")
                    with gr.Column():
                        with gr.Column():
                            gr.Markdown("## Dataset template to prepare your own and initiate training")
                            with gr.Row():
                                #get all filename in datasets folder
                                if not os.path.exists(f"./datasets"):
                                    os.makedirs(f"./datasets")

                                datasets = [(f"dataset{i}", name) for i, name in enumerate(os.listdir('./datasets'))]
                                
                                dataset_choosen = gr.Dropdown(datasets, label="Choose dataset", show_label=False, interactive=True, type="value")
                                # gr.Button("Download this dataset", variant="primary").click(download_btn, dataset_choosen, gr.HTML())
                                download_link = gr.HTML("""
                                        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
                                        <a href='' style="font-size:24px"><i class="fa fa-download" ></i> Download this dataset</a>""")
                                
                                dataset_choosen.select(download_btn, None, download_link)
                                
                                #when the button is clicked, download the dataset from dropdown
                                # download_btn
                            gr.Markdown("## Upload your sample dataset to have a trial training")
                            # gr.File(file_types=['tar','zip'])
                            gr.Interface(predict, gr.File(elem_classes=["upload_image"],file_types=['tar','zip']), 
                                gr.Label(elem_classes=["upload_image"],container = False), allow_flagging = False             
                    )
                            with gr.Row():
                                gr.Markdown(f"## You can attemp up to {2} FLOps")
                                gr.Button("Trial Train", variant="primary").click(trial_training, dataset_choosen, None)
                
                # with gr.TabItem("Download"):
                #     with gr.Column():
                #         gr.Markdown("## Download")
                #         with gr.Column():
                #             gr.HTML(get_checkpoint_list(project))

        gradio_app, local_url, share_url = demo.launch(share=True, quiet=True, prevent_thread_lock=True, server_name='0.0.0.0',show_error=True)
   
        return {"share_url": share_url, 'local_url': local_url}
    
    @mcp.tool()
    def download(self, project, **kwargs):
        from flask import send_from_directory,request
        file_path = request.args.get('path')
        print(request.args)
        return send_from_directory(os.getcwd(), file_path)
