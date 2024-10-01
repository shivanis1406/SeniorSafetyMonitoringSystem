import gradio as gr
import io
import numpy as np
import torch
#from decord import cpu, VideoReader, bridge
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from orb_motion_detection import detect_fast_motion
import time, os

def process_video(video, start_time, end_time, quant=8):
    start = time.time()

    output_dir = "motion_detection_results"
    os.system(f"rm -rf {output_dir}")
    os.system(f"mkdir {output_dir}")

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    MODEL_PATH = "THUDM/cogvlm2-video-llama3-base"

    if 'int4' in MODEL_PATH:
        quant = 4

    strategy = 'base' if 'cogvlm2-video-llama3-base' in MODEL_PATH else 'chat'
    print(f"Using {strategy} model")

    timestamps, fast_frames = detect_fast_motion(video.name, output_dir, end_time, start_time, motion_threshold=1.5)

    history = []
    if len(fast_frames) > 0:
        video_data = np.array(fast_frames[0:min(48, len(fast_frames))])  # Shape: (num_frames, height, width, channels)
        video_data = np.transpose(video_data, (3, 0, 1, 2))  # RGB channels first
        video_tensor = torch.tensor(video_data)  # Convert to tensor

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

        if quant == 4:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=TORCH_TYPE,
                trust_remote_code=True,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=TORCH_TYPE,
                ),
                low_cpu_mem_usage=True
            ).eval()
        elif quant == 8:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=TORCH_TYPE,
                trust_remote_code=True,
                quantization_config=BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_4bit_compute_dtype=TORCH_TYPE,
                ),
                low_cpu_mem_usage=True
            ).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=TORCH_TYPE,
                trust_remote_code=True
            ).eval().to(DEVICE)

        query = "Describe the actions in the video frames focusing on physical abuse, violence, or someone falling down."
        print(f"Query: {query}")

        inputs = model.build_conversation_input_ids(
            tokenizer=tokenizer,
            query=query,
            images=[video_tensor],
            history=history,
            template_version=strategy
        )

        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[inputs['images'][0].to('cuda').to(TORCH_TYPE)]],
        }

        gen_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": 128002,
            "top_k": 1,
            "do_sample": True,
            "top_p": 0.1,
            "temperature": 0.1,
        }

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("\nCogVLM2-Video:", response)
        history.append((query, response))

        result = f"Response: {response}"
    else:
        result = "No aggressive behaviour found. Nobody falling down."

    end = time.time()
    execution_time = f"Execution time for {video.name}: {end - start} seconds. Duration of the video was {end_time - start_time} seconds."

    return result


# Create Gradio Interface
def gradio_interface():
    video_input = gr.File(label="Upload video file (.mp4)", type="filepath")
    start_time = gr.Number(value=0.0, label="Start time (seconds)")
    end_time = gr.Number(value=15.0, label="End time (seconds)")

    interface = gr.Interface(
        fn=process_video,
        inputs=[video_input, start_time, end_time],
        outputs="text",
        title="Senior Safety Monitoring System",
        description="Upload a video and specify the time range for analysis. The model will detect fast motion and describe actions such as physical abuse or someone falling down."
    )

    interface.launch(share=True)


if __name__ == "__main__":
    gradio_interface()