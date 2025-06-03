import os
import requests
from tqdm import tqdm
import torch
import numpy as np
import laion_clap
from clap_module.factory import load_state_dict
import librosa
import pyloudnorm as pyln

# following documentation from https://github.com/LAION-AI/CLAP
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)


def clap_score_simple(prompts, audio_files, clap_model='630k-audioset-fusion-best.pt'):
    """
    Simplified CLAP score calculation for your own generated audio.
    
    Params:
    -- prompts: list of text prompts used to generate the audio
    -- audio_files: list of audio file paths (same length as prompts)
    -- clap_model: CLAP model to use
    
    Returns:
    -- Average CLAP score
    """
    if len(prompts) != len(audio_files):
        raise ValueError("Number of prompts must match number of audio files")
    
    # Create id2text dictionary using indices
    id2text = {i: prompt for i, prompt in enumerate(prompts)}
    
    # Load model (same as original function)
    if clap_model == 'music_speech_audioset_epoch_15_esc_89.98.pt':
        url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt'
        clap_path = 'load/clap_score/music_speech_audioset_epoch_15_esc_89.98.pt'
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base',  device='cuda')
    elif clap_model == 'music_audioset_epoch_15_esc_90.14.pt':
        url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt'
        clap_path = 'load/clap_score/music_audioset_epoch_15_esc_90.14.pt'
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base',  device='cuda')
    elif clap_model == 'music_speech_epoch_15_esc_89.25.pt':
        url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_epoch_15_esc_89.25.pt'
        clap_path = 'load/clap_score/music_speech_epoch_15_esc_89.25.pt'
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base',  device='cuda')
    elif clap_model == '630k-audioset-fusion-best.pt':
        url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt'
        clap_path = 'load/clap_score/630k-audioset-fusion-best.pt'
        model = laion_clap.CLAP_Module(enable_fusion=True, device='cuda')
    else:
        raise ValueError('clap_model not implemented')

    # Download model if needed
    if not os.path.exists(clap_path):
        print('Downloading ', clap_model, '...')
        os.makedirs(os.path.dirname(clap_path), exist_ok=True)

        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(clap_path, 'wb') as file:
            with tqdm(total=total_size, unit='B', unit_scale=True) as progress_bar:
                for data in response.iter_content(chunk_size=8192):
                    file.write(data)
                    progress_bar.update(len(data))

    # Load model
    pkg = load_state_dict(clap_path)
    pkg.pop('text_branch.embeddings.position_ids', None)
    model.model.load_state_dict(pkg)
    model.eval()

    # Extract text embeddings
    print('[EXTRACTING TEXT EMBEDDINGS]')
    batch_size = 64
    text_emb = {}
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_ids = list(range(i, min(i + batch_size, len(prompts))))
        batch_texts = [prompts[j] for j in batch_ids]
        with torch.no_grad():
            embeddings = model.get_text_embedding(batch_texts, use_tensor=True)
        for idx, emb in zip(batch_ids, embeddings):
            text_emb[idx] = emb

    # Calculate scores
    print('[EVALUATING GENERATIONS]')
    score = 0
    count = 0
    
    for i in tqdm(range(len(audio_files))):
        audio_file = audio_files[i]
        if not os.path.exists(audio_file):
            print(f"Warning: Audio file {audio_file} does not exist, skipping...")
            continue
            
        with torch.no_grad():
            audio, _ = librosa.load(audio_file, sr=48000, mono=True)
            audio = pyln.normalize.peak(audio, -1.0)
            audio = audio.reshape(1, -1)
            audio = torch.from_numpy(int16_to_float32(float32_to_int16(audio))).float()
            audio_embeddings = model.get_audio_embedding_from_data(x=audio, use_tensor=True)
        
        cosine_sim = torch.nn.functional.cosine_similarity(
            audio_embeddings, text_emb[i].unsqueeze(0), dim=1, eps=1e-8)[0]
        score += cosine_sim
        count += 1

    return score / count if count > 0 else 0


def clap_score_from_folder(audio_folder, prompt_text_mapping, audio_extension='.wav', clap_model='630k-audioset-fusion-best.pt'):
    """
    Alternative approach: Calculate CLAP score from a folder of audio files.
    
    Params:
    -- audio_folder: folder containing audio files
    -- prompt_text_mapping: dictionary mapping filename (without extension) to prompt text
    -- audio_extension: file extension for audio files
    -- clap_model: CLAP model to use
    
    Returns:
    -- Average CLAP score
    """
    # Get all audio files in folder
    audio_files = [f for f in os.listdir(audio_folder) if f.endswith(audio_extension)]
    
    prompts = []
    full_audio_paths = []
    
    for audio_file in audio_files:
        filename_without_ext = os.path.splitext(audio_file)[0]
        if filename_without_ext in prompt_text_mapping:
            prompts.append(prompt_text_mapping[filename_without_ext])
            full_audio_paths.append(os.path.join(audio_folder, audio_file))
        else:
            print(f"Warning: No prompt found for {audio_file}")
    
    if not prompts:
        raise ValueError("No matching prompts found for audio files")
    
    return clap_score_simple(prompts, full_audio_paths, clap_model)


if __name__ == "__main__":
    # Example usage 1: Direct lists
    prompts = [
        "a piano melody with soft drums",
        "rock music with electric guitar solo",
        "ambient electronic music"
    ]
    
    audio_files = [
        "generated_audio/sample_001.wav",
        "generated_audio/sample_002.wav", 
        "generated_audio/sample_003.wav"
    ]
    
    # Calculate CLAP score
    score = clap_score_simple(prompts, audio_files)
    print(f'CLAP score: {score}')
    
    # Example usage 2: From folder with mapping
    prompt_mapping = {
        "sample_001": "a piano melody with soft drums",
        "sample_002": "rock music with electric guitar solo",
        "sample_003": "ambient electronic music"
    }
    
    score = clap_score_from_folder("generated_audio/", prompt_mapping)
    print(f'CLAP score from folder: {score}') 

