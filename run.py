import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from configs import paths_config, hyperparameters, global_config
from utils.align_data import pre_process_images
from scripts.run_pti import run_PTI
from IPython.display import display
import matplotlib.pyplot as plt
from scripts.latent_editor_wrapper import LatentEditorWrapper

def load_generators(model_id, image_name):
    with open(f'{paths_config.checkpoints_dir}/model_{model_id}_{image_name}.pt', 'rb') as f_new: 
        new_G = torch.load(f_new).cuda()
    return new_G

def gen_vec(image_name, latent_editor, alpha=2):
    w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
    embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
    w_pivot = torch.load(f'{embedding_dir}/0.pt')
    latents_vec = latent_editor.get_single_interface_gan_edits(w_pivot, [-alpha,alpha])
    return latents_vec
    
def gen_img(image_name, model_id, latents_vec, base_save_path):
    generator_type = paths_config.multi_id_model_type if hyperparameters.use_multi_id_training else image_name
    new_G = load_generators(model_id, generator_type)
    for direction, factor_and_edit in latents_vec.items():
        for val, latent in factor_and_edit.items():
            img = new_G.synthesis(latent, noise_mode='const', force_fp32 = True)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]  
            img = Image.fromarray(img, mode='RGB')
            path = os.path.join(base_save_path, image_name, direction)
            os.makedirs(path, exist_ok=True)
            img.save(os.path.join(path, str(val) + image_name + '.jpg'))
            
def evaluate():
    os.makedirs(paths_config.input_data_path, exist_ok=True)
    pre_process_images('/home/data/image_original')
    model_id = run_PTI(use_wandb=False, use_multi_id_training=hyperparameters.use_multi_id_training)
    latent_editor = LatentEditorWrapper()
    name_list = os.listdir('/home/data/image_original')
    base_save_path = os.path.join('/home/data/image_results', paths_config.input_data_id)
    os.makedirs(base_save_path, exist_ok=True)
    with torch.no_grad():
        for image_name in tqdm(name_list):
            image_name = image_name.split('.')[0]
            latents_vec = gen_vec(image_name, latent_editor, alpha=2)
            gen_img(image_name, model_id, latents_vec, base_save_path)
            print(f'Done for {image_name}') 
    
    
if __name__ == "__main__":
    evaluate()