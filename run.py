import argparse
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
import shutil

def load_generators(model_id, image_name):
    with open(f'{paths_config.checkpoints_dir}/model_{model_id}_{image_name}.pt', 'rb') as f_new: 
        new_G = torch.load(f_new).cuda()
    return new_G

def gen_vec(image_name, latent_editor, alpha, step):
    w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
    embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
    w_pivot = torch.load(f'{embedding_dir}/0.pt')
    latents_vec = latent_editor.get_single_interface_gan_edits(w_pivot, np.linspace(-alpha, alpha, step))
    return latents_vec
    
def gen_img(image_name, model_id, latents_vec, base_save_path):
    image_name, ext = image_name.split('.')
    generator_type = paths_config.multi_id_model_type if hyperparameters.use_multi_id_training else image_name
    new_G = load_generators(model_id, generator_type)
    for direction, factor_and_edit in latents_vec.items():
        for val, latent in factor_and_edit.items():
            img = new_G.synthesis(latent, noise_mode='const', force_fp32 = True)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]  
            img = Image.fromarray(img, mode='RGB')
            path = os.path.join(base_save_path, image_name, direction)
            os.makedirs(path, exist_ok=True)
            img.save(os.path.join(path, str(val) + "_" + image_name + '.' + ext))
            
def evaluate(args):
    os.makedirs(paths_config.input_data_path, exist_ok=True)
    pre_process_images('/home/data/image_original')
    model_id = run_PTI(use_wandb=False, use_multi_id_training=hyperparameters.use_multi_id_training)
    latent_editor = LatentEditorWrapper()
    name_list = os.listdir('/home/data/image_original')
    base_save_path = os.path.join('/home/data/image_results', paths_config.input_data_id)
    os.makedirs(base_save_path, exist_ok=True)
    with torch.no_grad():
        for image_name in tqdm(name_list):
            latents_vec = gen_vec(image_name.split('.')[0], latent_editor, alpha=args.alpha, step=args.step)
            gen_img(image_name, model_id, latents_vec, base_save_path)
            print(f'Done for {image_name}') 
            
def clean_up():
    shutil.rmtree(paths_config.input_data_path)
    shutil.rmtree(paths_config.checkpoints_dir)
    shutil.rmtree(paths_config.embedding_base_dir)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=10, help="[-alpha,... alpha] range")
    parser.add_argument("--step", type=int, default=20, help="num for numpy.linspace")
    parser.add_argument("--data_name", type=str, default='test', help="dataset name")
    parser.add_argument('--clean_up', action='store_true', default=True, help='delete permanent files after run')
    args = parser.parse_args()
    paths_config.input_data_id = args.data_name
    evaluate(args)
    if args.clean_up:
        clean_up()