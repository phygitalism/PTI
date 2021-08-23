from configs import paths_config, global_config
import dlib
import glob
import os
from tqdm import tqdm
from utils.alignment import align_face



def pre_process_images(raw_images_path):
    IMG_SIZE = 1024#power2(min(imageio.imread(image_name).shape[:2]))
#     current_directory = os.getcwd()
    predictor = dlib.shape_predictor(paths_config.dlib)
#     os.chdir(raw_images_path)
    images_names = glob.glob(os.path.join(raw_images_path, '*'))

    aligned_images = []
    for image_name in tqdm(images_names):
        
        aligned_image = align_face(filepath=image_name,
                                   predictor=predictor, output_size=IMG_SIZE)
        aligned_images.append(aligned_image)

#     os.makedirs('./image_original/image_processed', exist_ok=True)
    for image, name in zip(aligned_images, images_names):
        real_name = os.path.basename(name).split('.')[0]
        image.save(os.path.join(paths_config.input_data_path, real_name+'.jpeg'))

#     os.chdir(current_directory)


if __name__ == "__main__":
    pre_process_images('')
