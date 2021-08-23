## Pretrained models paths
# e4e = '/home/data/pretrained_models/e4e_ffhq_encode.pt'
stylegan2_ada_ffhq = '/home/data/pretrained_models/ffhq.pkl'
# style_clip_pretrained_mappers = ''
# ir_se50 = '/home/data/pretrained_models/model_ir_se50.pth'
dlib = '/home/data/pretrained_models/align.dat'

## Dirs for output files
checkpoints_dir = './checkpoints'
embedding_base_dir = './embeddings'
# styleclip_output_dir = './StyleCLIP_results'
# experiments_output_dir = './output'

## Input info
### Input dir, where the images reside
input_data_path = './image_preprocessed/'
### Inversion identifier, used to keeping track of the inversion results. Both the latent code and the generator
input_data_id = 'test'

## Keywords
pti_results_keyword = 'PTI'
# e4e_results_keyword = 'e4e'
# sg2_results_keyword = 'SG2'
# sg2_plus_results_keyword = 'SG2_plus'
multi_id_model_type = 'multi_id'

## Edit directions
interfacegan_age = '/home/PTI/editings/interfacegan_directions/age.pt'
interfacegan_smile = '/home/PTI/editings/interfacegan_directions/smile.pt'
interfacegan_rotation = '/home/PTI/editings/interfacegan_directions/rotation.pt'
ffhq_pca = '/home/PTI/editings/ganspace_pca/ffhq_pca.pt'
