# Install the requirements
pip install -r requirements.txt

# Unconditional generation with angkorwat.jpg
!python main_train.py --gpu 0 --train_mode generation --input_name Images/Generation/angkorwat.jpg

# Modify learning rate, default value is 0.1
!python main_train.py --gpu 0 --train_mode generation --input_name Images/Generation/marinabaysands.png --lr_scale 0.5

# Modify training stages, default value is 6
!python main_train.py --gpu 0 --train_mode generation --input_name Images/Generation/zebra.png --train_stages 7

# Display outputs
cd TrainedModels/
%load_ext tensorboard
%tensorboard --logdir

# Unconditional generation of arbityrary sizes
!python main_train.py --gpu 0 --train_mode retarget --input_name Images/Generation/colusseum.png

# Image Harmonization
!python main_train.py --gpu 0 --train_mode harmonization --train_stages 3 --min_size 120 --lrelu_alpha 0.3 --niter 1000 --batch_norm --input_name Images/Harmonization/scream.jpg

# use a naive images to monitor the progress
!python main_train.py --gpu 0 --train_mode harmonization --train_stages 3 --min_size 120 --lrelu_alpha 0.3 --niter 1000 --batch_norm --input_name Images/Harmonization/scream.jpg --naive_img Images/Harmonization/scream_naive.jpg

# Fine-tune and pre-train model on a given image
!python main_train.py --gpu 0 --train_mode harmonization --input_name Images/Harmonization/scream.jpg --naive_img Images/Harmonization/scream_naive.jpg --fine_tune --model_dir TrainedModels/scream/...

# Harmonize a given image with a trained model
!python evaluate_model.py --gpu 0 --model_dir TrainedModels/scream/.../ --naive_img Images/Harmonization/scream_naive.jpg

# Image Editing, also a naive image monitoring the progress
!python main_train.py --gpu 0 --train_mode editing --batch_norm --niter 1000 --input_name Images/Editing/stone.png
!python main_train.py --gpu 0 --train_mode editing --batch_norm --niter 1000 --input_name Images/Editing/stone.png --naive_img Images/Editing/stone_edit_1.png
!python main_train.py --gpu 0 --input_name Images/Editing/stone.png --naive_img Images/Editing/stone_edit_1.png --fine_tune --model_dir TrainedModels/stone/...
!python evaluate_model.py --gpu 0 --model_dir TrainedModels/stone/.../ --naive_img Images/Harmonization/stone_edit_1.png


# Citation
# @inproceedings{hinz2021improved,
#     author    = {Hinz, Tobias and Fisher, Matthew and Wang, Oliver and Wermter, Stefan},
#     title     = {Improved Techniques for Training Single-Image GANs},
#     booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
#     month     = {January},
#     year      = {2021},
#     pages     = {1300--1309}
# }
