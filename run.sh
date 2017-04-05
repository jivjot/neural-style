rm output*
python neural_style.py --content ../testImages/contentImage.jpg --styles ../testImages/styleImage.jpg --output output50s.jpg --iterations 1000 --checkpoint-output "output%s.jpg" --checkpoint-iterations 10 --width 800 --style-segmentations styleSegmentation.pickle --content-segmentation contentSegmentation.pickle
