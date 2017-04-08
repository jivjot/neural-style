rm output*
#python neural_style.py --content Seth.jpg --styles Gogh.jpg --output output50s.jpg --iterations 1000 --checkpoint-output "output%s.jpg" --checkpoint-iterations 10  --style-segmentations Gogh_sem.pickle --content-segmentation Seth_sem.pickle 
#python neural_style.py --content ../testImages/contentImage.jpg --styles ../testImages/styleImage.jpg --output output50s.jpg --iterations 1000 --checkpoint-output "output%s.jpg" --checkpoint-iterations 10 --width 800 --style-segmentations styleSegmentation.pickle --content-segmentation contentSegmentation.pickle --content-weight 0 --tv-weight 0 
python neural_style.py --content Landscape_sem.png --styles Renoir.jpg --output output50s.jpg --iterations 1000 --checkpoint-output "output%s.jpg" --checkpoint-iterations 10  --style-segmentations Renoir_sem.pickle --content-segmentation Landscape_sem.pickle --content-weight 0 --tv-weight 0
