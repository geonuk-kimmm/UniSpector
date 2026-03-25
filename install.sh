pip install -r requirements.txt
git clone https://github.com/facebookresearch/Mask2Former
cd Mask2Former/mask2former/modeling/pixel_decoder/ops
sh make.sh
cd ../../../../..
python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
pip install git+https://github.com/cocodataset/panopticapi.git
pip install -r requirements.txt