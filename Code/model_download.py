# Downloads the model from Google Drive
import argparse
import gdown
import os

# Run
parser = argparse.ArgumentParser()
parser.add_argument("--path", default=None, type=str, required=True)  # Path of file
args = parser.parse_args()
PATH = args.path
MODEL_PATH = PATH + os.path.sep + 'Data'

os.chdir(MODEL_PATH)
url = 'https://drive.google.com/u/3/uc?id=1q2QAYGM-hmFxmH_ytHW-L4lw_792iAEx&export=download'
gdown.download(url, 'model_onehot.pt', quiet=False)
print('Done!')