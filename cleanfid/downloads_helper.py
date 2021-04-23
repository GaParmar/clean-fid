import os
import sys
from argparse import ArgumentParser
import urllib.request
import requests
import shutil


inception_url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt"


def check_download_inception(fpath="./"):
    inception_path = os.path.join(fpath, "inception-2015-12-05.pt")
    if not os.path.exists(inception_path):
        # download the file
        with urllib.request.urlopen(inception_url) as response, open(inception_path, 'wb') as f:
            shutil.copyfileobj(response, f)
    return inception_path


def check_download_url(local_folder, url):
    name = os.path.basename(url)
    local_path = os.path.join(local_folder, name)
    if not os.path.exists(local_path):
        os.makedirs(local_folder, exist_ok=True)
        print(f"downloading statistics to {local_path}")
        with urllib.request.urlopen(url) as response, open(local_path, 'wb') as f:
            shutil.copyfileobj(response, f)
    return local_path


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

# download stylegan-ffhq-1024 weights
def download_google_drive(file_id="1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT", out_path="stylegan2-ffhq-config-f.pt"):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    CHUNK_SIZE = 32768
    with open(out_path, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--download_stylegan_weights', default=False, action="store_true")
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()

    if args.download_stylegan_weights:
        gdrive_id = "1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT"
        print("downloading pretrained stylegan2 from google drive...")
        download_google_drive(file_id=gdrive_id, out_path=args.save_path)
        print("download complete")
