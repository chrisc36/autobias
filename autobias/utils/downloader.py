import logging
import math
import tempfile
import zipfile
from os import makedirs
from os.path import dirname

import requests
from tqdm import tqdm


def ensure_dir_exists(filename):
  """Make sure the parent directory of `filename` exists"""
  makedirs(dirname(filename), exist_ok=True)


def download_to_file(url, output_file):
  """Download `url` to `output_file`, intended for small files."""
  ensure_dir_exists(output_file)
  with requests.get(url) as r:
    r.raise_for_status()
    with open(output_file, 'wb') as f:
      f.write(r.content)


def download_zip(name, url, source, progress_bar=True):
  """Download zip file at `url` and extract to `source`"""
  makedirs(source, exist_ok=True)
  logging.info("Downloading %s" % name)

  # Probably best to download to a temp file to ensure we
  # don't eat a lot of RAM with downloading a large file
  tmp_f = tempfile.TemporaryFile()
  with requests.get(url, stream=True) as r:
    _write_to_stream(r, tmp_f, progress_bar)

  logging.info("Extracting to %s...." % source)
  with zipfile.ZipFile(tmp_f) as f:
    f.extractall(source)


DRIVE_URL = "https://docs.google.com/uc?export=download"


def download_from_drive(file_id, output_file, progress_bar=False):
  """Download the public google drive file `file_id` to `output_file`"""
  ensure_dir_exists(output_file)

  session = requests.Session()

  response = session.get(DRIVE_URL, params={'id': file_id}, stream=True)

  # Check to see if we need to send a second, confirm, request
  # https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
  for key, value in response.cookies.items():
    if key.startswith('download_warning'):
      params = {'id': file_id, 'confirm': value}
      response = session.get(DRIVE_URL, params=params, stream=True)
      break

  with open(output_file, "wb") as f:
    _write_to_stream(response, f, progress_bar)
  response.close()


def _write_to_stream(response, output_fh, progress_bar=True, chunk_size=32768):
  """Write streaming `response` to `output_fs` in chunks"""
  mb = 1024*1024
  response.raise_for_status()
  if progress_bar:
    # tqdm does not format decimal numbers. We could in theory add decimal formatting
    # using the `bar_format` arg, but in practice doing so is finicky, in particular it
    # seems impossible to properly format the `rate` parameter. Instead we just manually
    # ensure the 'total' and 'n' values of the bar are rounded to the 10th decimal place
    content_len = response.headers.get("Content-Length")
    if content_len is not None:
      total = math.ceil(10 * float(content_len) / mb) / 10
    else:
      total = None
    pbar = tqdm(desc="downloading", total=total, ncols=100, unit="mb")
  else:
    pbar = None

  cur_total = 0
  for chunk in response.iter_content(chunk_size=chunk_size):
    if chunk:  # filter out keep-alive new chunks
      if pbar is not None:
        cur_total += len(chunk)
        next_value = math.floor(10 * cur_total / mb) / 10.0
        pbar.update(next_value - pbar.n)
      output_fh.write(chunk)

  if pbar is not None:
    if pbar.total is not None:
      pbar.update(pbar.total - pbar.n)  # Fix rounding errors just for neatness
    pbar.close()
