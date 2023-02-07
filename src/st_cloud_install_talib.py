# Streamlit cloud TA-Lib installation taken from here:
# https://discuss.streamlit.io/t/ta-lib-streamlit-deploy-error/7643/10

import requests
import os
import sys
import subprocess

# add the library to our current environment
from ctypes import *

def install_talib():
    # check if the library folder already exists, to avoid building everytime you load the path
    if not os.path.isdir("/tmp/ta-lib"):

        # Download ta-lib to disk
        with open("/tmp/ta-lib-0.4.0-src.tar.gz", "wb") as file:
            response = requests.get(
                "http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz"
            )
            file.write(response.content)
        # get our current dir, to configure it back again. Just house keeping
        default_cwd = os.getcwd()
        os.chdir("/tmp")
        # untar
        os.system("tar -zxvf ta-lib-0.4.0-src.tar.gz")
        os.chdir("/tmp/ta-lib")
        os.system("ls -la /app/equity/")
        # build
        os.system("./configure --prefix=/home/appuser")
        os.system("make")
        # install
        os.system("make install")
        # back to the cwd
        os.chdir(default_cwd)
        sys.stdout.flush()

    lib = CDLL("/home/appuser/lib/libta_lib.so.0.0.0")
    # import library
    try:
        import talib
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--global-option=build_ext", "--global-option=-L/home/appuser/lib/", "--global-option=-I/home/appuser/include/", "ta-lib"])
    finally:
        import talib
