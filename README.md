# StableFusion

A web ui and deployable API for **Stable Diffusion Models**.

< under development, request features using issues, prs not accepted atm >

<a target="_blank" href="https://colab.research.google.com/github/abhishekkrthakur/diffuzers/blob/main/diffuzers.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


![image](https://github.com/abhishekkrthakur/diffuzers/raw/main/static/screenshot.jpeg)


If something doesnt work as expected, or if you need some features which are not available, then create request using [github issues](https://github.com/NeuralRealm/StableFusion/issues)


## Features available in the app:

- text to image
- image to image
- instruct pix2pix
- textual inversion
- image info
- stable diffusion upscaler
- gfpgan
- clip interrogator
- more coming soon!



## Installation

To install bleeding edge version of StableFusion, clone the repo and install it using pip.

```bash
git clone https://github.com/NeuralRealm/StableFusion
cd StableFusion
pip install -e .
```

Installation using pip:
    
```bash 
pip install stablefusion
```

## Usage

### Web App
To run the web app, run the following command:

```bash
stablefusion app
```
or
```bash
stablefusion app --port 10000 --ngrok_key YourNgrokAuthtoken --share
```

## All CLI Options for running the app:

```bash
❯ stablefusion app --help
usage: stablefusion <command> [<args>] app [-h] [--output OUTPUT] [--share] [--port PORT] [--host HOST]
                                        [--device DEVICE] [--ngrok_key NGROK_KEY]

✨ Run stablefusion app

optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT       Output path is optional, but if provided, all generations will automatically be saved to this
                        path.
  --share               Share the app
  --port PORT           Port to run the app on
  --host HOST           Host to run the app on
  --device DEVICE       Device to use, e.g. cpu, cuda, cuda:0, mps (for m1 mac) etc.
  --ngrok_key NGROK_KEY
                        Ngrok key to use for sharing the app. Only required if you want to share the app
```


## Using private models from huggingface hub

If you want to use private models from huggingface hub, then you need to login using `huggingface-cli login` command.

Note: You can also save your generations directly to huggingface hub if your output path points to a huggingface hub dataset repo and you have access to push to that repository. Thus, you will end up saving a lot of disk space. 
