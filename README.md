# StableFusion

A Web ui for **Stable Diffusion Models**.

< under development, request features using issues, prs not accepted atm >

<a target="_blank" href="https://colab.research.google.com/drive/1gUZBNGlpKnksc6aTuSbj2Hbgp8Fn_vp_?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


![image](https://raw.githubusercontent.com/NeuralRealm/StableFusion/master/static/Screenshot1.png)
![image](https://raw.githubusercontent.com/NeuralRealm/StableFusion/master/static/Screenshot2.png)

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
- Convert ckpt file to diffusers
- Add your own diffusers model 
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

For Local Host
```bash
stablefusion app
```
or

For Public Shareable Link
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

## Acknowledgements

I would like to express my gratitude to [Abhishek Thakur](https://github.com/abhishekkrthakur) for sharing his code for the [diffuzers package](https://github.com/abhishekkrthakur/diffuzers). This code formed the basis of the implementation used in this project, and I am grateful for his contributions to the open source community.

## Contributing

StableFusion is an open-source project, and we welcome contributions from the community. Whether you're a developer, designer, or user, there are many ways you can help make this project better. Here are a few ways you can get involved:

- **Report issues:** If you find a bug or have a feature request, please open an issue on our [GitHub repository](https://github.com/NeuralRealm/StableFusion/issues). We appreciate detailed bug reports and constructive feedback.
- **Submit pull requests:** If you're interested in contributing code, we welcome pull requests for bug fixes, new features, and documentation improvements.
- **Spread the word:** If you enjoy using StableFusion, please help us spread the word by sharing it with your friends, colleagues, and social media networks. We appreciate any support you can give us!

### We believe that open-source software is the future of technology, and we're excited to have you join us in making StableFusion a success. Thank you for your support!
