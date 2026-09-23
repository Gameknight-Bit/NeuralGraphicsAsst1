## Environemnt Setup:

To set up the python environment, stay in the root directory and run the following commands:
```
python -m venv venv
//Activate venv
pip install -r requirements.txt
``` 

## Reproducing Results
PyTorch should automatically detect what device you are running (CUDA, CPU, Etc.)
No extra setup is needed for this.

In order to reproduce results we can either run the two static stripts `$python runp6.py` or `$python runp7.py` to get the respective compressed images, model weights, and graphs that are relevant to problems 6 and 7 on the project page.

To test more dynamicaly run `$python run_textures.py <texture file names here> --output <output folder path>`\
The full arguments/usage are as follows:
`usage: run_textures.py [-h] [--output OUTPUT] [--steps STEPS] [--batch-size BATCH_SIZE] [--quantize-mlp] textures [textures ...]`

**Example:**\
`$python run_textures.py gradient.png bricks.png clouds.png --output p6_p7_results`

When executing `run_textures.py`, for every texture we compute the DXT1 baseline (in s3tc.py), train the small, medium, and large models, quantize each model and eval PSNR again, and finally report sizes, compression ratios, compression factors, and quantization loss. \
This gives good enough results to successfully analyze our methods!