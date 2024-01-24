# Deep iGlu Denoiser

The Deep iGlu Denoiser is a powerful tool designed for denoising microscopic recordings, offering pre-trained model weights for the **iGlu-Snfr3 sensor** ready to use. This denoising is built upon the [U-Net](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) model architecture. The model can be trained on any microscopic data, without the need for manual data curation.

| **raw**                                                             | denoised                                                                      |
| ------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| <img src="graphics/raw.gif" title="raw" alt="raw synapse" width="210"> | <img src="graphics/denoised.gif" title="denoised" alt="denoised synapse" width="210"> |

 

## Getting started

Follow these steps to set up Deep iGlu Denoiser:

1. **Create a new enviorment:**
   
   ```bash
   conda create -n deep_iglu_denoiser python=3.10 pip
   ```

2. **Activate enviorment:**
   
   ```bash
   conda activate deep_iglu_denoiser
   ```

3. **Clone the repository:**
   
   ```bash
   git clone https://github.com/s-weissbach/deep_iglu_denoiser.git
   ```

4. **Download Pre-trained Model:**
   Download the pre-trained model from [the release page](https://github.com/s-weissbach/deep_iglu_denoiser/releases/) and place it in the project directory.

5. **Install Requirements:**
   
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the following command to denoise images using the provided script:

```bash
python denoise.py --path /path/to/images --modelpath /path/to/model_weights --directory_mode -o /output/path
```

### Arguments:

- `--path (-p)`: Path to the input imagestack or directory containing images.
- `--modelpath (-m)`: Path to the pre-trained model weights.
- `--directory_mode (-d)`: Flag to enable directory mode.
- `--outputpath (-o)`: Path to the output directory.
- `--batchsize (-b)`: Number of frames predicted at once (depends on the vRAM of your graphics card) – default: 1.

### Supported File Formats

- tiff-based formats: `.tif, .tiff, .stk`

- Nikon format: `.nd2`

All files will be written as a `.tiff`file. 

If you need other file formats to be supported, feel free to open an issue on GitHub.

## Example

Denoise a single imagestack:

```bash
python denoise.py --path /path/to/imagestack.tiff --modelpath /path/to/model.pt --outputpath /output/path
```

Denoise all images in a directory:

```bash
python denoise.py --path /path/to/images_folder --modelpath /path/to/model_weights --directory_mode -o /output/path
```

Note: Make sure to replace "/path/to/" and "/output/path" with the actual paths on your system.

# Training a Custom Model

To train a custom model for denoising, follow these steps:

## 1. Prepare Training Data

Store all recordings with **one sensor** and store them in a directory `/path/to/traindata/`. **The recordings itself can be noisy.**

Use the `prepare_trainfiles.py` script to generate training data from a set of images. The script takes the following arguments:

- `--csv`: Output CSV file containing meta information for training examples.
- `--path (-p)`: Path to the folder containing images.
- `--fileendings (-f)`: List of file endings to consider.
- `--crop_size (-c)`: Crop size used during training (default: 32).
- `--roi_size`: Expected ROI size; assumes for detection square of (roi_size x roi_size) (default: 4).
- `--trainh5 (-t)`: Path to the output H5 file that will be created.
- `--min_z_score (-z)`: Minimum Z score to be considered an active patch (default: 2).
- `--before`: Number of frames to add before a detected event, to also train to reconstruct the typical raise of the sensor's signal (default: 0).
- `--after`: Number of frames to add after a detected event, to also train to reconstruct the typical decay of the sensor's signal (default: 0).
- `--window_size (-w)`: Number of frames used for rolling window z-normalization (default: 50).
- `--fgsplit (-s)`: Foreground to background split (default: 0.5).
- `--overwrite`: Overwrite existing H5 file. If false, data will be appended (default: False).

Example usage:

`python prepare_trainfiles.py --csv train_data.csv --path /path/to/traindata --fileendings tif tiff --crop_size 32 --roi_size 8 --trainh5 training_data.h5 --min_z_score 2.0 --before 0 --after 0 --window_size 50 --fgsplit 0.5 --overwrite False`

If the csv-file and the h5-file exist, the new videos will be appended to the h5 file. It is recommended to **backup the h5-file** before running this script.

## 2. Prepare config file

<a "config> Create a `trainconfig.yaml` file with the following configuration settings. </a>

```yaml
modelpath: 'unet.pt'
train_h5: '/path/to/train.h5'
batch_size: 16
learning_rate: 0.0001
num_epochs: 1
path_example_img: '/path/to/example.tiff'
target_frame_example_img: 101
predict_every_n_batches: 10000
noise_center: 0.0
noise_scale: 1.5
```

Adjust the paths and parameters in the configuration file based on your specific setup and requirements. This configuration file will be used during the training process to specify various parameters:

- **modelpath**: Path to save the model after training.

- **train_h5**: Path to the h5 file containing the training data 

- **batch_size**: Number of training examples utilized in one iteration (limited by the vRAM of your graphics card).

- **learning_rate**: Rate at which the model's weights are updated during training.

- **num_epochs**: Number of times the entire training dataset is passed forward and backward through the neural network.

- **path_example_img**: Path to an example image used for visualization of the training progress, if you don't want to use it set to `''`.

- **target_frame_example_img**: Frame number in the example image to be used as the prediction target during ttraining.

- **predict_every_n_batches**: Frequency at which the model predicts outputs during training, useful for monitoring progress.

- **noise_center**: Center of the noise added to the input data during training (noise augmentation). A center of 0 is recommended, since the z-normalized images have a mean of 0 for each pixel.

- **noise_scale**: Scale of the noise added to the input data during training (noise augmentation).

## 3. Train the model

Run the training script by executing the following command:

```bash
python start_training.py --trainconfigpath /path/to/trainconfig.yaml`
```

`--trainconfigpath (-p)`: Path to the [train config YAML file](config) containing training parameters.

When a CUDA capable GPU is found `GPU ready` will be printed; otherwise `Warning: only CPU found`. It is not recommended to train with a CPU only.