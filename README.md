# stylegan_ptq
Post-training quantization of StyleGAN 

This is an accompanying repository for the Skoltech's Machine Learning course project. The main goal of this project is to study the applicability of different methods of **post-training** quantization to StyleGAN generator.

## Experiments reproduction

In order to reproduce experiments one should firstly download FFHQ dataset with 128x128 resolution by following official instructions https://github.com/NVlabs/ffhq-dataset (```python download_ffhq.py -t```).
Next step is to generate datasets for computation of FID using prepare_data.py. Example of its usage:
```bash
> python prepare_data.py --weights ./weights/fp_model.pth --ffhq_path ./thumbnails128x128 --save_data_dir ./mlproject
```
See ``` python prepare_data.py --help``` for scripts' detailed explanation.

One also need to download pretrained weights that can be accesed by this [link](https://drive.google.com/file/d/13LHj4f739MRv41ABOzS2zCNpkjOgbein/view?usp=sharing).

Once you prepared data and download weights, experiments can be reproduced using notebooks provided in `notebooks/` directory:

- Notebook `reproduce_fids_from_pretrained.ipynb` contains code for reproduction of FIDs given pre-quantized (pre-trained) models. **Important note:** one might run this notebook directly on Google Colab by this [link](https://colab.research.google.com/drive/1NwRsoXC6R8VIWnIHFohTN3hbGRVvmBnz?usp=sharing) (all data handling steps are included). Total running time on colab ~ 8 hours (however one can reproduce only part of experiments by editing loop parameters). 
- Notebook `find_optimal_quantiles.ipynb` contains code for finding optimal quantiles for vanilla post-training quantization.
- Notebook `adaround-brecq.ipynb` and `lsq-brecq.ipynb` contain code for running BRECQ algorightm with AdaRound and LSQ weights quantization respectively.

# Results

See the table below containing detailed information about quality of generated images. An interesting finding of this project is that LSQ-based BRECQ significantly outperforms original, AdaRound-based BRECQ for 8 bit quantization.


| Method                       | Per-channel | Bits  | FID  | qFID |
|------------------------------|-------------|-------|------|------|
| full-precision               |     -       |   -   | 26.3 | 0.0  |
| vanilla PTQ                  |:heavy_check_mark:| a8/w8 | 66.5 | 51.2 |
| vanilla PTQ                  |   :x:     | a8/w8 | 65.9 | 50.4 |
| vanilla PTQ                   |:heavy_check_mark:| a8/w4 | 73.1 | 60.3 |
| vanilla PTQ                    |    :x:      | a8/w4 | 73.5 | 62.4 |
| AR-BRECQ                     | :heavy_check_mark:| a8/w8 | 66.8 | 49.9 |
| AR-BRECQ                     | :x:      | a8/w8 | 65.1 | 48.4 |
| AR-BRECQ                      | :heavy_check_mark:| a8/w4 | 61.2 | 46.2 |
| AR-BRECQ                      | :x:      | a8/w4 | 61.9 | 48.1 |
| LSQ-BRECQ                    | :heavy_check_mark:| a8/w8 | 52.9 | 37.2 |
| LSQ-BRECQ                     | :x:     | a8/w8 | 51.7 | 35.9 |
| LSQ-BRECQ                    | :heavy_check_mark:| a8/w4 | 61.1 | 49.8 |
| LSQ-BRECQ                    | :x:     | a8/w4 | 70.0 | 57.9 |

Overral, the results suggest that post-training quantization may provide samples of reasonable quality, however in order go further quantization-aware training is needed.  
