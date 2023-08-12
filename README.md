# Cross-Modality segmentation : MoDATTS

![Screenshot](MoDATTS.png)

## Initialization

Clone repo "git clone https://github.com/MaloADBA/MGenSeg_2D.git"

Run "pip install -r requirements.txt" to install dependencies. 

Run "git submodule init" to initialize submodules.

Run "git submodule update" to download submodules.

Before laucnhing any experiment run : "source register submodules.sh"

## Task : domain adaptation for tumor segmentation with BRATS

We build an unsupervised domain adaptation task with BraTS where each MR contrast (T1,T1ce,FLAIR,T2) is considered as a distinct modality. The models provided aim at reaching good segmentation performances on an unlabeled target modality dataset by leveraging annotated source images of an other modality.

We use the 2020 version of the BRATS data from https://www.med.upenn.edu/cbica/brats2020/data.html. Download the data to `<download_dir_brats>`.

### Modality translation phase

The data can be prepared for the cross-modality translation using a provided script, as follows:

```
python scripts/data_preparation/Prepare_multimodal_brats_trans_2D_uncut.py --data_dir "<download_dir_brats>" --save_to "/path/translation_transunet.h5" 

```
Data preparation creates a new dataset based on BRATS that contains 2D slices, split into subsets for each possible contrast (T1, T1ce, FLAIR, T2).

The modality translation model can be trained using the following command line:

```
python3 mbrats_2d_trans.py --data '/path/translation_transunet.h5' --path '/log_and_save_model_to/' --model_from 'model/configs/3d_mbrats/Transunet_mbrats_trans.py' --model_kwargs '{"lambda_enforce_sum": 1, "lambda_disc": 3, "lambda_seg": 1, "lambda_cyc": 40}' --weight_decay 0.0001 --source_modality 't1' --target_modality 't2' --batch_size_train 15 --batch_size_valid 15 --epochs 150 --opt_kwargs '{"betas": [0.5, 0.999], "lr": 0.0001}' --optimizer amsgrad --nb_proc_workers 2 --n_vis 4 --init_seed 1234 --data_seed 0 --dispatch_canada --account def-sakad --time '12:0:0' --cca_mem '32G' --cca_cpu 8 --copy_local

```

Loss hyperparameters can be changed in the `--model_kwargs` argument.

Pick source and target contrasts (t1, t1ce, flair or t2) with `--source_modality` and `--target_modality` arguments.
