# thapelo-sindane-msc-public
Public Repository containing msc code
This repo contains code for Part of Speech Tagging (POS_code), Named Entity Recignition (NER_Code), Machine Translation (MT_code), and Code Switching using cross-lingual embeddings. Embeddings are generally large and cannot be uploaded to github, and the the user needs to have two sets of embeddings to be able to use this code. The embedings need to following Glove or FasText text formats when saved.

# How to run:
## POS Code (MSC_Code_data/POS_code/)
* important directories are data_path is a path that contains a folder with tha train,dev, test files in text format for POS Tagging; cross_emb_path contains two files of cross-lingual embeddings for the observed languages, named using {source_language_code}-{target_language_code}-{projection_model_name}.txt ; model_desitnation_path is the output path; img_dir is the output directory for all plots.

* Once the paths are defined correclty, you run > nohup python crosslingual_embeddings_thapelo_msc.py. To run the same experiments for monolingual embeddings, the commented code in the script after Monolingual Traing must be uncommented and the top section should all be commented.
## NER Code (MSC_Code_data/NER_code/)
* important directories are data_path is a path that contains a folder with tha train,dev, test files in text format for NER; cross_emb_path contains two files of cross-lingual embeddings for the observed languages, named using {source_language_code}-{target_language_code}-{projection_model_name}.txt ; model_desitnation_path is the output path; img_dir is the output directory for all plots.

* Once the paths are defined correclty, you run > nohup python crosslingual_embeddings_thapelo_msc.py. To run the same experiments for monolingual embeddings, the commented code in the script after Monolingual Traing must be uncommented and the top section should all be commented.
## Machine Translation Code (MSC_Code_data/MT_code/)
* important directories are data_path is a path that contains a folder with tha train,dev, test files in text format for NER; cross_emb_path contains two files of cross-lingual embeddings for the observed languages, named using {source_language_code}-{target_language_code}-{projection_model_name}.txt ; model_desitnation_path is the output path; img_dir is the output directory for all plots.

* Once the paths are defined correclty, you run > nohup python crosslingual_embeddings_thapelo_msc.py. To run the same experiments for monolingual embeddings, the commented code in the script after Monolingual Traing must be uncommented and the top section should all be commented.
## News Headlines Classification Code (MSC_Code_data/NHC_code/)
* important directories is path containing source training data and target dataset.

* Once the paths are defined correclty, use notebook
