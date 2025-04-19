# Fair Point-of-Interest Recommendation

This repository includes all necessary steps (data sample generation, preprocessing, saving data files for plug-in into recommendation frameworks, re-ranking for popularity bias mitigation, accuracy-based evaluation, and user-centered evaluation) to generate baseline and re-ranked POI recommendations and evaluate their performance. To facilitate the reproducibility of the recommendations, we use the recommender frameworks [RecBole](https://github.com/andreafooo/RecBole) for general recommender models and [CAPRI](https://github.com/andreafooo/CAPRI) for Context-Aware Point-of-Interest Recommendation. 

#### Note: If you don't want to follow the entire recommendation pipeline, you can take a shortcut to the evaluation to perform this based on the results from the Yelp dataset. (TO-DO)


## Full Reproduction of Results

### Preprocessing

1. create a virtual environment and activate it
```
python3 -m venv venv
source venv/bin/activate
```
2. Install the requirements 
```
pip3 install -r requirements.txt
```
3. Create a script ```globals.py``` in the root directory and add the line ```BASE_DIR = /path/to/your/base/directory/```. This base directory will be used to store the datasets and the recommender outputs. 

4. In the ```BASE_DIR``` proceed by creating dataset folders with the following structure ```<dataset_name>_dataset``` and then place the original, (unzipped) data files in this folder.

Links to the original datasets used in this study: 
* [yelp_dataset](https://www.yelp.com/dataset)
* [gowalla_dataset](https://snap.stanford.edu/data/loc-gowalla.html)
* [brightkite_dataset](https://snap.stanford.edu/data/loc-brightkite.html)
* [foursquaretky_dataset](https://www.kaggle.com/datasets/chetanism/foursquare-nyc-and-tokyo-checkin-dataset)

5. Open the script ```data_sampler_alldata.ipynb``` and input the desired dataset for which you would like to generate the samples (default n=1500 users). The samples include three user groups; 1/3 that visited the most popular POIs, 1/3 around the popularity median and 1/3 that visited the least popular POIs. The train/validation/test (65/15/20) splits are performed based on a user-based temporal split. The samples are processed to fit the layout for RecBole and CAPRI and saved into the respective subfolders.

### Generate Recommendations
Generate Recommendations in [RecBole](https://github.com/andreafooo/RecBole) for general recommender models and [CAPRI](https://github.com/andreafooo/CAPRI) for Context-Aware Point-of-Interest Recommendation. Clone the repositories, generate data folders for the datasets and place the preprocessed files into the respective folders. 

#### Recbole 
In the forked RecBole repository, follow steps 1-3 again. Hyperparameter tuning: specify the hyperparameters to test in the file ```hyper.test```, then open ```config_hyper_parameter_creator.py``` and specify the datasets and (general recommendation) models for which you would like to perform hyperparameter tuning. The config files with the ideal hyperparameters will automatically be added to the folder ```config``` in the root directory. Once completed, open the script ```recbole_full_casestudy.py```. This will take into account the ideal hyperparameters and train the recommender models. The outputs of the general evaluation and the top-k recommendations can be found in the ```recommendations``` subfolder of the dataset in your ```BASE_DIR```. 
#### CAPRI
Follow steps 1-2 in the forked CAPRI repository. In the ```config.py```you can specify which contexts (Geographical, Social, Temporal, Interaction) each dataset features. For our study, social connections are not relevant, therefore we remove them from the data. Run the script ```main.py``` and choose the desired model and dataset. To reproduce the results, use "sum" fusion method. Create a folder with this exact structure in the ```recommendations``` subdirectory: ```<dataset>_sample-contextpoi-<model_name>-Jan-01-2024_09-00-00``` and manually transfer the Eval_ and Rec_ files for the respective model into this directory. Return to this repository, open the script ```capri_postprocessing.py```, specify the desired datasets and run the script to process the outputs (general evaluation and top-k recommendations) to be in line with those produced by RecBole.

### Evaluation

#### General Evaluation
The evaluation protocol includes overall and group-specific evaluations. The evaluations are accuracy-based (NDCG) and fairness-based (Average Recommendation Popularity aka ARP and Popularity Lift aka PopLift), aiming to mitigate popularity bias and keep performance adequate. The evaluation is also compared between the user subgroups. 

#### Debiasing by Re-ranking recommendations
For all models, re-sampling for popularity bias mitigation is performed and the outputs are combined with the respective base versions. The aim of conducting this simple debiasing method is to calibrate the recommendations to be more in line with the user profile popularity. This is done by calculating the deviation between the item popularity of the items in the candidate set, and the user profile popularity and re-ranking them based on the smallest deviation.

The scripts ```evaluation.ipynb``` and ```user_centered_evaluation.ipynb``` include the full evaluation protocol. 

