# Fair Point-of-Interest Recommendation

This repository includes all necessary steps (data sample generation, preprocessing, saving data files for plug-in into recommendation frameworks, re-ranking for popularity bias mitigation, accuracy-based evaluation, and user-centered evaluation) to generate baseline and re-ranked POI recommendations and evaluate their performance. To facilitate the reproducibility of the recommendations, we use the recommender frameworks [RecBole](https://github.com/RUCAIBox/RecBole) for general recommender models and [CAPRI](https://github.com/CapriRecSys/CAPRI) for Context-Aware Point-of-Interest Recommendation. 

#### Note: If you don't want to follow the entire recommendation pipeline, you can take a shortcut to the evaluation to perform this based on the results from the Foursquare dataset.


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
3. Create/update the script ```globals.py``` in the root directory and add the line ```BASE_DIR = /path/to/your/base/directory/```. This base directory will be used to store the datasets and the recommender outputs. 

4. In the ```BASE_DIR``` proceed by creating dataset folders with the following structure ```<dataset_name>_dataset``` and then place the original, (unzipped) data files in this folder.

Links to the original datasets used in this study: 
* [yelp_dataset](https://www.yelp.com/dataset)
* [gowalla_dataset](https://snap.stanford.edu/data/loc-gowalla.html)
* [brightkite_dataset](https://snap.stanford.edu/data/loc-brightkite.html)
* [foursquaretky_dataset](https://www.kaggle.com/datasets/chetanism/foursquare-nyc-and-tokyo-checkin-dataset)

5. Data Sampling & Preprocessing: Add the desired datasets to ```globals.py``` and call ```data_sampling.py```from the root directory (default n=1500 users). The samples include three user groups; 1/3 that visited the most popular POIs, 1/3 around the popularity median and 1/3 that visited the least popular POIs. The train/validation/test (65/15/20) splits are performed based on a user-based temporal split & duplicate check-ins are transformed into a check-in count. The samples are processed to fit the layout for RecBole and CAPRI and saved into the respective subfolders.

### Generate Recommendations
Generate Recommendations using [RecBole](https://github.com/RUCAIBox/RecBole) for general recommender models and [CAPRI](https://github.com/CapriRecSys/CAPRI) for Context-Aware Point-of-Interest Recommendation. RecBole works as a pip package inside this project, CAPRI is a separate Repository. 

#### Recbole 

* Inside the folder ```recbole_general_recs/dataset``` create a folder with the structure <\dataset name>_sample (e.g., foursquaretky_sample) & copy the files from your ```BASE_DIR/foursquaretky_dataset/processed_data_recbole``` into that folder. 

* Hyperparameter optimization: has already been done and saved to recbole_general_recs/config - if you wish to re-do it, cd to recbole_general_recs and run ```python3 config_hyperparameter_creator.py``` -- see hyper.test for the tested parameters

* cd back to the project's root directory, run: ```python3 recbole_general_recs/recbole_full_casestudy.py```
This creates a folder inside the BASE_DIR/<dataset> named "recommendations/BPR+timestamp including the config file that produced the recommendations, the general evaluation and the top_k_recommendations

* Note: In case of an error in Recbole, try:
```pip3 install ray```
```pip3 install "ray[tune]"```
In the recbole package in your virtual environment, comment out the line #from kmeans_pytorch import kmeans in the following path: recbole/model/general_recommender/ldiffrec.py



#### CAPRI
Follow steps 1-2 in the forked CAPRI repository. In the ```config.py```you can specify which contexts (Geographical, Social, Temporal, Interaction) each dataset features. For our study, social connections are not relevant, therefore we remove them from the data. Run the script ```main.py``` and choose the desired model and dataset. To reproduce the results, use "sum" fusion method. Create a folder with this exact structure in the ```recommendations``` subdirectory: ```<dataset>_sample-contextpoi-<model_name>-Jan-01-2024_09-00-00``` and manually transfer the Eval_ and Rec_ files for the respective model into this directory. Return to this repository, open the script ```capri_postprocessing.py```, specify the desired datasets and run the script to process the outputs (general evaluation and top-k recommendations) to be in line with those produced by RecBole.

### Popularity Calibration

* call both ```capri_postprocessing.py```and then ```postprocess_baseline_top_k.py```from the root directory. 
* call ```reranker.py```from the root directory. gridsearch = False since it is already included for foursquaretky for the best CP-parameters.

#### General Evaluation
The script ```offline_evaluation.ipynb```includes the full evaluation and plots. The evaluation metrics are found in ```evaluation_metrics.py```. 
