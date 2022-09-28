from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from category_encoders.one_hot import OneHotEncoder

    
#Creating our pipeline
sample_test_pipe = Pipeline([
    ('cat_encode', cat_encode),
    ('imputation', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('var', boruta),
    ('os', ADASYN()),
    ('ridge', RidgeClassifier(alpha=6.967))
    ], memory="./Cache/")

#Define a list of hyperparameters we want to tune
ada_search = {
    'os': [ADASYN()],
    'os__sampling_strategy': Real(0.2,1) #Search has to start at 0.2 otherwise the sampler raised an error
}

smote_search = {
    'os': [SMOTE()],
    'os__sampling_strategy': Real(0.2,1)
}


# define the search
search_sample = BayesSearchCV(
    sample_test_pipe,
    search_spaces=[(ada_search, 10), (smote_search, 10)],
    n_jobs=1,
    pre_dispatch= 1,
    scoring='roc_auc',
    cv=3,
    verbose=10)

search_sample.fit(X_train_reduced,y_train)

print("tuned hpyerparameters :(best parameters) ",search_sample.best_params_)
print("ROC AUC :",search_sample.best_score_)